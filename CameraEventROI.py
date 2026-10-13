"""
CameraEventROI.py
=================

Extiende EventCamera agregando ROI (Region Of Interest = region de interes)
y medicion por ventana de tiempo fija.

Por que hace falta
------------------
El EventCamera del repo cuenta TODOS los eventos del sensor. Para autoenfoque
eso no sirve: el ruido de todo el chip y las demas estrellas del campo tapan
la senal de tu estrella. Esta clase reusa la calibracion automatica de ROI de
tu script (los primeros 100 ms de eventos definen donde esta la estrella) y
cuenta solo adentro de ese recuadro.

Ademas expone `measure(duration_s)`, que cuenta durante un tiempo FIJO. Eso
es clave: si cada punto del barrido se mide durante un rato distinto, los
conteos no son comparables entre si.

Uso
---
    cam = RoiEventCamera(camera_id="433", half_size=50)
    cam.connect()
    cam.start_capture()
    cam.wait_until_calibrated(timeout_s=10)

    m = cam.measure(duration_s=1.0)
    print(m["rate"], "eventos/s en la ROI")
"""
import threading
import time
from collections import deque
from contextlib import nullcontext
from typing import Optional

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import RoiFilterAlgorithm, PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import EventLoop, Window
from metavision_core.event_io.raw_reader import initiate_device

from CameraEvent import EventCamera


def filter_outliers(x, y, n_std=2.0):
    """Descarta eventos espacialmente alejados del grueso del blob."""
    if len(x) < 3:
        return x, y
    mx, sx = x.mean(), x.std()
    my, sy = y.mean(), y.std()
    if sx == 0 or sy == 0:
        return x, y
    mask = (np.abs(x - mx) <= n_std * sx) & (np.abs(y - my) <= n_std * sy)
    if mask.sum() == 0:
        return x, y
    return x[mask], y[mask]


class RoiEventCamera(EventCamera):
    """
    Parametros nuevos respecto a EventCamera
    ----------------------------------------
    half_size : int
        Medio ancho de la ROI en pixeles. La ROI queda de (2*half_size+1) de
        lado. OJO: tiene que contener el blob desenfocado en los DOS extremos
        del barrido. Si el blob se sale por los bordes, el conteo baja de
        forma artificial y, como el autoenfoque busca un MINIMO, eso se
        disfraza de buen foco. Ante la duda, agrandala.
    calibration_time_us : int
        Cuanto tiempo de eventos se acumula para ubicar la estrella.
    outlier_n_std : float
        Umbral del filtro espacial de outliers.
    roi : tuple | None
        (x0, y0, x1, y1) fijo. Si lo entregas, se salta la calibracion.
    history : int
        Cuantas ventanas recientes se guardan en memoria.
    """

    def __init__(
        self,
        camera_id,
        delta_t=10000,
        accumulation_time_us=10000,
        activity_time_ths=100000,
        fps=25,
        show_live=True,
        half_size=50,
        calibration_time_us=1_000_000,      # 1 s, antes eran 100 ms
        calibration_min_events=100,         # minimo de eventos para confiar
        outlier_n_std=2.0,
        min_events=30, 
        roi=None,
        history=2000,
    ):
        super().__init__(
            camera_id=camera_id,
            delta_t=delta_t,
            accumulation_time_us=accumulation_time_us,
            activity_time_ths=activity_time_ths,
            fps=fps,
            show_live=show_live,
        )
        self.half_size = int(half_size)
        self.calibration_time_us = int(calibration_time_us)
        self.calibration_min_events = int(calibration_min_events)
        self.outlier_n_std = float(outlier_n_std)
        self.min_events = int(min_events)
        self.centroid = None            # (cx, cy) actual, o None
        self.show_centroid = False      # lo controla el boton
        self._device = None          # lo usa Grabar.py
        self.roi = roi
        self.calibrated = roi is not None
        self._calib_done = threading.Event()
        self._calib_request = threading.Event()   # el boton levanta esta
        self._calibrating = False
        self._n_poll = 0
        if roi is not None:
            self._calib_done.set()

        self._windows = deque(maxlen=int(history))
        self._window_seq = 0 
        

    # ------------------------------------------------------------------
    def wait_until_calibrated(self, timeout_s: float = 15.0) -> bool:
        """Bloquea hasta que la ROI quede definida. True si lo logro."""
        ok = self._calib_done.wait(timeout=timeout_s)
        if not ok and self.last_error:
            raise RuntimeError(f"[Camera {self.camera_id}] {self.last_error}")
        return ok

    def request_calibration(self, calibration_time_us=None, half_size=None):
        """Pide al bucle de captura que calibre la ROI ahora."""
        if calibration_time_us is not None:
            self.calibration_time_us = int(calibration_time_us)
        if half_size is not None:
            self.half_size = int(half_size)
        self.roi = None
        self.calibrated = False
        self.last_error = None
        self._calib_done.clear()
        self._calib_request.set()

    def clear_roi(self):
        """Apaga la ROI: vuelve a contar todo el sensor."""
        self._calib_request.clear()
        self._calibrating = False
        self.roi = None
        self.calibrated = False
        self._calib_done.clear()

    recalibrate = request_calibration   # compatibilidad con el nombre viejo


        # ------------------------------------------------------------------
    def window_mark(self) -> int:
        """Numero de secuencia de la ultima ventana guardada."""
        with self.lock:
            return self._window_seq

    def windows_since(self, mark: int) -> list:
        """Copia de las ventanas guardadas DESPUES de `mark`."""
        with self.lock:
            out = []
            for w in reversed(self._windows):
                if w["seq"] <= mark:
                    break
                out.append(dict(w))
        return out[::-1]

    # ------------------------------------------------------------------
    def measure(self, duration_s: float = 1.0, discard_s: float = 0.0) -> dict:
        """
        Cuenta eventos de la ROI durante un tiempo FIJO y devuelve un resumen.

        discard_s descarta el comienzo de la medicion, util si quieres
        asegurarte de no arrastrar la vibracion del motor.

        Devuelve
        --------
        dict con:
            n_windows : cuantas ventanas de integracion entraron
            total     : eventos totales en la ROI
            rate      : eventos por segundo  <-- la metrica de foco
            median    : mediana de eventos por ventana (robusta)
            spread    : dispersion espacial media del blob, en pixeles
            counts    : lista con el conteo de cada ventana
        """
        if discard_s > 0:
            time.sleep(discard_s)

        t0 = time.time()
        marca = self.window_mark()
        time.sleep(duration_s)
        nuevas = self.windows_since(marca)
        elapsed = time.time() - t0

        if not nuevas:
            return {
                "n_windows": 0,
                "total": 0,
                "rate": float("nan"),
                "median": float("nan"),
                "spread": float("nan"),
                "counts": [],
            }

        counts = np.array([w["count"] for w in nuevas], dtype=float)
        spreads = np.array([w["spread"] for w in nuevas], dtype=float)
        spreads = spreads[np.isfinite(spreads)]

        return {
            "n_windows": len(nuevas),
            "total": float(counts.sum()),
            "rate": float(counts.sum() / elapsed) if elapsed > 0 else float("nan"),
            "median": float(np.median(counts)),
            "spread": float(np.mean(spreads)) if spreads.size else float("nan"),
            "counts": counts.tolist(),
        }

    # ------------------------------------------------------------------
    def _capture_loop(self):
        """Cuenta dentro de la ROI y ademas muestra el Live con el
        recuadro dibujado encima."""
        try:
            destino = str(self.camera_id) if self.camera_id else ""
            # mv_iterator = EventsIterator(input_path=destino, delta_t=self.delta_t)
            device = initiate_device(destino)
            self._device = device                       #grabador
            mv_iterator = EventsIterator.from_device(device=device, delta_t=self.delta_t)
            height, width = mv_iterator.get_size()

            activity_filter = ActivityNoiseFilterAlgorithm(
                width, height, self.activity_time_ths
            )
            activity_buffer = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

            roi_filter = None
            roi_buffer = None
            if self.roi is not None:
                roi_filter, roi_buffer = self._build_roi(*self.roi)

            calib_start_t = None
            calib_xs, calib_ys = [], []

            # ---------- preparacion del Live ----------
            periodic_gen = None
            if self.show_live:
                periodic_gen = PeriodicFrameGenerationAlgorithm(
                    width, height,
                    accumulation_time_us=self.accumulation_time_us,
                    fps=self.fps,
                )

            win_ctx = (
                Window(f"Camera {self.camera_id} (ROI)", width, height,
                       Window.RenderMode.BGR)
                if self.show_live else nullcontext()
            )

            with win_ctx as window:

                if periodic_gen is not None:
                    def on_frame(ts, frame):
                        if self._calibrating:
                            pass
                        elif self.roi is not None:
                            x0, y0, x1, y1 = self.roi
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
                            cv2.putText(frame, "ROI", (x0, max(12, y0 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame, "sin ROI (sensor completo)", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        cv2.putText(frame, f"{self.camera_id}  ts={ts}", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        if self.show_centroid and self.centroid is not None:
                            px = int(round(self.centroid[0]))
                            py = int(round(self.centroid[1]))
                            cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                        window.show(frame)

                    periodic_gen.set_output_callback(on_frame)

                # ---------- bucle principal ----------
                for evs in mv_iterator:
                    
                    if self.stop_event.is_set():
                        break

                    if self.show_live:
                        EventLoop.poll_and_dispatch()

                    activity_filter.process_events(evs, activity_buffer)
                    clean = activity_buffer.numpy()

                    if periodic_gen is not None:
                        periodic_gen.process_events(activity_buffer)

                    if self.show_live and window.should_close():
                        self.stop_event.set()
                        break

                    # ---------- calibracion de la ROI ----------
                    # ---------- calibracion, solo si el boton la pidio ----------
                    if self._calib_request.is_set():
                        if not self._calibrating:
                            self._calibrating = True
                            calib_start_t = None
                            calib_xs, calib_ys, calib_n = [], [], 0
                            roi_filter, roi_buffer = None, None

                        if len(clean) > 0:
                            if calib_start_t is None:
                                calib_start_t = clean["t"][0]
                            calib_xs.append(clean["x"].copy())
                            calib_ys.append(clean["y"].copy())
                            calib_n += len(clean)

                            transcurrido = clean["t"][-1] - calib_start_t
                            listo_tiempo = transcurrido >= self.calibration_time_us
                            listo_datos = calib_n >= self.calibration_min_events
                            se_paso = transcurrido >= 5 * self.calibration_time_us

                            if listo_tiempo and (listo_datos or se_paso):
                                if calib_n < 3:
                                    self.last_error = (
                                        f"muy pocos eventos ({calib_n}) para calibrar"
                                    )
                                    self._calibrating = False
                                    self._calib_request.clear()
                                    self._calib_done.set()
                                    continue

                                ax = np.concatenate(calib_xs)
                                ay = np.concatenate(calib_ys)
                                ax, ay = filter_outliers(ax, ay, self.outlier_n_std)

                                cx, cy = float(ax.mean()), float(ay.mean())
                                x0 = max(0, int(cx - self.half_size))
                                y0 = max(0, int(cy - self.half_size))
                                x1 = min(width - 1, int(cx + self.half_size))
                                y1 = min(height - 1, int(cy + self.half_size))

                                roi_filter, roi_buffer = self._build_roi(x0, y0, x1, y1)
                                self.roi = (x0, y0, x1, y1)
                                self.calibrated = True
                                self._calibrating = False
                                self._calib_request.clear()
                                self._calib_done.set()
                                print(f"[Camera {self.camera_id}] ROI calibrada con "
                                      f"{calib_n} eventos: {self.roi}")
                                calib_xs, calib_ys = [], []
                        continue    # mientras calibra no se cuenta

                    # ---------- conteo ----------
                    if self.calibrated and roi_filter is not None:
                        roi_filter.process_events(activity_buffer, roi_buffer)
                        eventos = roi_buffer.numpy()
                    else:
                        eventos = clean          # sin ROI: sensor completo

                    n_raw = len(eventos)         # sin filtrar outliers (como el ref.)
                    n = n_raw
                    spread = float("nan")
                    cx = cy = float("nan")

                    if n_raw > 0:
                        ex, ey = filter_outliers(eventos["x"], eventos["y"],
                                                 self.outlier_n_std)
                        n = len(ex)
                        if n >= 3:
                            spread = float(np.sqrt(ex.var() + ey.var()))

                        if n >= self.min_events:
                            cx, cy = float(ex.mean()), float(ey.mean())
                            self.centroid = (cx, cy)
                        else:
                            self.centroid = None    # muy pocos eventos: no confiable
                    else:
                        self.centroid = None

                    with self.lock:
                        self.event_counter += n
                        self._window_seq += 1
                        self._windows.append({
                            "seq": self._window_seq,
                            "t": time.time(),
                            "count": n,
                            "raw": n_raw,
                            "spread": spread,
                            "cx": cx,
                            "cy": cy,
                        })

        except Exception as e:
            self.last_error = str(e)
            self._calib_done.set()
            print(f"[Camera {self.camera_id}] ERROR en captura: {e}")
        finally:
            self._device = None            

    # ------------------------------------------------------------------
    @staticmethod
    def _build_roi(x0, y0, x1, y1):
        f = RoiFilterAlgorithm(
            x0=x0, y0=y0, x1=x1, y1=y1, output_relative_coordinates=False
        )
        return f, RoiFilterAlgorithm.get_empty_output_buffer()
