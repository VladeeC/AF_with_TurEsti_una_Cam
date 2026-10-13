import threading
import time
import numpy as np 
from AutoFocus import AutoFocus
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit, QLineEdit
)

# Colores del indicador (el recuadro del terminal)
COLOR_IDLE  = "#9e9e9e"   # gris  -> sin hacer nada
COLOR_OK    = "#2e7d32"   # verde -> todo correcto
COLOR_BUSY  = "#f9a825"   # ambar -> trabajando
COLOR_ERROR = "#c62828"   # rojo  -> stop o error

def mediana_movil(y, k=11):
    """Mediana movil robusta. k impar; si es par se le suma 1."""
    y = np.asarray(y, dtype=float)
    if k % 2 == 0:
        k += 1
    if len(y) < k or k < 3:
        return y
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    vistas = np.lib.stride_tricks.sliding_window_view(yp, k)
    return np.median(vistas, axis=1)

def sigma_r_movil(x, y, k=101):
    """Sigma radial en una ventana deslizante de k muestras."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < k or k < 3:
        return None, 0
    ker = np.ones(k) / k
    mx  = np.convolve(x,     ker, mode="valid")
    mx2 = np.convolve(x * x, ker, mode="valid")
    my  = np.convolve(y,     ker, mode="valid")
    my2 = np.convolve(y * y, ker, mode="valid")
    var = np.maximum(mx2 - mx**2, 0) + np.maximum(my2 - my**2, 0)
    return np.sqrt(var), k // 2


def acomodar_figuras(margen=0, pantalla=0):
    """Reparte las figuras abiertas en una grilla 2x2 (+ extras apiladas).
    pantalla: 0 = principal, 1 = secundaria, etc."""
    import matplotlib.pyplot as plt
    from PyQt5.QtWidgets import QApplication

    pantallas = QApplication.instance().screens()
    idx = min(pantalla, len(pantallas) - 1)
    geo = pantallas[idx].availableGeometry()
    X0, Y0, W, H = geo.x(), geo.y(), geo.width(), geo.height()

    w, h = W // 2 - margen, H // 2 - margen
    posiciones = [
        (X0,         Y0),           # sup izq
        (X0 + W // 2, Y0),          # sup der
        (X0,         Y0 + H // 2),  # inf izq
        (X0 + W // 2, Y0 + H // 2), # inf der
    ]

    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        try:
            win = fig.canvas.manager.window
        except AttributeError:
            continue
        if i < 4:
            x, y = posiciones[i]
        else:
            # las que sobran se apilan desplazadas en el centro
            x = X0 + W // 4 + (i - 4) * 30
            y = Y0 + H // 4 + (i - 4) * 30
        win.setGeometry(x, y, w, h)

class CameraUI(QWidget):
    # Señal para escribir en la UI desde hilos secundarios (Qt no permite
    # tocar widgets fuera del hilo principal; esto evita cierres aleatorios)
    log_signal = pyqtSignal(str, str)
    roi_text_signal = pyqtSignal(str)
    plot_signal = pyqtSignal() 
    af_plot_signal = pyqtSignal()

    def __init__(self, station, sweep_cfg=None, noise_cfg=None, grabador=None, af_cfg=None):
        super().__init__()
        self.grabador=grabador
        self.station = station
        self.af_cfg=af_cfg or {}
        self.last_af = None
        self.sweep_cfg = sweep_cfg or {"start": 0, "end": 7200, "step": 150, "settle_s": 0.30}
        self.noise_cfg = noise_cfg or dict(self.sweep_cfg, settle_s=1.00)
        self.perfil_ruido = None

        self.last_result = None              # datos del ultimo barrido -> Plot
        self.abort_event = threading.Event() # bandera de parada de emergencia
        self._worker = None                  # hilo de la tarea en curso
        self.perfil_ruido = None

        self.setWindowTitle("Autofocus Camera Manager")
        self.resize(760, 380)

        self.log_signal.connect(self._on_log)
        self.roi_text_signal.connect(self._on_roi_text)
        self.plot_signal.connect(self.run_plot) 
        self.af_plot_signal.connect(self.run_plot_autofocus)
        self._build_ui()
        self.set_status("idle", COLOR_IDLE)
        

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        box = QGroupBox(f"Cell {self.station.station_id}")
        box_layout = QVBoxLayout(box)

        # ---- fila superior: info a la izquierda, terminal a la derecha ----
        top = QHBoxLayout()

        info_col = QVBoxLayout()
        self.info_label = QLabel(
            f"Lens serial: {self.station.serial_number}\n"
            f"Camera ID: {self.station.camera.camera_id}"
        )
        self.status_label = QLabel("Estado: idle")
        info_col.addWidget(self.info_label)
        info_col.addWidget(self.status_label)
        info_col.addStretch()
        top.addLayout(info_col, 1)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        top.addWidget(self.log_box, 2)

        box_layout.addLayout(top)

        # ---- botonera 3x3 (igual a la imagen) ----
        grid = QGridLayout()
        self.btn_connect   = QPushButton("Conectar cámara")
        self.btn_grabar = QPushButton("GRABAR")
        self.btn_live      = QPushButton("Live")
        self.btn_stop      = QPushButton("Stop")
        self.btn_reset     = QPushButton("Reset")
        self.btn_barrido   = QPushButton("Barrido")
        self.btn_plot      = QPushButton("Plot")
        self.btn_autofocus = QPushButton("Autofocus")
        self.btn_roi = QPushButton("Activar ROI")
        self.btn_centroid = QPushButton("Ver centroide")
        self.btn_ruido = QPushButton("Obtener ruido")
        self.btn_barrido_plot = QPushButton("Barrido y plot")

        grid.addWidget(self.btn_connect,   0, 0)
        grid.addWidget(self.btn_live,      0, 1)
        grid.addWidget(self.btn_stop,      0, 2)
        grid.addWidget(self.btn_reset,     1, 0)
        grid.addWidget(self.btn_barrido,   1, 1)
        grid.addWidget(self.btn_plot,      1, 2)
        grid.addWidget(self.btn_roi,       2, 0)
        grid.addWidget(self.btn_centroid, 2, 1)
        grid.addWidget(self.btn_autofocus, 2, 2)
        grid.addWidget(self.btn_ruido, 3, 0)
        grid.addWidget(self.btn_grabar, 3,1)
        grid.addWidget(self.btn_barrido_plot, 3, 2)
        box_layout.addLayout(grid)
        fila_grab = QHBoxLayout()
        self.nombre_grab = QLineEdit()
        self.nombre_grab.setPlaceholderText("nombre del .raw (vacío = fecha y hora)")
        fila_grab.addWidget(QLabel("Archivo:"))
        fila_grab.addWidget(self.nombre_grab)
        box_layout.addLayout(fila_grab)

        main_layout.addWidget(box)

        self.btn_centroid.clicked.connect(self.toggle_centroid)
        self.btn_connect.clicked.connect(self.connect_camera)
        self.btn_live.clicked.connect(self.start_live)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_reset.clicked.connect(self.reset_lens)
        self.btn_barrido.clicked.connect(lambda _: self.run_barrido())
        self.btn_plot.clicked.connect(self.run_plot)
        self.btn_autofocus.clicked.connect(self.run_autofocus)
        self.btn_roi.clicked.connect(self.toggle_roi)
        self.btn_ruido.clicked.connect(self.obtener_ruido)
        self.btn_barrido_plot.clicked.connect(lambda _: self.barrido_y_plot())
        self.btn_grabar.clicked.connect(self.toggle_grabacion)

    def toggle_grabacion(self):
        import os
        if self.grabador is None:
            self.set_status("no hay grabador configurado", COLOR_ERROR)
            return
        try:
            if self.grabador.grabando:
                info = self.grabador.stop()
                self.btn_grabar.setText("Grabar .raw")
                self.nombre_grab.setEnabled(True)
                self.set_status(
                    f"guardado: {os.path.basename(info['path'])} "
                    f"({info['duracion_s']:.1f} s, {info['mb']:.1f} MB)", COLOR_OK)
            else:
                nombre = self.nombre_grab.text().strip() or None
                ruta = self.grabador.start(nombre)
                self.btn_grabar.setText("■ Detener grabación")
                self.nombre_grab.setEnabled(False)   # no se cambia a mitad de camino
                self.set_status(f"GRABANDO -> {os.path.basename(ruta)}", COLOR_BUSY)
        except Exception as e:
            self.set_status(f"error grabación: {e}", COLOR_ERROR)

    # ------------------------------------------------------- estado / log --

    def set_status(self, text, color=""):
        """Cambia 'Estado:', escribe en el terminal y (opcional) el color."""
        self.log_signal.emit(text, color)

    def log(self, text):
        """Solo escribe en el terminal, sin tocar el estado."""
        self.log_signal.emit("\x00" + text, "")

    def _on_log(self, text, color):
        if text.startswith("\x00"):
            self.log_box.append(f"    {text[1:]}")
        else:
            self.status_label.setText(f"Estado: {text}")
            self.log_box.append(f"[Cell {self.station.station_id}] {text}")

        if color:
            self.log_box.setStyleSheet(
                f"QTextEdit {{ border: 2px solid {color}; }}"
            )
    def _on_roi_text(self, texto):
        self.btn_roi.setText(texto)

    def toggle_centroid(self):
        cam = self.station.camera

        if not hasattr(cam, "show_centroid"):
            self.set_status("esta cámara no calcula centroide", COLOR_ERROR)
            return

        cam.show_centroid = not cam.show_centroid
        if cam.show_centroid:
            self.btn_centroid.setText("Ocultar centroide")
            self.set_status("centroide visible", COLOR_OK)
        else:
            self.btn_centroid.setText("Ver centroide")
            self.set_status("centroide oculto", COLOR_IDLE)

    def toggle_roi(self):
        cam = self.station.camera

        if not hasattr(cam, "request_calibration"):
            self.set_status("esta cámara no soporta ROI", COLOR_ERROR)
            return

        # ya hay ROI -> apagarla
        if cam.calibrated:
            cam.clear_roi()
            self.roi_text_signal.emit("Activar ROI")
            self.set_status("ROI apagada — contando todo el sensor", COLOR_IDLE)
            return

        def worker():
            try:
                if not (cam.capture_thread and cam.capture_thread.is_alive()):
                    self.set_status("abre el Live antes de calibrar la ROI",
                                    COLOR_ERROR)
                    return

                segundos = cam.calibration_time_us / 1e6
                self.set_status(f"calibrando ROI ({segundos:.1f} s)...", COLOR_BUSY)
                cam.request_calibration()
                cam._calib_done.wait(timeout=segundos * 5 + 10)

                if cam.roi is None:
                    motivo = cam.last_error or "sin eventos suficientes"
                    self.set_status(f"ROI no calibró: {motivo}", COLOR_ERROR)
                    return

                self.roi_text_signal.emit("Quitar ROI")
                self.set_status(f"ROI activa: {cam.roi}", COLOR_OK)
            except Exception as e:
                self.set_status(f"error ROI: {e}", COLOR_ERROR)

        self._launch(worker)

    def _launch(self, fn):
        """Lanza una tarea en segundo plano si no hay otra corriendo."""
        if self._worker and self._worker.is_alive():
            self.set_status("hay una tarea en curso, espera a que termine")
            return
        self._worker = threading.Thread(target=fn, daemon=True)
        self._worker.start()

    # ------------------------------------------------- CONECTAR CAMARA -----
    @staticmethod
    def _listar_seriales():
        """Pide al HAL (Hardware Abstraction Layer = capa de abstracción de
        hardware) de Metavision los seriales de las cámaras conectadas."""
        try:
            from metavision_hal import DeviceDiscovery
            return [str(s) for s in DeviceDiscovery.list()]
        except Exception:
            return []

    def connect_camera(self):
        def worker():
            try:
                self.abort_event.clear()
                cam_id = str(self.station.camera.camera_id)
                self.set_status("verificando ID de la cámara...", COLOR_BUSY)

                seriales = self._listar_seriales()
                if seriales:
                    self.log(f"cámaras detectadas: {seriales}")
                    if not any(cam_id in s for s in seriales):
                        self.set_status(
                            f"ID incorrecto: se esperaba {cam_id}", COLOR_ERROR
                        )
                        return
                    self.log(f"ID {cam_id} verificado")
                else:
                    self.log("no se pudo leer la lista de seriales; "
                             "se continúa sin verificar")

                self.station.camera.connect()
                self.station.camera.start_capture()   # live automatico
                self.set_status("cámara conectada — live iniciado", COLOR_OK)
                

            except Exception as e:
                self.set_status(f"error cámara: {e}", COLOR_ERROR)

        self._launch(worker)

    # ------------------------------------------------------------- LIVE ----
    def start_live(self):
        def worker():
            cam = self.station.camera
            try:
                self.abort_event.clear()
                if not cam.connected:
                    cam.connect()

                if cam.capture_thread and cam.capture_thread.is_alive():
                    self.set_status("el live ya está abierto", COLOR_OK)
                    return

                cam.stop()              # limpia el hilo viejo si quedó colgado
                cam.start_capture()
                self.set_status("live reabierto", COLOR_OK)
            except Exception as e:
                self.set_status(f"error live: {e}", COLOR_ERROR)

        self._launch(worker)

    # ------------------------------------------------------------- STOP ----
    def stop_all(self):
        """Parada de emergencia: corta el barrido y cierra el live."""
        self.abort_event.set()
        self.set_status("STOP — deteniendo sistema...", COLOR_ERROR)

        def worker():
            try:
                self.station.camera.stop()
                self.set_status("sistema detenido", COLOR_ERROR)
            except Exception as e:
                self.set_status(f"error stop: {e}", COLOR_ERROR)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------ RESET ----
    def reset_lens(self):
        def worker():
            try:
                self.abort_event.clear()
                if not self.station.lens.connected:
                    self.station.lens.connect()
                self.set_status("reseteando lente...", COLOR_BUSY)
                self.station.lens.reset(fast_step=1000, fast_settle=0.5)
                self.set_status("reset completo", COLOR_OK)
            except Exception as e:
                self.set_status(f"error reset: {e}", COLOR_ERROR)

        self._launch(worker)

    # ---------------------------------------------------------- BARRIDO --ss-s---------------------------------------------
    def run_barrido(self, cfg=None):
        def worker():
            cfg_local = cfg or self.sweep_cfg
            lens = self.station.lens
            cam = self.station.camera
            try:
                ...
                self.abort_event.clear()
                self.set_status("barrido iniciado", COLOR_BUSY)

                if not lens.connected:
                    lens.connect()
                if not cam.connected:
                    cam.connect()
                if not (cam.capture_thread and cam.capture_thread.is_alive()):
                    cam.start_capture()

                if getattr(cam, "roi", None) is not None:
                    self.log(f"barrido con ROI = {cam.roi}")
                else:
                    self.log("barrido SIN ROI (sensor completo)")

                lens.reset(fast_step=1000, fast_settle=0.5)
                time.sleep(0.1)
                start, end = int(cfg_local["start"]), int(cfg_local["end"])
                step = int(cfg_local["step"])
                settle_s = float(cfg_local["settle_s"])
                discard_s = float(cfg_local.get("discard_s", 0.0))

                direction = 1 if end >= start else -1
                positions = list(range(start, end, step * direction))
                if not positions or positions[-1] != end:
                    positions.append(end)

               
                t0_barrido = time.time()
                ventanas = []
                step_events, event_gen, step_targets = [], [], []
                total = len(positions)
                # ==================================================

                for i, target in enumerate(positions, start=1):
                    if self.abort_event.is_set():
                        self.set_status("barrido abortado por STOP", COLOR_ERROR)
                        break

                    cmd = f"M{target:04d}#"
                    cam.reset_counter()

                    with cam.lock:
                        marca = cam._window_seq

                    reached = lens.move_to(target)
                    time.sleep(settle_s)
                    event_count = cam.get_and_reset_counter()

                    with cam.lock:
                        nuevas = [w for w in cam._windows if w.get("seq", 0) > marca]

                    for w in nuevas:
                        ventanas.append({
                            "t_rel":  w["t"] - t0_barrido,
                            "count":  w["count"],
                            "raw":    w.get("raw", w["count"]),
                            "spread": w.get("spread", float("nan")),   # <-- FALTABA
                            "target": target,
                            "step_index": i,
                            "cx": w.get("cx", float("nan")),
                            "cy": w.get("cy", float("nan")),
                        })

                    cxs = [w["cx"] for w in nuevas
                           if np.isfinite(w.get("cx", np.nan))]
                    cys = [w["cy"] for w in nuevas
                           if np.isfinite(w.get("cy", np.nan))]
                    cx_mean = float(np.mean(cxs)) if cxs else float("nan")
                    cy_mean = float(np.mean(cys)) if cys else float("nan")

                    step_events.append({
                        "index": i,
                        "cmd": cmd,
                        "target": target,
                        "reached": reached,
                        "event_count": event_count,
                        "n_windows": len(nuevas),
                        "timestamp": time.time(),
                        "cx_mean": cx_mean,
                        "cy_mean": cy_mean,
                        "n_centroides": len(cxs),
                    })
                    event_gen.append(event_count)
                    step_targets.append(target)

                    self.log(f"[{i:03d}/{total:03d}] step={target:04d} "
                             f"({cmd}) -> eventCount={event_count} "
                             f"| ventanas={len(nuevas)}")

                self.last_result = {
                    "station_id": self.station.station_id,
                    "serial_number": self.station.serial_number,
                    "step_events": step_events,
                    "event_gen": event_gen,
                    "step_targets": step_targets,
                    "windows": ventanas,
                    "delta_t_s": cam.delta_t / 1e6,
                    
                }

                cb = getattr(self, "_post_barrido", None)
                if cb is not None:
                    self._post_barrido = None
                    cb()

                if not self.abort_event.is_set():
                    self.set_status(
                        f"barrido terminado | steps={len(step_events)} | "
                        f"ventanas={len(ventanas)} | "
                        f"eventos totales={sum(event_gen)}", COLOR_OK
                    )
                    self.log(f"event_gen = {event_gen}")

            except Exception as e:
                self.set_status(f"error barrido: {e}", COLOR_ERROR)


        self._launch(worker)

#===================================================================#
#=================   barrido y graficar altiro      ========= ======#
#===================================================================#


    def barrido_y_plot(self):
        """Corre el barrido y grafica automaticamente al terminar."""
        def cuando_termine():
            self.plot_signal.emit()      # el plot va al hilo principal

        self._post_barrido = cuando_termine
        self.run_barrido()

#===================================================================#
#===================================Obtener ruido===================#
#===================================================================#
    def obtener_ruido(self):
        """Barrido SIN turbulencia para medir el ruido mecanico.
        Usa noise_cfg (settle_s mas largo) sin tocar la config del barrido."""
        s = self.noise_cfg["settle_s"]
        n = len(range(self.noise_cfg["start"],
                      self.noise_cfg["end"] + 1,
                      self.noise_cfg["step"]))
        self.log(f"=== RUIDO: apaga la turbulencia. "
                 f"{n} steps x {s:.2f} s ≈ {n*s + 25:.0f} s ===")

        def cuando_termine():
            if not self.last_result or not self.last_result.get("windows"):
                self.set_status("no se pudo guardar el perfil", COLOR_ERROR)
                return

            perfil = self._perfil_por_step(self.last_result["windows"])
            k = len(perfil["sigma_c"])

            if k == 0:
                self.perfil_ruido = None
                self.set_status("perfil VACÍO: muy pocos centroides. "
                                "sube settle_s o baja min_events", COLOR_ERROR)
                return

            self.perfil_ruido = perfil
            self.last_result["es_ruido"] = True        # <-- marca
            self.set_status(f"perfil de ruido guardado ({k}/{n} steps)", COLOR_OK)
            self.log("ahora enciende la turbulencia y corre Barrido")

        self._post_barrido = cuando_termine
        self.run_barrido(cfg=self.noise_cfg)

    @staticmethod
    def _perfil_por_step(ventanas):
        """sigma_c, spread y piso de cada step."""
        sc, sp, pi = {}, {}, {}
        agr = {}
        for w in ventanas:
            agr.setdefault(w["target"], []).append(w)

        for x, ws in agr.items():
            cx = np.array([w["cx"] for w in ws
                           if np.isfinite(w.get("cx", np.nan))])
            cy = np.array([w["cy"] for w in ws
                           if np.isfinite(w.get("cy", np.nan))])
            spv = np.array([w["spread"] for w in ws
                            if np.isfinite(w.get("spread", np.nan))])
            nnv = np.array([w["count"] for w in ws
                            if np.isfinite(w.get("spread", np.nan))], dtype=float)

            if len(cx) >= 5:
                dx = cx - np.median(cx)
                dy = cy - np.median(cy)
                mad_x = float(np.median(np.abs(dx)) * 1.4826)
                mad_y = float(np.median(np.abs(dy)) * 1.4826)
                sc[x] = float(np.sqrt(mad_x**2 + mad_y**2))
            if spv.size:
                sp[x] = float(np.median(spv))
                ok = nnv > 0
                if ok.any():
                    pi[x] = float(np.mean(spv[ok] / np.sqrt(nnv[ok])))

        return {"sigma_c": sc, "spread": sp, "piso": pi}
    
    # ------------------------------------------------------------- PLOT ----
    # ------------------------------------------------------------- PLOT ----
    
    
    def run_plot(self):
        """Se ejecuta en el hilo principal a proposito: matplotlib no es
        seguro desde hilos secundarios. Dibuja las 5 graficas en una sola
        figura 3x2 (el sexto hueco queda oculto)."""
        if not self.last_result or not self.last_result.get("windows"):
            self.set_status("no hay datos: corre primero el Barrido", COLOR_BUSY)
            return

        try:
            import matplotlib.pyplot as plt

            res = self.last_result
            ventanas = res["windows"]
            dt_s = res["delta_t_s"]
            dt_ms = dt_s * 1000

            t   = np.array([w["t_rel"]  for w in ventanas])
            c   = np.array([w["count"]  for w in ventanas], dtype=float)
            pos = np.array([w["target"] for w in ventanas])

            K_MED     = max(5,  int(round(0.5 / dt_s)) | 1)
            SIGMA_WIN = max(11, int(round(1.0 / dt_s)) | 1)

            sp_all = np.array([w.get("spread", np.nan) for w in ventanas])
            n_sp = int(np.isfinite(sp_all).sum())
            n_cx = int(np.isfinite(
                np.array([w.get("cx", np.nan) for w in ventanas])).sum())
            self.log(f"ventanas={len(ventanas)} | con spread={n_sp} | "
                     f"con centroide={n_cx} | Δt={dt_ms:.0f} ms")

            # ================= figura unica 3x2 =================
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            ax1, ax2 = axes[0, 0], axes[0, 1]
            ax3, ax4 = axes[1, 0], axes[1, 1]
            ax5, ax6 = axes[2, 0], axes[2, 1]
            ax6.axis("off")          # hueco sobrante

            titulo = (f"Barrido | Δt={dt_ms:.0f} ms | "
                      f"{len(ventanas)} ventanas | "
                      f"steps={len(res.get('step_events', []))}")
            fig.suptitle(titulo, fontsize=13)

            # ---------------- 1. Event count en el tiempo ----------------
            ax1.plot(t, c, linewidth=0.5, alpha=0.25, color="gray",
                     label=f"Crudo ({dt_ms:.0f} ms)")
            ax1.plot(t, mediana_movil(c, k=K_MED), linewidth=1.6,
                     color="tab:orange",
                     label=f"Mediana móvil ({K_MED * dt_s:.2f} s)")
            ax1.set_xlabel("Tiempo [s]")
            ax1.set_ylabel("Eventos por ventana")
            ax1.set_title(f"Event count en la ROI (ventana = {dt_ms:.0f} ms)")
            ax1.grid(True)

            ax1b = ax1.twinx()
            ax1b.plot(t, pos, color="tab:blue", linewidth=1.0, alpha=0.4,
                      label="Posición lente")
            ax1b.set_ylabel("Posición del lente [step]")

            lin = ax1.get_lines() + ax1b.get_lines()
            ax1.legend(lin, [l.get_label() for l in lin],
                       loc="upper left", fontsize=8)

            # ---------------- datos de centroide ----------------
            dat = [w for w in ventanas if np.isfinite(w.get("cx", np.nan))]

            if not dat:
                self.log("centroide: ninguna ventana válida "
                         "(¿min_events muy alto?)")
                for a, txt in ((ax2, "sin centroide"), (ax3, "sin centroide")):
                    a.text(0.5, 0.5, txt, ha="center", va="center",
                           transform=a.transAxes, color="gray")
                    a.set_xticks([]); a.set_yticks([])
            else:
                tc  = np.array([w["t_rel"]  for w in dat])
                cx  = np.array([w["cx"]     for w in dat])
                cy  = np.array([w["cy"]     for w in dat])
                p_c = np.array([w["target"] for w in dat])

                dx = cx - cx.mean()
                dy = cy - cy.mean()
                sr = float(np.sqrt(dx.var() + dy.var()))

                sp_d = np.array([w.get("spread", np.nan) for w in dat])
                nn_d = np.array([w["count"] for w in dat], dtype=float)
                ok = np.isfinite(sp_d) & (nn_d > 0)
                piso = (float(np.mean(sp_d[ok] / np.sqrt(nn_d[ok])))
                        if ok.any() else float("nan"))

                # ------------ 2. Movimiento del centroide ------------
                ax2.plot(tc, dx, linewidth=0.8, label=f"X (σ={dx.std():.3f} px)")
                ax2.plot(tc, dy, linewidth=0.8, label=f"Y (σ={dy.std():.3f} px)")
                ax2.set_xlabel("Tiempo [s]")
                ax2.set_ylabel("Movimiento del centroide [píxeles]")
                ax2.set_title("Movimiento del centroide")
                ax2.grid(True)
                ax2.legend(fontsize=8)

                # ------------ 3. Estabilidad (sigma movil) ------------
                if len(dat) <= SIGMA_WIN:
                    self.log(f"σ móvil: solo {len(dat)} muestras, se necesitan "
                             f"más de {SIGMA_WIN}")
                    ax3.text(0.5, 0.5, "muestras insuficientes",
                             ha="center", va="center", transform=ax3.transAxes,
                             color="gray")
                    ax3.set_xticks([]); ax3.set_yticks([])
                else:
                    sig, off = sigma_r_movil(dx, dy, SIGMA_WIN)
                    t_sig = tc[off:off + len(sig)]
                    p_sig = p_c[off:off + len(sig)]
                    i_min = int(np.argmin(sig))
                    i_max = int(np.argmax(sig))

                    self.log(f"centroide: σ_r global = {sr:.3f} px | "
                             f"más estable en t={t_sig[i_min]:.2f}s "
                             f"(step={p_sig[i_min]}, σ_r={sig[i_min]:.3f} px)")
                    if np.isfinite(piso):
                        self.log(f"piso de ruido ≈ {piso:.3f} px | "
                                 f"razón σ_r/piso = {sr / piso:.2f}")
                    else:
                        self.log("piso de ruido: no calculable (sin spread)")

                    ax3.plot(t_sig, sig, linewidth=1.0, color="tab:blue")
                    ax3.axhline(sr, color="gray", linestyle="--", linewidth=0.8,
                                label=f"σ_r global = {sr:.3f} px")
                    if np.isfinite(piso):
                        ax3.axhline(piso, color="tab:red", linestyle=":",
                                    linewidth=1.2,
                                    label=f"piso ≈ {piso:.3f} px")
                    ax3.plot(t_sig[i_min], sig[i_min], "go", markersize=7,
                             label=f"más estable ({sig[i_min]:.3f} px)")
                    ax3.plot(t_sig[i_max], sig[i_max], "ro", markersize=7,
                             label=f"menos estable ({sig[i_max]:.3f} px)")
                    ax3.set_xlabel("Tiempo [s]")
                    ax3.set_ylabel("σ_r local [píxeles]")
                    ax3.set_title(f"Estabilidad del centroide — ventana móvil "
                                  f"{SIGMA_WIN} muestras (~{SIGMA_WIN*dt_s:.1f} s)")
                    ax3.grid(True)

                    ax3b = ax3.twinx()
                    ax3b.plot(t_sig, p_sig, color="tab:gray",
                              linewidth=1.0, alpha=0.5)
                    ax3b.set_ylabel("Posición del lente [step]")
                    ax3.legend(loc="upper left", fontsize=8)

            # ---------------- sigma_c por step ----------------
            sc_step = {}
            for w in ventanas:
                if np.isfinite(w.get("cx", np.nan)):
                    sc_step.setdefault(w["target"], []).append((w["cx"], w["cy"]))

            xs_f, sc_f, n_f = [], [], []
            for x in sorted(sc_step):
                arr = np.array(sc_step[x])
                if len(arr) >= 10:
                    xs_f.append(x)
                    sc_f.append(float(np.sqrt(arr[:, 0].var() + arr[:, 1].var())))
                    n_f.append(len(arr))

            if len(xs_f) < 5:
                self.log(f"σ_c vs posición: solo {len(xs_f)} steps válidos")
                for a in (ax4, ax5):
                    a.text(0.5, 0.5, "steps insuficientes", ha="center",
                           va="center", transform=a.transAxes, color="gray")
                    a.set_xticks([]); a.set_yticks([])
            else:
                sc_f = np.array(sc_f)
                i_foco = int(np.argmin(sc_f))
                self.log(f"FOCO por σ_c: step={xs_f[i_foco]} "
                         f"(σ_c={sc_f[i_foco]:.3f} px) | "
                         f"contraste = {sc_f.max()/sc_f.min():.2f}x")

                # ---- minimo por parabola local ----
                foco_par = None
                if len(sc_f) >= 7:
                    k = 5
                    sc_smooth = np.convolve(sc_f, np.ones(k) / k, mode="same")
                    h = k // 2
                    sc_smooth[:h] = np.inf
                    sc_smooth[-h:] = np.inf
                    i0 = int(np.argmin(sc_smooth))

                    i_ini = max(0, i0 - 3)
                    i_fin = min(len(sc_f), i0 + 4)
                    xv = np.asarray(xs_f[i_ini:i_fin], dtype=float)
                    yv = sc_f[i_ini:i_fin]

                    if len(xv) >= 5:
                        a, b, _ = np.polyfit(xv, yv, 2)
                        if a > 0:
                            cand = -b / (2 * a)
                            if xv.min() <= cand <= xv.max():
                                foco_par = cand
                                self.log(f"FOCO por parábola local ≈ "
                                         f"step {foco_par:.0f}")

                # ------------ 4. sigma_c vs posicion ------------
                ax4.plot(xs_f, sc_f, marker="o", markersize=3,
                         color="tab:purple", label="σ_c medido")
                ax4.plot(xs_f[i_foco], sc_f[i_foco], "ko", markersize=9,
                         label=f"foco: step {xs_f[i_foco]}")

                if foco_par is not None:
                    ax4.axvline(foco_par, color="tab:red", linestyle="--",
                                linewidth=1.2,
                                label=f"foco parábola: {foco_par:.0f}")

                # ---- resta del ruido mecanico ----
                ruido_sc, ruido_pi = {}, {}
                hay_perfil = (not res.get("es_ruido", False)
                              and self.perfil_ruido is not None)

                if res.get("es_ruido", False):
                    self.log("esta es la corrida de RUIDO: no se resta nada")
                elif self.perfil_ruido is None:
                    self.log("sin perfil de ruido: corre «Obtener ruido» primero")
                else:
                    ruido_sc = self.perfil_ruido["sigma_c"]
                    ruido_pi = self.perfil_ruido["piso"]
                    lim, n_restados, n_descartados = [], 0, 0

                    for x, s in zip(xs_f, sc_f):
                        r = ruido_sc.get(x)
                        if r is None:
                            lim.append(s)
                            continue
                        pi_r = ruido_pi.get(x, 0.0)
                        motor2 = max(r**2 - pi_r**2, 0.0)
                        if motor2 >= s**2:
                            lim.append(np.nan)
                            n_descartados += 1
                            continue
                        lim.append(float(np.sqrt(s**2 - motor2)))
                        n_restados += 1

                    lim = np.array(lim, dtype=float)

                    if n_descartados:
                        self.log(f"{n_descartados} steps descartados "
                                 f"(ruido ≥ señal)")

                    if np.isfinite(lim).any():
                        i_lim = int(np.nanargmin(lim))
                        self.log(f"FOCO tras restar ruido: step={xs_f[i_lim]} "
                                 f"(σ_c={lim[i_lim]:.3f} px) | "
                                 f"{n_restados}/{len(xs_f)} steps con perfil")
                        ax4.plot(xs_f, lim, marker="^", markersize=3,
                                 color="tab:orange",
                                 label="σ_c sin ruido mecánico")
                        ax4.plot(xs_f[i_lim], lim[i_lim], "ks", markersize=9,
                                 label=f"foco limpio: step {xs_f[i_lim]}")
                    else:
                        self.log("resta inutilizable: todos los steps descartados")

                ax4.set_xlabel("Posición del lente [step]")
                ax4.set_ylabel("σ_c del centroide [píxeles]")
                ax4.set_title("Estabilidad del centroide vs posición — foco")
                ax4.grid(True)
                ax4.legend(fontsize=8)

                # ------------ 5. señal vs ruido mecanico ------------
                if hay_perfil:
                    motor_curve = []
                    for x in xs_f:
                        r = ruido_sc.get(x)
                        if r is None:
                            motor_curve.append(np.nan)
                            continue
                        pi_r = ruido_pi.get(x, 0.0)
                        motor_curve.append(np.sqrt(max(r**2 - pi_r**2, 0.0)))
                    motor_curve = np.asarray(motor_curve)

                    ax5.plot(xs_f, sc_f, "o-", markersize=3,
                             label="σ_c medido")
                    ax5.plot(xs_f, motor_curve, "o-", markersize=3,
                             label="componente mecánica estimada")
                    ax5.set_xlabel("Posición del lente [step]")
                    ax5.set_ylabel("σ [píxeles]")
                    ax5.set_title("Señal medida vs ruido mecánico estimado")
                    ax5.grid(True)
                    ax5.legend(fontsize=8)
                else:
                    ax5.text(0.5, 0.5, "sin perfil de ruido", ha="center",
                             va="center", transform=ax5.transAxes, color="gray")
                    ax5.set_xticks([]); ax5.set_yticks([])

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show(block=False)

            self.set_status(f"gráficos generados | {len(ventanas)} ventanas",
                            COLOR_OK)

        except Exception as e:
            self.set_status(f"error plot: {e}", COLOR_ERROR)




    # -------------------------------------------------------- AUTOFOCUS ----
    def run_autofocus(self):
        def worker():
            lens = self.station.lens
            cam = self.station.camera
            try:
                self.abort_event.clear()
                self.set_status("autofoco iniciado", COLOR_BUSY)

                if not lens.connected:
                    lens.connect()
                if not cam.connected:
                    cam.connect()
                if not (cam.capture_thread and cam.capture_thread.is_alive()):
                    cam.start_capture()
                    time.sleep(0.5)          # que lleguen las primeras ventanas

                if getattr(cam, "roi", None) is None:
                    self.log("OJO: autofoco SIN ROI (sensor completo)")
                else:
                    self.log(f"autofoco con ROI = {cam.roi}")

                cfg = dict(self.af_cfg)
                reset_first = cfg.pop("reset_first", True)
                af = AutoFocus(lens=lens, camera=cam, log=self.log, **cfg)

                res = af.run(should_abort=self.abort_event.is_set,
                             reset_first=reset_first)
                self.last_af = res
                self.af_plot_signal.emit()   # el grafico va al hilo principal

                color = {"OK": COLOR_OK, "DUDOSO": COLOR_BUSY}.get(
                    res.veredicto, COLOR_ERROR)
                self.set_status(f"autofoco {res.veredicto} | {res.goto_command} "
                                f"| error {res.error_pct:+.1f} %", color)
            except Exception as e:
                self.set_status(f"error autofoco: {e}", COLOR_ERROR)

        self._launch(worker)

    def run_plot_autofocus(self):
        if self.last_af is None:
            self.set_status("no hay autofoco que graficar", COLOR_BUSY)
            return
        try:
            AutoFocus.plot(self.last_af)
        except Exception as e:
            self.set_status(f"error plot autofoco: {e}", COLOR_ERROR)

    # ------------------------------------------------------------ CIERRE ---
    def closeEvent(self, event):
        try:
            if self.grabador:
                self.grabador.stop()
        except Exception:
            pass       

        self.abort_event.set()
        try:
            self.station.camera.stop()
        except Exception:
            pass
        event.accept()