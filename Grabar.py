"""
Grabar.py
=========

Enciende y apaga la grabacion .raw de la camara de eventos, en cualquier
momento y sin interferir con el Live, la ROI ni el barrido.

Como funciona
-------------
No decodifica nada ni cuenta eventos. Solo le pide al HAL (Hardware
Abstraction Layer = capa de abstraccion de hardware) que ademas de entregar
los eventos al programa, los vaya escribiendo a disco tal como salen del
sensor. Por eso es gratis en rendimiento y totalmente independiente.

Importante
----------
El .raw guarda TODO el sensor, sin filtro de ruido y sin ROI. La ROI es un
recorte que tu programa hace despues, en software. Eso es bueno: el archivo
queda intacto y despues puedes reprocesarlo con la ROI que quieras.
"""

import os
import threading
import time
import re

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
def _limpiar(nombre):
    """Deja solo caracteres seguros. Devuelve None si queda vacio."""
    if nombre is None:
        return None
    nombre = os.path.basename(str(nombre).strip())   # corta rutas tipo ../otro/
    nombre = re.sub(r"[^A-Za-z0-9._\- ]", "_", nombre)
    return nombre.strip() or None


class Grabador:
    def __init__(self, camera, carpeta=None, prefijo="rec"):
        self.camera = camera
        self.carpeta = carpeta or CARPETA_PROYECTO
        self.prefijo = prefijo

        self._lock = threading.Lock()
        self.grabando = False
        self.archivo = None
        self.t_inicio = None

    # ------------------------------------------------------------------
    def _stream(self):
        """Devuelve la facility I_EventsStream de la camara viva."""
        device = getattr(self.camera, "_device", None)
        if device is None:
            raise RuntimeError("la camara no esta capturando: abre el Live primero")
        stream = device.get_i_events_stream()
        if stream is None:
            raise RuntimeError("este dispositivo no expone I_EventsStream")
        return stream


    @property
    def disponible(self):
        """True si se puede grabar ahora mismo."""
        return getattr(self.camera, "_device", None) is not None

    @staticmethod
    def _sin_pisar(ruta):
        """Si el archivo ya existe, agrega _2, _3... para no sobrescribir."""
        if not os.path.exists(ruta):
            return ruta
        base, ext = os.path.splitext(ruta)
        n = 2
        while os.path.exists(f"{base}_{n}{ext}"):
            n += 1
        return f"{base}_{n}{ext}"

    # ------------------------------------------------------------------
    def start(self, nombre=None):
        """Empieza a grabar. nombre=None -> prefijo + camara + fecha y hora."""
        with self._lock:
            if self.grabando:
                return self.archivo

            stream = self._stream()

            nombre = _limpiar(nombre)
            if nombre is None:
                marca = time.strftime("%Y%m%d_%H%M%S")
                nombre = f"{self.prefijo}_{self.camera.camera_id}_{marca}"
            if not nombre.lower().endswith(".raw"):
                nombre += ".raw"

            os.makedirs(self.carpeta, exist_ok=True)
            ruta = self._sin_pisar(os.path.join(self.carpeta, nombre))

            stream.log_raw_data(ruta)
            self.grabando = True
            self.archivo = ruta
            self.t_inicio = time.time()
            print(f"[Grabador] GRABANDO -> {ruta}")

            return ruta


    def stop(self):
        """Detiene y cierra el archivo. Devuelve un resumen o None."""
        with self._lock:
            if not self.grabando:
                return None

            ruta = self.archivo
            dur = time.time() - self.t_inicio
            try:
                self._stream().stop_log_raw_data()
            except RuntimeError:
                pass          # la camara ya se fue; Metavision cerró el archivo solo
            finally:
                self.grabando = False
                self.archivo = None
                self.t_inicio = None

            mb = os.path.getsize(ruta) / 1e6 if os.path.exists(ruta) else 0.0
            print(f"[Grabador] listo: {ruta} | {dur:.1f} s | {mb:.1f} MB")
            return {"path": ruta, "duracion_s": dur, "mb": mb}

    def toggle(self):
        """Un solo boton: si graba, detiene; si no, arranca."""
        return self.stop() if self.grabando else self.start()

    def elapsed(self):
        """Segundos grabados hasta ahora."""
        return (time.time() - self.t_inicio) if self.grabando else 0.0