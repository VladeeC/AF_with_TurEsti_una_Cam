
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication

from CameraEvent import EventCamera
from Cell import Cell
from Lens import Lens
from CameraUI import CameraUI
from CameraEventROI import RoiEventCamera
from CameraEventROI import RoiEventCamera
from Grabar import Grabador




# ===================== CONFIGURACION DE LA UNICA CELDA =====================
LENS_SERIAL = "AB0KXVF2A"#"AG0JKSUXA"
CAMERA_ID   = "00000433"

# ---- comun a los dos modos ----
SHOW_LIVE  = True
FPS        = 25
HALF_SIZE  = 80            # medio ancho de la ROI en pixeles
CALIB_US   = 1_000_000     # 1 s de eventos para ubicar la estrella
CALIB_MIN  = 300           # o al menos 300 eventos, lo que tarde mas


# ===========================================================================
#  MODO: "foco"        -> barrido completo, integracion larga
#        "turbulencia" -> lente quieto, integracion corta
# ===========================================================================
MODO = "foco"        # <--
# ===========================================================================
if MODO == "foco":
    DELTA_T = 10000
    HISTORY = 20000
    MIN_EVENTS = 1          # 10
    SWEEP_CFG = {
        "start":     0,
        "end":       7200,
        "step":     150,
        "settle_s":  0.03, # segundos
    }
    NOISE_CFG = {            # solo para «Obtener ruido»
        "start":     0,
        "end":       7200,
        "step":      150,
        "settle_s":  1.00,   # mas tiempo para integrar, sin turbulencia hay poca senal
    }
    AF_CFG = {

        "n_steps":           49,    
        "start":             0,
        "end":               7200,   # coherente con MAX_POS = 7200
        "settle_s":          0.030,
        "measure_time_s":    0.03,
        "min_centroids":     20,
        "approach_offset":   300,
        "verify_points":     5,
        "verify_span_steps": 6.0,
        "tol_pct":           15.0,
        "reset_first":       True,

    }

elif MODO == "turbulencia":
    DELTA_T = 2000    # 1 ms: sigue la turbulencia a 1 kHz
    HISTORY = 200000       # 200 s de ventanas de 1 ms
    MIN_EVENTS = 15        # con 1 ms hay ~10x menos eventos por ventana
    POS_FIJA = 3000        # posicion del lente durante la medicion
    NOISE_CFG = None
    AF_CFG = None
    SWEEP_CFG = {
        "start":     POS_FIJA,
        "end":       POS_FIJA,   # igual a start -> un solo step
        "step":      150,        # no se usa
        "settle_s":  20.0,       # 20 s de medicion
        "discard_s": 0.15,
    }

else:
    raise ValueError(f"MODO desconocido: {MODO}")
# ===========================================================================


def build_station():
    """Arma la celda: un lente + una camara."""
    lens = Lens(serial_number=LENS_SERIAL)
    camera = RoiEventCamera(
        camera_id=CAMERA_ID,
        roi=(253, 159, 413, 319),
        half_size=HALF_SIZE,
        delta_t=DELTA_T,
        fps=FPS,
        show_live=SHOW_LIVE,
        calibration_time_us=CALIB_US,
        calibration_min_events=CALIB_MIN,
        min_events=MIN_EVENTS,
        history=HISTORY,
    )

    return Cell(
        station_id=1,
        serial_number=LENS_SERIAL,
        lens=lens,
        camera=camera,
    )

def main():
    print(f"=== MODO: {MODO} | delta_t={DELTA_T} us | "
          f"barrido {SWEEP_CFG['start']}→{SWEEP_CFG['end']} ===")

    station = build_station()          
    grabador = Grabador(station.camera)

    app = QApplication(sys.argv)
    ui = CameraUI(station=station, sweep_cfg=SWEEP_CFG,
                  noise_cfg=NOISE_CFG, grabador=grabador, af_cfg=AF_CFG)
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()



