"""
AutoFocus.py
============

Autoenfoque por estabilidad del centroide para el proyecto AF_VLAD.

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

MIN_POS, MAX_POS = 0, 7200
NAN = float("nan")


# ===========================================================================
# Utilidades estadisticas
# ===========================================================================
def sigma_centroide(cx, cy, robusto=True):
    """Devuelve (sigma_c, sigma_x, sigma_y) de una nube de centroides.

    robusto=True usa MAD (median absolute deviation = desviacion absoluta
    mediana) x 1.4826. Con datos gaussianos da lo mismo que la desviacion
    estandar, pero ignora ventanas sueltas con saltos raros."""
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    if cx.size < 2:
        return NAN, NAN, NAN
    if robusto:
        sx = 1.4826 * np.median(np.abs(cx - np.median(cx)))
        sy = 1.4826 * np.median(np.abs(cy - np.median(cy)))
    else:
        sx = cx.std(ddof=1)
        sy = cy.std(ddof=1)
    return float(np.hypot(sx, sy)), float(sx), float(sy)


def error_relativo_sigma(n, robusto=True):
    """Error relativo esperado de una sigma calculada con n muestras.
    Desviacion estandar clasica: 1/sqrt(2(n-1)). La MAD aprovecha peor los
    datos (eficiencia ~37 %), por eso se multiplica por ~1.65."""
    if n < 2:
        return NAN
    base = 1.0 / np.sqrt(2.0 * (n - 1))
    return float(base * (1.65 if robusto else 1.0))


def _mediana3(y):
    """Mediana movil de 3 puntos (bordes repetidos)."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return y.copy()
    yp = np.pad(y, 1, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(yp, 3), axis=1)


# ===========================================================================
# Resultados
# ===========================================================================
@dataclass
class StepMeasure:
    """Medicion en UNA posicion del lente."""
    fase: str                 # "barrido", "verificacion" o "final"
    index: int
    target: int
    reached: int
    cmd: str
    n_windows: int            # ventanas de 10 ms que entraron
    n_valid: int              # ventanas con centroide valido
    sigma_c: float            # px (pixeles)
    sigma_x: float
    sigma_y: float
    err_rel: float            # error relativo esperado de sigma_c
    count_mean: float         # eventos medios por ventana
    spread_mean: float        # tamano medio del blob, px
    t_rel: float              # segundos desde el inicio del autofoco
    cx: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    cy: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.sigma_c))


@dataclass
class AutoFocusResult:
    steps: List[StepMeasure]
    best_index: int
    best_target: int
    best_command: str
    best_sigma: float
    refined_target: float
    goto_target: int
    goto_command: str
    sigma_esperado: float     # sigma_c que el barrido predice en goto_target
    err_esperado: float       # su error relativo
    at_edge: bool
    contraste: float
    parabola: Optional[tuple] = None    # (a, b, c, x_ref, x_min, x_max)
    verif: List[StepMeasure] = field(default_factory=list)
    final: Optional[StepMeasure] = None
    error_pct: float = NAN
    z: float = NAN
    correlacion: float = NAN
    veredicto: str = "SIN VERIFICAR"
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        n_ok = sum(s.ok for s in self.steps)
        L = [
            "=== Autofoco por sigma del centroide ===",
            f"Steps validos      : {n_ok}/{len(self.steps)}",
            f"Minimo del barrido : {self.best_command} "
            f"(sigma_c = {self.best_sigma:.3f} px)",
            f"Foco refinado      : {self.refined_target:.1f}",
            f"Lente enviado a    : {self.goto_command}",
            f"Contraste curva    : {self.contraste:.2f}",
            f"sigma_c esperado   : {self.sigma_esperado:.3f} px "
            f"(+-{100 * self.err_esperado:.0f} %)",
        ]
        if self.final is not None:
            f = self.final
            L.append(f"sigma_c medido     : {f.sigma_c:.3f} px "
                     f"(+-{100 * f.err_rel:.0f} %) en reached={f.reached}")
            L.append(f"Error              : {self.error_pct:+.1f} %  |  z = {self.z:.2f}")
        if np.isfinite(self.correlacion):
            L.append(f"Correlacion forma  : r = {self.correlacion:.3f}")
        L.append(f"VEREDICTO          : {self.veredicto}")
        for w in self.warnings:
            L.append(f"AVISO: {w}")
        return "\n".join(L)


# ===========================================================================
# Autofoco
# ===========================================================================
class AutoFocus:
    """
    Parametros principales
    ----------------------
    n_steps, start, end : barrido (61 sobre 0..9000 -> paso 150).
    settle_s            : espera tras llegar, ANTES de medir (vibracion).
    measure_time_s      : tiempo de medicion por step (1 s ~ 100 ventanas).
    min_centroids       : centroides validos minimos para confiar en un step.
    robusto             : True = MAD, False = desviacion estandar clasica.
    approach_offset     : anti-backlash. Si hay que bajar, primero se va a
                          target - offset y luego se sube. 0 lo desactiva.
    pos_tol             : diferencia aceptada entre target y reached.
    use_refined         : True = va al vertice de la parabola,
                          False = al mejor punto de la grilla.
    verify_points       : puntos del mini-barrido de verificacion (0 = no).
    verify_span_steps   : semiancho del mini-barrido, en pasos del barrido.
    final_measure_s     : tiempo de la medicion final (None = measure_time_s).
    tol_pct, z_max, r_min : criterios del veredicto.
    log                 : funcion para escribir mensajes (print o ui.log).
    """

    def __init__(
        self,
        lens,
        camera,
        n_steps: int = 49,
        start: int = 0,
        end: int = 7200,
        settle_s: float = 0.30,
        measure_time_s: float = 1.0,
        min_centroids: int = 20,
        robusto: bool = True,
        approach_offset: int = 300,
        pos_tol: int = 5,
        use_refined: bool = True,
        verify_points: int = 5,
        verify_span_steps: float = 6.0,
        final_measure_s: Optional[float] = None,
        tol_pct: float = 15.0,
        z_max: float = 2.0,
        r_min: float = 0.8,
        log: Callable[[str], None] = print,
    ) -> None:
        if n_steps < 3:
            raise ValueError("n_steps debe ser al menos 3")
        self.lens = lens
        self.camera = camera
        self.n_steps = int(n_steps)
        self.start = int(start)
        self.end = int(end)
        self.settle_s = float(settle_s)
        self.measure_time_s = float(measure_time_s)
        self.min_centroids = int(min_centroids)
        self.robusto = bool(robusto)
        self.approach_offset = int(approach_offset)
        self.pos_tol = int(pos_tol)
        self.use_refined = bool(use_refined)
        self.verify_points = int(verify_points)
        self.verify_span_steps = float(verify_span_steps)
        self.final_measure_s = final_measure_s
        self.tol_pct = float(tol_pct)
        self.z_max = float(z_max)
        self.r_min = float(r_min)
        self._log = log or (lambda s: None)

        self._pos: Optional[int] = None
        self._t0 = time.time()
        self._warnings: List[str] = []
        self.result: Optional[AutoFocusResult] = None

    # ------------------------------------------------------------------
    @property
    def targets(self) -> np.ndarray:
        return np.rint(np.linspace(self.start, self.end, self.n_steps)).astype(int)

    @property
    def step_size(self) -> float:
        return abs(self.end - self.start) / (self.n_steps - 1)

    def commands(self) -> List[str]:
        return [f"M{t:04d}#" for t in self.targets]

    def estimated_duration_s(self) -> float:
        """Sin contar el viaje del motor."""
        n = self.n_steps + max(self.verify_points, 0) + 1
        return n * (self.settle_s + self.measure_time_s)

    # ---------------------------------------------------- camara ----------
    def _mark(self) -> int:
        cam = self.camera
        if hasattr(cam, "window_mark"):
            return cam.window_mark()
        with cam.lock:
            return cam._window_seq

    def _windows_since(self, mark: int) -> list:
        cam = self.camera
        if hasattr(cam, "windows_since"):
            return cam.windows_since(mark)
        with cam.lock:
            out = []
            for w in reversed(cam._windows):
                if w.get("seq", 0) <= mark:
                    break
                out.append(dict(w))
        return out[::-1]

    # ---------------------------------------------------- lente -----------
    def _goto(self, target) -> int:
        """Mueve el lente llegando SIEMPRE desde abajo (anti-backlash)."""
        target = int(np.clip(int(round(target)), MIN_POS, MAX_POS))

        if self.approach_offset > 0 and (self._pos is None or self._pos > target):
            previa = max(MIN_POS, target - self.approach_offset)
            if previa != target:
                self.lens.move_to(previa)

        reached = self.lens.move_to(target)
        if reached is None or abs(int(reached) - target) > self.pos_tol:
            # move_to puede volver antes de que el motor arranque: reintento
            time.sleep(0.1)
            reached = self.lens.move_to(target)

        if reached is None:
            self._warnings.append(f"target {target}: sin lectura de posicion")
            reached = target
        elif abs(int(reached) - target) > self.pos_tol:
            self._warnings.append(
                f"target {target}: llego a {reached} (dif {int(reached) - target})")

        self._pos = int(reached)
        return int(reached)

    # ---------------------------------------------------- medicion --------
    def _measure_at(self, fase, index, target, reached, duracion) -> StepMeasure:
        time.sleep(self.settle_s)          # deja morir la vibracion
        mark = self._mark()                # todo lo anterior queda fuera
        time.sleep(duracion)
        ws = self._windows_since(mark)
        return self._resumir(fase, index, target, reached, ws)

    def _resumir(self, fase, index, target, reached, ws) -> StepMeasure:
        validas = [w for w in ws
                   if np.isfinite(w.get("cx", NAN)) and np.isfinite(w.get("cy", NAN))]
        cx = np.array([w["cx"] for w in validas], dtype=float)
        cy = np.array([w["cy"] for w in validas], dtype=float)
        n_valid = int(cx.size)

        if n_valid >= self.min_centroids:
            sc, sx, sy = sigma_centroide(cx, cy, self.robusto)
            er = error_relativo_sigma(n_valid, self.robusto)
        else:
            sc = sx = sy = er = NAN

        counts = np.array([w.get("count", 0) for w in ws], dtype=float)
        spreads = np.array([w.get("spread", NAN) for w in ws], dtype=float)
        spreads = spreads[np.isfinite(spreads)]

        return StepMeasure(
            fase=fase, index=int(index), target=int(target), reached=int(reached),
            cmd=f"M{int(target):04d}#",
            n_windows=len(ws), n_valid=n_valid,
            sigma_c=sc, sigma_x=sx, sigma_y=sy, err_rel=er,
            count_mean=float(counts.mean()) if counts.size else NAN,
            spread_mean=float(spreads.mean()) if spreads.size else NAN,
            t_rel=time.time() - self._t0,
            cx=cx, cy=cy,
        )

    def _log_step(self, st: StepMeasure, total: int):
        self._log(f"[{st.fase[:5]} {st.index + 1:03d}/{total:03d}] {st.cmd} "
                  f"reached={st.reached:04d} | ventanas={st.n_windows} | "
                  f"centroides={st.n_valid} | sigma_c={st.sigma_c:.3f} px")

    # ---------------------------------------------------- flujo -----------
    def run(
        self,
        should_abort: Optional[Callable[[], bool]] = None,
        on_step: Optional[Callable[[int, int, float], None]] = None,
        reset_first: bool = False,
    ) -> AutoFocusResult:
        """Barrido + analisis + verificacion + ida al foco.
        Asume lente y camara conectados y la captura corriendo."""
        abort = should_abort or (lambda: False)
        self._warnings = []
        self._pos = None
        self._t0 = time.time()

        if reset_first:
            self._log("reseteando lente...")
            self.lens.reset(fast_step=1000, fast_settle=0.5)
            self._pos = 0

        objs = self.targets
        n = len(objs)
        self._log(f"barrido: {n} puntos {objs[0]}->{objs[-1]} | paso "
                  f"{self.step_size:.0f} | ~{self.estimated_duration_s():.0f} s "
                  f"+ viaje del motor")

        # ---- 1. barrido ----
        steps: List[StepMeasure] = []
        for i, target in enumerate(objs):
            if abort():
                self._warnings.append(f"barrido cancelado en el punto {i + 1}/{n}")
                break
            try:
                reached = self._goto(target)
            except Exception as e:
                self._warnings.append(f"punto {i + 1} ({target}): fallo move_to ({e})")
                continue
            st = self._measure_at("barrido", i, target, reached, self.measure_time_s)
            steps.append(st)
            self._log_step(st, n)
            if on_step is not None:
                on_step(i, reached, st.sigma_c)

        # ---- 2-3. minimo + parabola ----
        res = self._analizar(steps)
        self.result = res
        if abort():
            res.veredicto = "CANCELADO"
            return res

        self._log(f"minimo: {res.best_command} (sigma_c={res.best_sigma:.3f} px) "
                  f"| refinado {res.refined_target:.1f} -> {res.goto_command}")

        # ---- 4. mini-barrido de verificacion ----
        if self.verify_points >= 3:
            self._mini_barrido(res, abort)

        # ---- 5. ir al foco y medir de nuevo ----
        if not abort():
            reached = self._goto(res.goto_target)
            dur = self.final_measure_s or self.measure_time_s
            res.final = self._measure_at("final", 0, res.goto_target, reached, dur)
            self._log_step(res.final, 1)

        self._veredicto(res, abort())
        res.warnings = list(self._warnings)
        for linea in res.summary().split("\n"):
            self._log(linea)
        return res

    # ------------------------------------------------------------------
    def _curva(self, steps):
        """(x, y) validos del barrido, ordenados por posicion."""
        b = [s for s in steps if s.ok]
        x = np.array([s.reached for s in b], dtype=float)
        y = np.array([s.sigma_c for s in b], dtype=float)
        orden = np.argsort(x)
        return x[orden], y[orden]

    def _analizar(self, steps: List[StepMeasure]) -> AutoFocusResult:
        if len(steps) < 3:
            raise RuntimeError("se necesitan al menos 3 steps medidos")

        malos = sum(not s.ok for s in steps)
        if malos:
            self._warnings.append(
                f"{malos} steps con menos de {self.min_centroids} centroides: "
                f"se ignoran")

        xv, yv = self._curva(steps)
        if xv.size < 3:
            raise RuntimeError("menos de 3 steps con sigma_c valido: revisa la "
                               "ROI, min_events o measure_time_s")

        j = int(np.argmin(_mediana3(yv)))
        best = min((s for s in steps if s.ok), key=lambda s: abs(s.reached - xv[j]))
        at_edge = j in (0, xv.size - 1)

        refinada, parabola = float(xv[j]), None
        if at_edge:
            self._warnings.append("el minimo quedo en un extremo del barrido: el "
                                  "foco real puede estar fuera del rango")
        else:
            a0, a1 = max(0, j - 2), min(xv.size, j + 3)
            xx, yy = xv[a0:a1], yv[a0:a1]
            if np.unique(xx).size >= 3:
                x_ref = float(xv[j])
                a, b, c = np.polyfit(xx - x_ref, yy, 2)   # centrado: estable
                if a > 0:
                    v = x_ref - b / (2 * a)
                    if xx[0] <= v <= xx[-1]:
                        refinada = float(v)
                        parabola = (a, b, c, x_ref, float(xx[0]), float(xx[-1]))

        contraste = float((yv.max() - yv.min()) / yv.max()) if yv.max() > 0 else 0.0
        if contraste < 0.2:
            self._warnings.append(f"curva casi plana (contraste {contraste:.2f}): "
                                  "sube measure_time_s o revisa la ROI")

        goto = refinada if self.use_refined else float(xv[j])
        goto = int(np.clip(round(goto), MIN_POS, MAX_POS))

        sigma_esp = float(yv[j])
        if parabola is not None and self.use_refined:
            a, b, c, x_ref, _, _ = parabola
            u = goto - x_ref
            sigma_esp = float(a * u * u + b * u + c)

        return AutoFocusResult(
            steps=steps,
            best_index=best.index,
            best_target=best.target,
            best_command=best.cmd,
            best_sigma=best.sigma_c,
            refined_target=refinada,
            goto_target=goto,
            goto_command=f"M{goto:04d}#",
            sigma_esperado=sigma_esp,
            err_esperado=best.err_rel,
            at_edge=at_edge,
            contraste=contraste,
            parabola=parabola,
        )

    def _mini_barrido(self, res: AutoFocusResult, abort):
        span = self.verify_span_steps * self.step_size
        pts = res.goto_target + np.linspace(-span, span, self.verify_points)
        pts = np.unique(np.clip(np.rint(pts), MIN_POS, MAX_POS).astype(int))
        self._log(f"verificacion: mini-barrido en {pts.tolist()}")

        for k, t in enumerate(pts):
            if abort():
                return
            r = self._goto(t)
            st = self._measure_at("verificacion", k, t, r, self.measure_time_s)
            res.verif.append(st)
            self._log_step(st, len(pts))

        xv, yv = self._curva(res.steps)
        pares = np.array([(np.interp(s.reached, xv, yv), s.sigma_c)
                          for s in res.verif if s.ok])
        if len(pares) >= 3 and pares[:, 0].std() > 0 and pares[:, 1].std() > 0:
            res.correlacion = float(np.corrcoef(pares[:, 0], pares[:, 1])[0, 1])
        else:
            self._warnings.append("no se pudo calcular la correlacion de forma")

    def _veredicto(self, res: AutoFocusResult, abortado: bool):
        if abortado:
            res.veredicto = "CANCELADO"
            return
        f = res.final
        if f is None or not f.ok or not np.isfinite(res.sigma_esperado) \
                or res.sigma_esperado <= 0:
            res.veredicto = "SIN DATOS"
            return

        d = f.sigma_c - res.sigma_esperado
        res.error_pct = 100.0 * d / res.sigma_esperado
        e1 = res.sigma_esperado * res.err_esperado
        e2 = f.sigma_c * f.err_rel
        den = float(np.hypot(e1, e2))
        res.z = abs(d) / den if den > 0 else NAN

        forma_ok = (not np.isfinite(res.correlacion)) or res.correlacion >= self.r_min
        if abs(res.error_pct) <= self.tol_pct and forma_ok:
            res.veredicto = "OK"
        elif np.isfinite(res.z) and res.z <= self.z_max:
            res.veredicto = "DUDOSO"
        else:
            res.veredicto = "FALLA"

    # ---------------------------------------------------- grafico ---------
    @staticmethod
    def plot(res: AutoFocusResult, show: bool = True):
        """Llamar SOLO desde el hilo principal (matplotlib no es thread-safe)."""
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        a1, a2, a3, a4 = axs.ravel()
        b = [s for s in res.steps if s.ok]
        x = np.array([s.reached for s in b])
        y = np.array([s.sigma_c for s in b])
        e = np.array([s.sigma_c * s.err_rel for s in b])

        # 1. curva de foco
        a1.errorbar(x, y, yerr=e, fmt="o-", ms=3, capsize=2, color="tab:purple",
                    label="barrido")
        if res.parabola is not None:
            a, bb, c, x_ref, x0, x1 = res.parabola
            xp = np.linspace(x0, x1, 100)
            a1.plot(xp, a * (xp - x_ref) ** 2 + bb * (xp - x_ref) + c,
                    color="tab:red", lw=1.2, label="parabola del valle")
        v = [s for s in res.verif if s.ok]
        if v:
            a1.errorbar([s.reached for s in v], [s.sigma_c for s in v],
                        yerr=[s.sigma_c * s.err_rel for s in v], fmt="s",
                        ms=6, capsize=2, color="tab:orange",
                        label=f"verificacion (r={res.correlacion:.2f})")
        if res.final is not None and res.final.ok:
            f = res.final
            a1.errorbar([f.reached], [f.sigma_c], yerr=[f.sigma_c * f.err_rel],
                        fmt="*", ms=15, color="red", capsize=3,
                        label=f"final ({res.error_pct:+.1f} %)")
        a1.axvline(res.goto_target, color="gray", ls="--", lw=1,
                   label=f"foco {res.goto_command}")
        a1.set_xlabel("Posicion del lente [step]")
        a1.set_ylabel("sigma_c del centroide [px]")
        a1.set_title("Curva de foco (minimo = mejor foco)")
        a1.grid(True)
        a1.legend(fontsize=8)

        # 2. componentes X e Y
        a2.plot(x, [s.sigma_x for s in b], "o-", ms=3, label="sigma_x")
        a2.plot(x, [s.sigma_y for s in b], "o-", ms=3, label="sigma_y")
        a2.axvline(res.goto_target, color="gray", ls="--", lw=1)
        a2.set_xlabel("Posicion del lente [step]")
        a2.set_ylabel("[px]")
        a2.set_title("Componentes del temblor del centroide")
        a2.grid(True)
        a2.legend(fontsize=8)

        # 3. calidad de la senal
        todos = res.steps
        xt = [s.reached for s in todos]
        a3.plot(xt, [s.n_valid for s in todos], "o-", ms=3, color="tab:green",
                label="centroides validos")
        a3.set_xlabel("Posicion del lente [step]")
        a3.set_ylabel("centroides validos por step")
        a3.grid(True)
        a3b = a3.twinx()
        a3b.plot(xt, [s.spread_mean for s in todos], ".-", color="tab:brown",
                 alpha=0.7, label="tamano blob")
        a3b.set_ylabel("tamano del blob (spread) [px]")
        a3.set_title("Calidad de senal por step")
        h = a3.get_lines() + a3b.get_lines()
        a3.legend(h, [l.get_label() for l in h], fontsize=8)

        # 4. nube de centroides: mejor step vs final
        best = next((s for s in res.steps if s.index == res.best_index), None)
        for s, col, nom in ((best, "tab:purple", "barrido"),
                            (res.final, "red", "final")):
            if s is not None and s.cx.size:
                a4.plot(s.cx - np.median(s.cx), s.cy - np.median(s.cy), ".",
                        ms=4, alpha=0.6, color=col,
                        label=f"{nom} {s.cmd} (sigma_c={s.sigma_c:.3f})")
        a4.set_aspect("equal", adjustable="datalim")
        a4.set_xlabel("dx [px]")
        a4.set_ylabel("dy [px]")
        a4.set_title("Centroides en el foco")
        a4.grid(True)
        a4.legend(fontsize=8)

        fig.suptitle(f"Autofoco | {res.goto_command} | {res.veredicto} | "
                     f"error {res.error_pct:+.1f} %", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if show:
            plt.show(block=False)
        return fig


# # ===========================================================================
# # Prueba en seco: lente y camara simulados, sin hardware
# # ===========================================================================
# if __name__ == "__main__":
#     import threading
#     from collections import deque

#     rng = np.random.default_rng(1)

#     class LenteFalso:
#         def __init__(self):
#             self.pos = 0

#         def move_to(self, target):
#             self.pos = int(np.clip(target, 0, 9999))
#             time.sleep(0.005)
#             return self.pos

#         def get_position(self):
#             return self.pos

#         def reset(self, **kw):
#             self.pos = 0

#     class CamaraFalsa:
#         """Una ventana cada 10 ms; el centroide tiembla mas con desenfoque."""
#         def __init__(self, lente, foco=4663.0):
#             self.lente, self.foco = lente, foco
#             self.lock = threading.Lock()
#             self._windows = deque(maxlen=5000)
#             self._window_seq = 0
#             threading.Thread(target=self._loop, daemon=True).start()

#         def _loop(self):
#             while True:
#                 d = abs(self.lente.pos - self.foco)
#                 spread = 2.0 + 0.004 * d          # tamano del blob, px
#                 n = 60                             # eventos por ventana
#                 ruido = spread / np.sqrt(n)        # error de centroide
#                 tilt = 0.25                        # turbulencia, px
#                 cx = 330 + rng.normal(0, tilt) + rng.normal(0, ruido)
#                 cy = 240 + rng.normal(0, tilt) + rng.normal(0, ruido)
#                 with self.lock:
#                     self._window_seq += 1
#                     self._windows.append({"seq": self._window_seq,
#                                           "t": time.time(), "count": n,
#                                           "spread": spread, "cx": cx, "cy": cy})
#                 time.sleep(0.01)

#     lente = LenteFalso()
#     cam = CamaraFalsa(lente)
#     af = AutoFocus(lente, cam, n_steps=49, start=0, end=7200,
#                    settle_s=0.0, measure_time_s=0.3, min_centroids=15,
#                    log=lambda s: None)
#     res = af.run()
#     print(res.summary())
#     print(f"\nFoco verdadero: {cam.foco:.0f} | error de posicion: "
#           f"{res.goto_target - cam.foco:+.0f} steps")