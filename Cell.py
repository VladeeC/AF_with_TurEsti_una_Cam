import time

class Cell:
    def __init__(self, station_id, serial_number, lens, camera):
        self.station_id = station_id
        self.serial_number = serial_number
        self.lens = lens
        self.camera = camera

    def connect_all(self):
        print(f"[Station {self.station_id}] conectando...")
        self.lens.connect()
        self.camera.connect()

    def disconnect_all(self):
        print(f"[Station {self.station_id}] desconectando...")
        self.camera.disconnect()
        self.lens.disconnect()


    def run_stage1(self, start, end, step, settle_s, sample_period_s=None):
        """
        Etapa 1:
        - prende cámara
        - mueve el lente paso a paso
        - asocia eventos al step enviado
        - guarda también un arreglo simple con los event_count
        """
        self.connect_all()

        try:
            self.lens.reset(fast_step=1000, fast_settle=0.5)
            time.sleep(0.1)

            self.camera.start_capture()

            step_events = []
            event_gen = []
            step_targets = []

            direction = 1 if end >= start else -1
            positions = list(range(start, end, step * direction))
            if not positions or positions[-1] != end:
                positions.append(end)

            for i, target in enumerate(positions, start=1):
                cmd = f"M{target:04d}#"

                # empezar conteo para este step
                self.camera.reset_counter()

                reached = self.lens.move_to(target)
                time.sleep(settle_s)

                # eventos asociados a este step
                event_count = self.camera.get_and_reset_counter()

                row = {
                    "index": i,
                    "cmd": cmd,
                    "target": target,
                    "reached": reached,
                    "event_count": event_count,
                    "timestamp": time.time(),
                }
                step_events.append(row)

                # arreglos simples para usar después
                event_gen.append(event_count)
                step_targets.append(target)

                print(f"[Station {self.station_id}] {cmd} -> {event_count} eventos")

            self.camera.stop()

            print(f"\n[Station {self.station_id}] event_gen = {event_gen}\n")

            return {
                "station_id": self.station_id,
                "serial_number": self.serial_number,
                "step_events": step_events,
                "event_gen": event_gen,
                "step_targets": step_targets,
            }

        finally:
            self.disconnect_all()







            
    # def run_stage1(self, start, end, step, settle_s, sample_period_s=None):
    #     """
    #     Etapa 1:
    #     - prende cámara
    #     - mueve el lente paso a paso
    #     - asocia eventos al step enviado
    #     """
    #     self.connect_all()

    #     try:
    #         self.lens.reset(fast_step=1000, fast_settle=0.5)
    #         time.sleep(0.1)

    #         self.camera.start_capture()

    #         step_events = []

    #         direction = 1 if end >= start else -1
    #         positions = list(range(start, end, step * direction))
    #         if not positions or positions[-1] != end:
    #             positions.append(end)

    #         for i, target in enumerate(positions, start=1):
    #             cmd = f"M{target:04d}#"

    #             # empezar conteo para este step
    #             self.camera.reset_counter()

    #             reached = self.lens.move_to(target)
    #             time.sleep(settle_s)

    #             # eventos asociados a este step
    #             event_count = self.camera.get_and_reset_counter()

    #             row = {
    #                 "index": i,
    #                 "cmd": cmd,
    #                 "target": target,
    #                 "reached": reached,
    #                 "event_count": event_count,
    #                 "timestamp": time.time(),
    #             }
    #             step_events.append(row)

    #             print(f"[Station {self.station_id}] {cmd} -> {event_count} eventos")
                

    #         self.camera.stop()

    #         return {
    #             "station_id": self.station_id,
    #             "serial_number": self.serial_number,
    #             "step_events": step_events,
    #         }

    #     finally:
    #         self.disconnect_all()










# import time
# class Cell:
#     def __init__(self, station_id, serial_number, lens, camera):
#         self.station_id = station_id
#         self.serial_number = serial_number
#         self.lens = lens
#         self.camera = camera

#     def connect_all(self):
#         print(f"[Station {self.station_id}] conectando...")
#         self.lens.connect()
#         self.camera.connect()

#     def disconnect_all(self):
#         print(f"[Station {self.station_id}] desconectando...")
#         self.camera.disconnect()
#         self.lens.disconnect()

#     def run_stage1(self, start, end, step, settle_s, sample_period_s):
#         """
#         Etapa 1:
#         - prende cámara
#         - empieza captura en paralelo
#         - mueve el lente
#         - detiene captura
#         - devuelve log del lente y muestras de cámara
#         """
#         self.connect_all()

#         try:
#             reset = self.lens.reset(fast_step=1000, fast_settle=0.5)
#             time.sleep(0.1)
#             self.camera.start_capture(sample_period_s=sample_period_s)
#             sweep_log = self.lens.sweep(start=start, end=end, step=step, settle_s=settle_s)
#             self.camera.stop()
#             camera_samples = self.camera.get_samples()

#             return {
#                 "station_id": self.station_id,
#                 "serial_number": self.serial_number,
#                 "reset": reset,
#                 "sweep_log": sweep_log,
#                 "camera_samples": camera_samples,
#             }

#         finally:
#             self.disconnect_all()