import threading
import numpy as np
import cv2

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import EventLoop, Window


class EventCamera:
    def __init__(
        self,
        camera_id,
        delta_t=10000,
        accumulation_time_us=10000,
        activity_time_ths=20000,
        fps=25,
        show_live=True,
    ):
        self.camera_id = camera_id
        self.delta_t = delta_t
        self.accumulation_time_us = accumulation_time_us
        self.activity_time_ths = activity_time_ths
        self.fps = fps
        self.show_live = show_live

        self.connected = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.event_counter = 0
        self.capture_thread = None

    def connect(self):
        self.connected = True
        print(f"[Camera {self.camera_id}] lista")

    def disconnect(self):
        self.stop()
        self.connected = False
        print(f"[Camera {self.camera_id}] desconectada")

    def start_capture(self):
        if not self.connected:
            raise RuntimeError(f"[Camera {self.camera_id}] no conectada")

        self.stop_event.clear()
        self.event_counter = 0
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()

    def reset_counter(self):
        with self.lock:
            self.event_counter = 0

    def get_and_reset_counter(self):
        with self.lock:
            value = self.event_counter
            self.event_counter = 0
            return value

    def _capture_loop(self):
        # Si camera_id no es identificador real de cámara, usa "" para probar con una sola
        mv_iterator = EventsIterator(input_path=self.camera_id, delta_t=self.delta_t)
        height, width = mv_iterator.get_size()

        periodic_gen = PeriodicFrameGenerationAlgorithm(
            width,
            height,
            accumulation_time_us=self.accumulation_time_us,
            fps=self.fps,
        )

        activity_filter = ActivityNoiseFilterAlgorithm(
            width, height, self.activity_time_ths
        )
        events_buf = activity_filter.get_empty_output_buffer()

        if self.show_live:
            with Window(
                f"Camera {self.camera_id}",
                width,
                height,
                Window.RenderMode.BGR
            ) as window:

                def on_frame(ts, frame):
                    cv2.putText(
                        frame,
                        f"{self.camera_id}  ts={ts}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
                    window.show(frame)

                periodic_gen.set_output_callback(on_frame)

                for evs in mv_iterator:
                    if self.stop_event.is_set():
                        break

                    EventLoop.poll_and_dispatch()

                    activity_filter.process_events(evs, events_buf)
                    periodic_gen.process_events(events_buf)

                    try:
                        filtered_event_count = events_buf.size()
                    except AttributeError:
                        filtered_events = np.array(events_buf, copy=False)
                        filtered_event_count = filtered_events.size

                    with self.lock:
                        self.event_counter += filtered_event_count

                    if window.should_close():
                        self.stop_event.set()
                        break
        else:
            for evs in mv_iterator:
                if self.stop_event.is_set():
                    break

                activity_filter.process_events(evs, events_buf)
                periodic_gen.process_events(events_buf)

                try:
                    filtered_event_count = events_buf.size()
                except AttributeError:
                    filtered_events = np.array(events_buf, copy=False)
                    filtered_event_count = filtered_events.size

                with self.lock:
                    self.event_counter += filtered_event_count



# import threading
# import numpy as np

# from metavision_core.event_io import EventsIterator
# from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
# from metavision_sdk_cv import ActivityNoiseFilterAlgorithm


# class EventCamera:
#     def __init__(
#         self,
#         camera_id,
#         delta_t=10000,
#         accumulation_time_us=10000,
#         activity_time_ths=20000,
#     ):
#         self.camera_id = camera_id
#         self.delta_t = delta_t
#         self.accumulation_time_us = accumulation_time_us
#         self.activity_time_ths = activity_time_ths

#         self.connected = False
#         self.stop_event = threading.Event()
#         self.lock = threading.Lock()

#         self.event_counter = 0
#         self.capture_thread = None

#     def connect(self):
#         self.connected = True
#         print(f"[Camera {self.camera_id}] lista")

#     def disconnect(self):
#         self.stop()
#         self.connected = False
#         print(f"[Camera {self.camera_id}] desconectada")

#     def start_capture(self):
#         if not self.connected:
#             raise RuntimeError(f"[Camera {self.camera_id}] no conectada")

#         self.stop_event.clear()
#         self.event_counter = 0

#         self.capture_thread = threading.Thread(
#             target=self._capture_loop,
#             daemon=True,
#         )
#         self.capture_thread.start()

#     def stop(self):
#         self.stop_event.set()
#         if self.capture_thread and self.capture_thread.is_alive():
#             self.capture_thread.join()

#     def reset_counter(self):
#         with self.lock:
#             self.event_counter = 0

#     def get_counter(self):
#         with self.lock:
#             return self.event_counter

#     def get_and_reset_counter(self):
#         with self.lock:
#             value = self.event_counter
#             self.event_counter = 0
#             return value

#     def _capture_loop(self):
#         mv_iterator = EventsIterator(input_path="", delta_t=self.delta_t)
#         height, width = mv_iterator.get_size()

#         event_frame_gen = PeriodicFrameGenerationAlgorithm(
#             width, height, accumulation_time_us=self.accumulation_time_us
#         )

#         activity_filter = ActivityNoiseFilterAlgorithm(
#             width, height, self.activity_time_ths
#         )
#         events_buf = activity_filter.get_empty_output_buffer()

#         for evs in mv_iterator:
#             if self.stop_event.is_set():
#                 break

#             activity_filter.process_events(evs, events_buf)
#             event_frame_gen.process_events(events_buf)

#             try:
#                 filtered_event_count = events_buf.size()
#             except AttributeError:
#                 filtered_events = np.array(events_buf, copy=False)
#                 filtered_event_count = filtered_events.size

#             with self.lock:
#                 self.event_counter += filtered_event_count



























# import time
# import threading
# import numpy as np
# import Cell

# from metavision_core.event_io import EventsIterator
# from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
# from metavision_sdk_cv import ActivityNoiseFilterAlgorithm


# class EventCamera:
#     def __init__(
#         self,
#         camera_id,
#         delta_t=10000,
#         accumulation_time_us=10000,
#         activity_time_ths=20000,
#     ):
#         self.camera_id = camera_id


#         self.delta_t = delta_t
#         self.accumulation_time_us = accumulation_time_us
#         self.activity_time_ths = activity_time_ths

#         self.connected = False
#         self.stop_event = threading.Event()
#         self.lock = threading.Lock()

#         self.samples = []
#         self.capture_thread = None

#     def connect(self):
#         self.connected = True
#         print(f"[Camera {self.camera_id}] lista")

#     def disconnect(self):
#         self.stop()
#         self.connected = False
#         print(f"[Camera {self.camera_id}] desconectada")

#     def start_capture(self, sample_period_s=0.3, max_duration_s=None):
#         if not self.connected:
#             raise RuntimeError(f"[Camera {self.camera_id}] no conectada")

#         self.stop_event.clear()
#         self.samples = []
#         self.capture_thread = threading.Thread(            #Hilo secundario 1
#             target=self._capture_loop,
#             args=(sample_period_s, max_duration_s),
#             daemon=True,
#         )
#         self.capture_thread.start()

#     def stop(self):
#         self.stop_event.set()
#         if self.capture_thread and self.capture_thread.is_alive():
#             self.capture_thread.join()

#     def get_samples(self):
#         with self.lock:
#             return list(self.samples)
#     def _send_command(self, cmd: str):
#         if not self.ser or not self.ser.is_open:
#             raise RuntimeError(f"[{self.serial_number}] Puerto no conectado")
#         self.ser.write(cmd.encode("ascii"))

#         t0 = time.time()
#         last_pos = None
#         same_count = 0    
    

#     def move_to(self, target, timeout_move=6.0, poll_s=0.1, stable_reads=3):
#         target = max(self.MIN_POS, min(self.MAX_POS, int(target)))

#         with self.lock:
#             self._send_command(f"M{target:04d}#")



#     def _capture_loop(self, sample_period_s, max_duration_s):
#         mv_iterator = EventsIterator(input_path="", delta_t=self.delta_t)
#         height, width = mv_iterator.get_size()

#         event_frame_gen = PeriodicFrameGenerationAlgorithm(
#             width, height, accumulation_time_us=self.accumulation_time_us
#         )

#         activity_filter = ActivityNoiseFilterAlgorithm(
#             width, height, self.activity_time_ths
#         )
#         events_buf = activity_filter.get_empty_output_buffer()

#         start_time = time.time()
#         last_sample_time = start_time
#         total_filtered_events = 0

#         for evs in mv_iterator:
#             if self.stop_event.is_set():
#                 break

#             activity_filter.process_events(evs, events_buf)
#             event_frame_gen.process_events(events_buf)

#             try:
#                 filtered_event_count = events_buf.size()
#             except AttributeError:
#                 filtered_events = np.array(events_buf, copy=False)
#                 filtered_event_count = filtered_events.size

#             total_filtered_events += filtered_event_count
#             now = time.time()

#             if now - last_sample_time >= sample_period_s:
#                 sample = {
#                     "timestamp": now,
#                     "event_count": total_filtered_events
#                 }
#                 with self.lock:
#                     self.samples.append(sample)

#                 print(f"[Camera {self.camera_id}] eventos: {total_filtered_events}")
#                 total_filtered_events = 0
#                 last_sample_time = now

#             if max_duration_s is not None and (now - start_time) >= max_duration_s:
#                 break