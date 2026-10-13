import time
import threading
import serial
from serial.tools import list_ports


class Lens:
    MIN_POS = 0
    MAX_POS = 9999

    def __init__(self, serial_number, baudrate=38400, timeout=0.5):
        self.serial_number = serial_number
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.lock = threading.Lock()
        self.connected = False

    def connect(self):
        port = self._find_port()
        if not port:
            raise RuntimeError(f"[{self.serial_number}] Dispositivo no encontrado")

        self.ser = serial.Serial(
            port=port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        time.sleep(0.2)

        pos = self.get_position()
        if pos is None:
            raise RuntimeError(f"[{self.serial_number}] Conectó en {port} pero no respondió")
        
        print("Esperando 3 segundos..")
        time.sleep(1)
        print("Esperando 2 segundos..")
        time.sleep(1)
        print("Esperando 1 segundos..")
        time.sleep(1)
        self.connected = True
        print(f"[{self.serial_number}] Conectado en {port} | posición inicial: {pos}")


    def disconnect(self):
        with self.lock:
            if self.ser and self.ser.is_open:
                self.ser.close()
        self.connected = False

    def _find_port(self):
        for p in list_ports.comports():
            if p.serial_number == self.serial_number:
                return p.device
        return None

    def _send_command(self, cmd: str):
        if not self.ser or not self.ser.is_open:
            raise RuntimeError(f"[{self.serial_number}] Puerto no conectado")
        self.ser.write(cmd.encode("ascii"))

    def get_position(self):
        with self.lock:
            if not self.ser or not self.ser.is_open:
                raise RuntimeError(f"[{self.serial_number}] Puerto no conectado")
            self.ser.reset_input_buffer()
            self._send_command("P#")
            response = self.ser.read_until(b"#")

        if not response:
            return None

        text = response.decode("ascii", errors="ignore").strip().rstrip("#")
        return int(text) if text.isdigit() else None

    def move_to(self, target, timeout_move=6.0, poll_s=0.02, stable_reads=3):
        target = max(self.MIN_POS, min(self.MAX_POS, int(target)))

        with self.lock:
            self._send_command(f"M{target:04d}#")

        t0 = time.time()
        last_pos = None
        same_count = 0

        while time.time() - t0 < timeout_move:
            pos = self.get_position()
            if pos is None:
                time.sleep(poll_s)
                continue

            if pos == target:
                return pos

            if pos == last_pos:
                same_count += 1
                if same_count >= stable_reads:
                    return pos
            else:
                same_count = 0

            last_pos = pos
            time.sleep(poll_s)

        raise TimeoutError(
            f"[{self.serial_number}] Timeout. Target={target}, última pos={last_pos}"
        )

    def reset(self, fast_step=1000, fast_settle=0.5):
        tag = f"[{self.serial_number}]"
        self.move_to(9000, timeout_move=8.0)


        for pos in range(9000, -1, -fast_step):
            self.move_to(max(pos, 0), timeout_move=4.0)
            time.sleep(fast_settle)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            current = self.get_position()
            if current == 0:
                print("Esperando 5 segundos..")
                time.sleep(1)
                print("Esperando 4 segundos..")
                time.sleep(1)
                print("Esperando 3 segundos..")
                time.sleep(1)
                print("Esperando 2 segundos..")
                time.sleep(1)          
                print("Esperando 1 segundos..")
                time.sleep(1)  
                print(f"{tag} Reset completo. En posición 0.")
     
                return
            time.sleep(0.1)

        current = self.get_position()
        raise RuntimeError(
            f"{tag} Reset no confirmó posición 0. Última lectura: {current}"
        )

    def sweep(self, start=0, end=9000, step=50, settle_s=0.02):
        start = max(self.MIN_POS, min(self.MAX_POS, int(start)))
        end = max(self.MIN_POS, min(self.MAX_POS, int(end)))

        if step <= 0:
            raise ValueError("step debe ser > 0")

        direction = 1 if end >= start else -1
        positions = list(range(start, end, step * direction))
        if not positions or positions[-1] != end:
            positions.append(end)

        tag = f"[{self.serial_number}]"
        print(f"\n{tag} Barrido: {start}→{end} | paso={step} | {len(positions)} posiciones\n")

        log = []
        for i, target in enumerate(positions, start=1):
            reached = self.move_to(target)
            time.sleep(settle_s)
            print(f"{tag} [{i:03d}/{len(positions):03d}] target={target:04d} | reached={reached:04d}")
            log.append({
                "target": target,
                "reached": reached,
                "timestamp": time.time()
            })

        print(f"\n{tag} Barrido terminado.\n")
        return log