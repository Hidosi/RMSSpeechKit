import sys
import queue
import threading
import numpy as np
import sounddevice as sd
import subprocess
from PyQt5 import QtWidgets, QtCore
import grpc

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc

CHUNK_SIZE = 4000  # байт
SAMPLE_RATE = 8000
CHANNELS = 1
RMS_THRESHOLD = 500  # Порог RMS для детекции речи
COST_PER_15_SECONDS = 0.16  # Стоимость 15 секунд аудио в рублях
SILENCE_DURATION_MS = 5000  # Длительность тишины для отправки в сервис (5 секунд)

class AudioListener(QtCore.QObject):
    rms_signal = QtCore.pyqtSignal(float)
    audio_chunk_signal = QtCore.pyqtSignal(bytes)

    def __init__(self, use_parec=False, device_index=None):
        super().__init__()
        self.use_parec = use_parec
        self.device_index = device_index
        self.stream = None
        self.process = None
        self.running = False

    def start(self):
        self.running = True
        if self.use_parec:
            self._start_parec()
        else:
            self._start_sounddevice()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.process:
            self.process.terminate()
            self.process = None

    def _start_sounddevice(self):
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Sounddevice status: {status}", file=sys.stderr)
            data = bytes(indata)
            audio_array = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            self.rms_signal.emit(rms)
            self.audio_chunk_signal.emit(data)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            device=self.device_index,
            blocksize=CHUNK_SIZE // 2,
            callback=callback
        )
        self.stream.start()

    def _start_parec(self):
        command = ["parec", "--format=s16le", "--channels=1", "--rate=8000"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=CHUNK_SIZE)
        threading.Thread(target=self._read_parec_output, daemon=True).start()

    def _read_parec_output(self):
        while self.running and self.process and self.process.stdout:
            data = self.process.stdout.read(CHUNK_SIZE)
            if not data:
                break
            audio_array = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            self.rms_signal.emit(rms)
            self.audio_chunk_signal.emit(data)

class RecognizerClient(QtCore.QObject):
    partial_text_signal = QtCore.pyqtSignal(str)
    final_text_signal = QtCore.pyqtSignal(str)
    final_refinement_text_signal = QtCore.pyqtSignal(str)
    error_signal = QtCore.pyqtSignal(str)
    cost_signal = QtCore.pyqtSignal(float)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.channel = grpc.secure_channel('stt.api.cloud.yandex.net:443', grpc.ssl_channel_credentials())
        self.stub = stt_service_pb2_grpc.RecognizerStub(self.channel)
        self.audio_queue = queue.Queue()
        self.running = False
        self.total_audio_seconds = 0.0
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._recognize_streaming, daemon=True).start()

    def stop(self):
        self.running = False

    def send_audio(self, data):
        if self.running:
            self.audio_queue.put(data)

    def send_silence_chunk(self, duration_ms):
        if self.running:
            self.audio_queue.put(('silence', int(duration_ms)))

    def _audio_generator(self):
        recognize_options = stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=SAMPLE_RATE,
                        audio_channel_count=CHANNELS
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                    profanity_filter=True,
                    literature_text=False
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=['ru-RU']
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME
            )
        )
        yield stt_pb2.StreamingRequest(session_options=recognize_options)

        while self.running:
            try:
                item = self.audio_queue.get(timeout=1)
                if isinstance(item, tuple) and item[0] == 'silence':
                    # Отправляем пустой аудиочанк для обозначения тишины
                    yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=b''))
                elif isinstance(item, bytes):
                    yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=item))
                    seconds = len(item) / (SAMPLE_RATE * 2 * CHANNELS)
                    with self.lock:
                        self.total_audio_seconds += seconds
                        cost = self._calculate_cost(self.total_audio_seconds)
                    self.cost_signal.emit(cost)
                else:
                    print(f"Unexpected item type in audio_queue: {type(item)}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Exception in _audio_generator: {e}")
                break

    def _calculate_cost(self, total_seconds):
        units = int(np.ceil(total_seconds / 15))
        return units * COST_PER_15_SECONDS

    def _recognize_streaming(self):
        try:
            responses = self.stub.RecognizeStreaming(self._audio_generator(), metadata=(
                ('authorization', f'Api-Key {self.api_key}'),
            ))
            for r in responses:
                event_type = r.WhichOneof('Event')
                if event_type == 'partial' and len(r.partial.alternatives) > 0:
                    alternatives = [a.text for a in r.partial.alternatives]
                    text = ' / '.join(alternatives)
                    self.partial_text_signal.emit(text)
                elif event_type == 'final' and len(r.final.alternatives) > 0:
                    alternatives = [a.text for a in r.final.alternatives]
                    text = ' / '.join(alternatives)
                    self.final_text_signal.emit(text)
                elif event_type == 'final_refinement':
                    alternatives = [a.text for a in r.final_refinement.normalized_text.alternatives]
                    if alternatives:
                        text = ' / '.join(alternatives)
                        self.final_refinement_text_signal.emit(text)
        except grpc.RpcError as e:
            self.error_signal.emit(f"gRPC error: {e.code()} - {e.details()}")

class MainWindow(QtWidgets.QWidget):
    def __init__(self, api_key, use_parec, device_index):
        super().__init__()
        self.setWindowTitle("Speech Recognizer with RMS Detection")
        self.resize(800, 600)

        self.api_key = api_key
        self.use_parec = use_parec
        self.device_index = device_index

        self.audio_listener = AudioListener(use_parec=use_parec, device_index=device_index)
        self.recognizer = RecognizerClient(api_key)

        self.init_ui()
        self.connect_signals()

        self.speech_active = False
        self.silence_counter = 0
        self.max_silence_chunks = int(5 / (CHUNK_SIZE / (SAMPLE_RATE * 2)))  # 5 секунд тишины

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        self.rms_label = QtWidgets.QLabel("RMS: 0")
        self.status_label = QtWidgets.QLabel("Status: Waiting for speech...")

        layout.addWidget(self.rms_label)
        layout.addWidget(self.status_label)

        # Поля для вывода текста разных типов
        self.partial_text_edit = QtWidgets.QTextEdit()
        self.partial_text_edit.setReadOnly(True)
        self.partial_text_edit.setPlaceholderText("Partial (промежуточный) текст")

        self.final_text_edit = QtWidgets.QTextEdit()
        self.final_text_edit.setReadOnly(True)
        self.final_text_edit.setPlaceholderText("Final (окончательный) текст")

        self.final_refinement_text_edit = QtWidgets.QTextEdit()
        self.final_refinement_text_edit.setReadOnly(True)
        self.final_refinement_text_edit.setPlaceholderText("Final refinement (нормализованный) текст")

        layout.addWidget(QtWidgets.QLabel("Partial results:"))
        layout.addWidget(self.partial_text_edit)
        layout.addWidget(QtWidgets.QLabel("Final results:"))
        layout.addWidget(self.final_text_edit)
        layout.addWidget(QtWidgets.QLabel("Final refinement results:"))
        layout.addWidget(self.final_refinement_text_edit)

        self.cost_label = QtWidgets.QLabel("Estimated cost: 0.00 ₽")
        layout.addWidget(self.cost_label)

        self.setLayout(layout)

    def connect_signals(self):
        self.audio_listener.rms_signal.connect(self.on_rms)
        self.audio_listener.audio_chunk_signal.connect(self.on_audio_chunk)
        self.recognizer.partial_text_signal.connect(self.on_partial_text)
        self.recognizer.final_text_signal.connect(self.on_final_text)
        self.recognizer.final_refinement_text_signal.connect(self.on_final_refinement_text)
        self.recognizer.error_signal.connect(self.on_error)
        self.recognizer.cost_signal.connect(self.on_cost)

    def on_rms(self, rms):
        self.rms_label.setText(f"RMS: {rms:.2f}")
        if rms > RMS_THRESHOLD:
            if not self.speech_active:
                self.status_label.setText("Status: Speech detected, sending audio...")
                self.speech_active = True
                self.silence_counter = 0
        else:
            if self.speech_active:
                self.silence_counter += 1
                if self.silence_counter > self.max_silence_chunks:
                    # Перед остановкой отправляем сигнал тишины сервису
                    self.recognizer.send_silence_chunk(SILENCE_DURATION_MS)
                    self.status_label.setText("Status: Silence detected, stopping audio sending.")
                    self.speech_active = False
                    self.silence_counter = 0

    def on_audio_chunk(self, data):
        if self.speech_active:
            self.recognizer.send_audio(data)

    def on_partial_text(self, text):
        self.partial_text_edit.setPlainText(text)

    def on_final_text(self, text):
        self.final_text_edit.append(text)

    def on_final_refinement_text(self, text):
        self.final_refinement_text_edit.append(text)

    def on_error(self, error_msg):
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)

    def on_cost(self, cost):
        self.cost_label.setText(f"Estimated cost: {cost:.2f} ₽")

    def start(self):
        self.audio_listener.start()
        self.recognizer.start()

    def stop(self):
        self.audio_listener.stop()
        self.recognizer.stop()

def find_loopback_device():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and 'loopback' in dev['name'].lower():
            return i
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Speech Recognizer with RMS detection and GUI")
    parser.add_argument('--token', required=True, help='Yandex Cloud API key')
    parser.add_argument('--device', type=int, help='Audio input device index (loopback recommended)')
    parser.add_argument('--manual-audio', action='store_true', help='Use parec for audio input')
    args = parser.parse_args()

    device_index = args.device
    if not args.manual_audio:
        if device_index is None:
            device_index = find_loopback_device()
            if device_index is None:
                print("No loopback audio input device found. Please specify device index with --device.")
                print("Available devices:")
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        print(f"{i}: {dev['name']} (input channels: {dev['max_input_channels']})")
                sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args.token, use_parec=args.manual_audio, device_index=device_index)
    window.show()
    window.start()
    exit_code = app.exec_()
    window.stop()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()