import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QLabel, QComboBox)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QThread
from transcriber import Transcriber
from translator import Translator

class WorkerSignals(QObject):
    transcription_received = pyqtSignal(str)
    translation_received = pyqtSignal(str)

class TranscribeThread(QThread):
    def __init__(self, transcriber, signals, device_index=None):
        super().__init__()
        self.transcriber = transcriber
        self.signals = signals
        self.device_index = device_index

    def run(self):
        self.transcriber.start(self.signals.transcription_received.emit, self.device_index)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Transcribe & Translate")
        self.setMinimumSize(800, 600)

        # Initialize components
        self.transcriber = Transcriber(model_size="small")
        self.translator = Translator(model="gemma3:4b")
        self.signals = WorkerSignals()
        
        # Connect signals
        self.signals.transcription_received.connect(self.handle_transcription)
        self.signals.translation_received.connect(self.handle_translation)

        # UI Setup
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Controls
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Listening")
        self.start_btn.clicked.connect(self.toggle_listening)
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px; font-weight: bold;")
        controls_layout.addWidget(self.start_btn)

        self.lang_selector = QComboBox()
        self.lang_selector.addItems(["Persian", "English", "German", "Spanish", "French"])
        self.lang_selector.currentTextChanged.connect(self.change_language)
        controls_layout.addWidget(QLabel("Target Language:"))
        controls_layout.addWidget(self.lang_selector)

        # Device Selector
        self.device_selector = QComboBox()
        self.devices = self.transcriber.get_input_devices()
        for dev in self.devices:
            self.device_selector.addItem(dev['name'], dev['index'])
        
        controls_layout.addWidget(QLabel("Input Device:"))
        controls_layout.addWidget(self.device_selector)

        # Model Selector
        self.model_selector = QComboBox()
        available_models = self.translator.get_local_models()
        self.model_selector.addItems(available_models)
        # Try to select gemma3:4b if available
        index = self.model_selector.findText("gemma3:4b")
        if index >= 0:
            self.model_selector.setCurrentIndex(index)
        self.model_selector.currentTextChanged.connect(self.change_model)

        controls_layout.addWidget(QLabel("Ollama Model:"))
        controls_layout.addWidget(self.model_selector)
        
        layout.addLayout(controls_layout)

        # Text Displays
        display_layout = QHBoxLayout()
        
        # Original Transcription
        trans_vbox = QVBoxLayout()
        trans_vbox.addWidget(QLabel("Transcription (Original):"))
        self.trans_display = QTextEdit()
        self.trans_display.setReadOnly(True)
        self.trans_display.setStyleSheet("background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;")
        trans_vbox.addWidget(self.trans_display)
        display_layout.addLayout(trans_vbox)

        # Translation
        tl_vbox = QVBoxLayout()
        tl_vbox.addWidget(QLabel("Translation:"))
        self.tl_display = QTextEdit()
        self.tl_display.setReadOnly(True)
        self.tl_display.setStyleSheet("background-color: #e9ecef; border-radius: 5px; border: 1px solid #dee2e6;")
        tl_vbox.addWidget(self.tl_display)
        display_layout.addLayout(tl_vbox)

        layout.addLayout(display_layout)

        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

    def toggle_listening(self):
        if not self.transcriber.is_running:
            device_index = self.device_selector.currentData()
            self.trans_thread = TranscribeThread(self.transcriber, self.signals, device_index)
            self.trans_thread.start()
            self.start_btn.setText("Stop Listening")
            self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px; font-weight: bold;")
            self.status_label.setText("Status: Listening...")
            self.device_selector.setEnabled(False)
        else:
            self.transcriber.stop()
            self.start_btn.setText("Start Listening")
            self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px; font-weight: bold;")
            self.status_label.setText("Status: Stopped")
            self.device_selector.setEnabled(True)

    def handle_transcription(self, text):
        if text.strip():
            self.trans_display.append(text)
            # Send to translation thread (or simulate here for simplicity)
            threading.Thread(target=self.process_translation, args=(text,), daemon=True).start()

    def process_translation(self, text):
        tl_text = self.translator.translate(text)
        self.signals.translation_received.emit(tl_text)

    def handle_translation(self, text):
        if text.strip():
            self.tl_display.append(text)

    def change_language(self, lang):
        self.translator.set_target_language(lang)
        print(f"Language changed to {lang}")

    def change_model(self, model_name):
        self.translator.update_model(model_name)
        print(f"Model updated to {model_name}")

    def closeEvent(self, event):
        self.transcriber.stop()
        event.accept()

import threading # Added missing import

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
