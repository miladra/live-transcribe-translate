import sys
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QLabel, QComboBox, QCheckBox)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QThread, QTimer
from transcriber import Transcriber
from translator import Translator

class WorkerSignals(QObject):
    transcription_received = pyqtSignal(str)
    translation_received = pyqtSignal(str)
    explanation_received = pyqtSignal(str)

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
        self.signals.explanation_received.connect(self.handle_explanation)

        # Timer for debouncing selection
        self.selection_timer = QTimer()
        self.selection_timer.setSingleShot(True)
        self.selection_timer.timeout.connect(self.request_explanation)

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

        # Translation Toggle Row
        translation_toggle_layout = QHBoxLayout()
        self.translation_enabled = QCheckBox("Enable Translation")
        self.translation_enabled.setChecked(False)
        self.translation_enabled.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #2c3e50;
                font-size: 14px;
                padding: 10px 0;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        translation_toggle_layout.addWidget(self.translation_enabled)
        layout.addLayout(translation_toggle_layout)

        # Text Displays
        display_layout = QHBoxLayout()
        
        # Original Transcription
        trans_vbox = QVBoxLayout()
        trans_vbox.addWidget(QLabel("Transcription (Original):"))
        self.trans_display = QTextEdit()
        self.trans_display.setReadOnly(False)
        self.trans_display.setStyleSheet("background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;")
        self.trans_display.selectionChanged.connect(self.handle_selection_changed)
        trans_vbox.addWidget(self.trans_display)
        
        # Add controls for Transcription
        trans_btn_layout = QHBoxLayout()
        trans_copy_btn = QPushButton("Copy")
        trans_copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.trans_display))
        trans_copy_btn.setStyleSheet("background-color: #3498db; color: white; padding: 5px; border-radius: 3px;")
        trans_clear_btn = QPushButton("Clear")
        trans_clear_btn.clicked.connect(self.trans_display.clear)
        trans_clear_btn.setStyleSheet("background-color: #95a5a6; color: white; padding: 5px; border-radius: 3px;")
        trans_btn_layout.addWidget(trans_copy_btn)
        trans_btn_layout.addWidget(trans_clear_btn)
        
        self.ollama_btn = QPushButton("Query Ollama")
        self.ollama_btn.clicked.connect(self.request_full_explanation)
        self.ollama_btn.setStyleSheet("background-color: #f39c12; color: white; padding: 5px; border-radius: 3px;")
        trans_btn_layout.addWidget(self.ollama_btn)
        
        trans_vbox.addLayout(trans_btn_layout)
        
        display_layout.addLayout(trans_vbox)

        # Explanation (New Middle Box)
        expl_vbox = QVBoxLayout()
        expl_vbox.addWidget(QLabel("Explanation:"))
        self.expl_display = QTextEdit()
        self.expl_display.setReadOnly(False)
        self.expl_display.setPlaceholderText("Select text in transcription for explanation...")
        self.expl_display.setStyleSheet("background-color: #fff3cd; border-radius: 5px; border: 1px solid #ffeeba;")
        expl_vbox.addWidget(self.expl_display)
        
        # Add controls for Explanation
        expl_btn_layout = QHBoxLayout()
        expl_copy_btn = QPushButton("Copy")
        expl_copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.expl_display))
        expl_copy_btn.setStyleSheet("background-color: #3498db; color: white; padding: 5px; border-radius: 3px;")
        expl_clear_btn = QPushButton("Clear")
        expl_clear_btn.clicked.connect(self.expl_display.clear)
        expl_clear_btn.setStyleSheet("background-color: #95a5a6; color: white; padding: 5px; border-radius: 3px;")
        expl_btn_layout.addWidget(expl_copy_btn)
        expl_btn_layout.addWidget(expl_clear_btn)
        expl_vbox.addLayout(expl_btn_layout)
        
        display_layout.addLayout(expl_vbox)

        # Translation
        tl_vbox = QVBoxLayout()
        tl_vbox.addWidget(QLabel("Translation:"))
        self.tl_display = QTextEdit()
        self.tl_display.setReadOnly(False)
        self.tl_display.setStyleSheet("background-color: #e9ecef; border-radius: 5px; border: 1px solid #dee2e6;")
        tl_vbox.addWidget(self.tl_display)
        
        # Add controls for Translation
        tl_btn_layout = QHBoxLayout()
        tl_copy_btn = QPushButton("Copy")
        tl_copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.tl_display))
        tl_copy_btn.setStyleSheet("background-color: #3498db; color: white; padding: 5px; border-radius: 3px;")
        tl_clear_btn = QPushButton("Clear")
        tl_clear_btn.clicked.connect(self.tl_display.clear)
        tl_clear_btn.setStyleSheet("background-color: #95a5a6; color: white; padding: 5px; border-radius: 3px;")
        tl_btn_layout.addWidget(tl_copy_btn)
        tl_btn_layout.addWidget(tl_clear_btn)
        tl_vbox.addLayout(tl_btn_layout)
        
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
            # Send to translation thread if enabled
            if self.translation_enabled.isChecked():
                threading.Thread(target=self.process_translation, args=(text,), daemon=True).start()

    def process_translation(self, text):
        tl_text = self.translator.translate(text)
        self.signals.translation_received.emit(tl_text)

    def handle_translation(self, text):
        if text.strip():
            self.tl_display.append(text)

    def handle_selection_changed(self):
        # Start a short timer to avoid spamming calls while selecting
        self.selection_timer.start(500)

    def request_explanation(self):
        cursor = self.trans_display.textCursor()
        selected_text = cursor.selectedText()
        
        if selected_text.strip():
            # Get the block (context) where the selection is
            # We clone the cursor to not affect the UI selection
            context_cursor = self.trans_display.textCursor()
            context_cursor.select(context_cursor.SelectionType.BlockUnderCursor)
            context = context_cursor.selectedText()
            
            self.expl_display.setPlaceholderText("Explaining...")
            threading.Thread(target=self.process_explanation, args=(selected_text, context), daemon=True).start()

    def process_explanation(self, text, context):
        explanation = self.translator.explain(text, context)
        self.signals.explanation_received.emit(explanation)

    def request_full_explanation(self):
        text = self.trans_display.toPlainText()
        if text.strip():
            self.expl_display.clear()
            self.expl_display.setPlaceholderText("Ollama is thinking...")
            threading.Thread(target=self.process_full_explanation, args=(text,), daemon=True).start()

    def process_full_explanation(self, text):
        result = self.translator.simple_query(text)
        self.signals.explanation_received.emit(result)

    def handle_explanation(self, text):
        if text.strip():
            self.expl_display.setText(text)

    def change_language(self, lang):
        self.translator.set_target_language(lang)
        print(f"Language changed to {lang}")

    def change_model(self, model_name):
        self.translator.update_model(model_name)
        print(f"Model updated to {model_name}")

    def copy_to_clipboard(self, text_edit):
        text = text_edit.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText(f"Status: Copied to clipboard")
            # Clear status after 2 seconds
            QTimer.singleShot(2000, lambda: self.status_label.setText("Status: Ready"))

    def closeEvent(self, event):
        self.transcriber.cleanup()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
