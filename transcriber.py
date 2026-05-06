import os
import signal
import sys
import queue
import threading
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        print(f"Loading Whisper model '{model_size}' from local storage...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        # Using a slightly larger model for better accuracy
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=model_path)
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.transcription_callback = None
        self.history = "" # Keep track of recent transcription for context

        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1600 # 100ms
        self.pa = pyaudio.PyAudio()
        self.input_device_index = None

    def get_input_devices(self):
        devices = []
        info = self.pa.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for i in range(0, num_devices):
            device_info = self.pa.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name')
                })
        return devices

    def start(self, callback, device_index=None):
        if self.is_running:
            print("Transcriber already running.")
            return

        self.transcription_callback = callback
        self.input_device_index = device_index
        self.is_running = True
        
        # Clear queue before starting
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback
            )
            print("Transcriber started.")
            self._process_audio() # Blocking call
        except Exception as e:
            print(f"Failed to start transcriber: {e}")
            self.is_running = False

    def stop(self):
        self.is_running = False
        if hasattr(self, 'stream'):
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                del self.stream
            except Exception as e:
                print(f"Error closing stream: {e}")
        
        # Drain the queue to unblock any pending processing
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        print("Transcriber stopped.")

    def cleanup(self):
        self.stop()
        if hasattr(self, 'pa'):
            self.pa.terminate()
        print("PyAudio terminated.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def _process_audio(self):
        # Use a lock to ensure only one processing loop runs at a time
        if not hasattr(self, '_processing_lock'):
            self._processing_lock = threading.Lock()
            
        if not self._processing_lock.acquire(blocking=False):
            print("Processing loop already running.")
            return

        try:
            # Accumulate audio for better transcription segments
            audio_buffer = []
            
            while self.is_running:
                try:
                    # Get data from queue with shorter timeout for faster response to stop
                    data = self.audio_queue.get(timeout=0.2)
                    audio_buffer.append(np.frombuffer(data, dtype=np.int16))
                    
                    # Increase buffer to 3 seconds for better context/accuracy (30 * 100ms)
                    if len(audio_buffer) >= 30: 
                        audio_data = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                        audio_buffer = [] # Clear for next segment
                        
                        # Using larger beam size and initial prompt for better coherence
                        segments, info = self.model.transcribe(
                            audio_data, 
                            beam_size=10, 
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            initial_prompt=self.history[-200:] if self.history else None
                        )
                        
                        for segment in segments:
                            if self.transcription_callback:
                                text = segment.text.strip()
                                if text:
                                    self.history += " " + text
                                    self.transcription_callback(text)
                                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in transcription loop: {e}")
        finally:
            self.is_running = False
            self._processing_lock.release()
            print("Processing loop finished.")
