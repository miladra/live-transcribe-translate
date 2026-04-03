# Live Transcribe & Translate

A real-time desktop application that captures audio from your microphone, transcribes it using local Whisper, and translates it using local Ollama models.

## Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com/)
3.  **Ollama Models**: Pull the models you want to use. Recommended:
    ```bash
    ollama pull gemma3:4b
    ollama pull gemma3:1b
    ollama pull translategemma:4b
    ```
4.  **System Dependencies** (macOS):
    ```bash
    brew install portaudio
    ```

## Installation

1.  Clone or download this repository.
2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

```bash
python main.py
```

## Features

-   **Real-time Transcription**: Uses `faster-whisper` for fast, local STT. Models are stored in the `models/` folder after the first download to ensure they are loaded locally thereafter.
-   **Local Translation**: Uses `ollama` with `gemma3:4b` (or chosen model) for privacy and offline translation.
-   **Interactive GUI**: Simple PyQt6 interface to start/stop listening, select target languages, and manage output with **Copy** and **Clear** buttons.
-   **Multi-language Support**: Easily switch target languages from the dropdown menu.
