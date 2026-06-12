# 🎙️ Automated RVC Voice Conversion & Google Drive Uploader

A fully automated, local pipeline that takes male audio inputs, converts them to high-fidelity, natural female voices using **Retrieval-based Voice Conversion (RVC v2)**, and automatically uploads the output to **Google Drive**, generating public, shareable links.

---

## 🌟 Key Features

1. **Local, High-Fidelity Conversion**: Uses RVC v2 under the hood for SOTA voice transfer that operates entirely locally on your machine.
2. **Breath & Emotion Retention**: Configured to preserve non-verbal cues (whispers, laughter, sighs, breaths) and structural speech cadence—no robotic, synthetic text-to-speech output.
3. **1-Click Setup & Startup**: Simple Windows batch scripts (`setup.bat` and `run.bat`) compile a Python virtual environment and run the pipeline without complex configuration.
4. **End-to-End Automation**: Automatically downloads the necessary RVC base feature extractors and female voice models on the first run.
5. **Headless Google Drive Upload**: Connects to the Google Drive API, uploads files, adjusts file sharing to public, and prints the links to the console.
6. **Premium Gradio Web UI**: Includes a modern browser interface with interactive waveforms, slider tuning, real-time logging, and batch folder processing.

---

## 📁 File Structure

```
Freelance_Contest/
├── inputs/                # Place male input audio here (e.g. Test IA.wav)
├── outputs/               # Converted female audio outputs
├── models/                # Downloaded RVC models (PTH and Index files)
├── setup.bat              # 1-Click Python environment installer
├── run.bat                # 1-Click Local Web UI launcher
├── pipeline.py            # Core Python automation backend & CLI tool
├── app.py                 # local Gradio Web UI server
├── README.md              # Documentation and guide
├── service_account.json   # (Optional) Headless Drive credentials
└── client_secrets.json    # (Optional) OAuth Drive credentials
```

---

## 🚀 Quick Start Guide

### Step 1: Prerequisites
1. **Python**: Ensure you have Python installed (Python 3.10 to 3.13 are fully supported). Verify with `python --version`.
2. **FFmpeg**: RVC requires FFmpeg on your system's PATH to parse audio. 
   * *If not installed, download it from [ffmpeg.org](https://ffmpeg.org/download.html) and add its `/bin` directory to your system Environment Variables.*

### Step 2: Setup Dependencies
Double-click [setup.bat](file:///c:/Users/skfir/Desktop/Serious/Freelance_Contest/setup.bat). This will:
* Set up a local Python virtual environment (`venv`).
* Install PyTorch and Torchaudio.
* Download and install Gradio, Google Drive API modules, and `rvc-python`.
* Create the `inputs`, `outputs`, and `models` folders.

### Step 3: Set Up Google Drive Integration (Optional)
To enable automated uploads, you need to provide credentials from the Google Cloud Console. Choose **one** of the two methods:

#### Method A: OAuth Authentication (Recommended)
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project, search for the **Google Drive API**, and click **Enable**.
3. Go to **Credentials**, click **Create Credentials** -> **OAuth client ID** (select *Desktop App*).
4. Download the JSON file, rename it to `client_secrets.json`, and place it in the root folder of this project.
5. *The first time you run the script, a browser tab will open asking you to log in. Once authenticated, a local `token.json` file is saved and subsequent runs will be 100% automated and headless.*

#### Method B: Service Account (Fully Headless)
1. In the Google Cloud Console Credentials tab, click **Create Credentials** -> **Service Account**.
2. Go to the service account page, click the **Keys** tab -> **Add Key** -> **Create new key** (JSON format).
3. Place this JSON file in the root folder and rename it to `service_account.json`.
4. *This runs headlessly from the first run. Make sure you share the target Google Drive folder with the Service Account email address.*

---

## 🖥️ Usage

### Option 1: Start the Local Web UI
Double-click [run.bat](file:///c:/Users/skfir/Desktop/Serious/Freelance_Contest/run.bat).
1. Open your browser and navigate to `http://127.0.0.1:7860`.
2. Drag and drop any male audio file into the box, adjust sliders if necessary, and click **Convert & Upload**.
3. View the converted female waveform in the browser and instantly copy the shareable Google Drive links.
4. Switch to the **Batch Conversion** tab to process an entire directory at once.

### Option 2: Command Line Interface (CLI)
You can call the Python script directly from the command line using the virtual environment interpreter:

* **Convert single file and upload**:
  ```bash
  venv\Scripts\python.exe pipeline.py --input inputs\Test_IA.wav
  ```
* **Batch convert a folder**:
  ```bash
  venv\Scripts\python.exe pipeline.py --input inputs\ --pitch 13 --index_rate 0.5
  ```
* **Process and upload to a specific Google Drive folder**:
  ```bash
  venv\Scripts\python.exe pipeline.py --input inputs\Test_IA.wav --folder_id "YOUR_DRIVE_FOLDER_ID"
  ```

---

## 🎛️ Tuning for Natural Quality

RVC voice conversion sounds highly human because it overlays a female vocal timbre on top of the original speaker's pronunciation, pauses, and emotional dynamics. To calibrate the voice for your input:

* **Spectral Denoising (Enabled by default)**: Estimates constant noise profiles from the silent frames and subtracts them. Essential for ASMR and quiet whisper tracks to remove background voice/hum without clipping. Disable with `--no_spectral_denoise`.
* **Pitch Shift (`--pitch`)**: Shifting by **`+13` semitones** delivers the most natural feminine tone for male whispering inputs.
* **Index Rate (`--index_rate`)**: Controls timbre retrieval. A value of **`0.5`** is the optimized default for ASMR, preventing phonetic leakage and keeping word pronunciation (like "Half") clean.
* **Protect Rate (`--protect`)**: Protects voiceless consonants and breath sounds. A value of **`0.33`** ensures that whispers and breath sounds are preserved naturally.
