# 🎙️ Automated RVC Voice Conversion & Google Drive Uploader

A local pipeline that takes male audio inputs, converts them to high-fidelity female voices using **Retrieval-based Voice Conversion (RVC v2)**, and automatically uploads the output to **Google Drive**, generating public, shareable links.

---

## 🌟 Key Features

1. **Local, High-Fidelity Conversion**: Uses RVC v2 under the hood for voice transfer that operates entirely locally on your machine.
2. **Breath & Emotion Retention**: Configured to preserve non-verbal cues (whispers, breaths, laughter) and speech cadence.
3. **1-Click Setup & Startup**: Simple Windows batch scripts (`setup.bat` and `run.bat`) compile the Python virtual environment and run the pipeline.
4. **Google Drive Upload**: Connects to the Google Drive API, uploads files, adjusts file sharing to public, and displays the direct download links.
5. **Gradio Web UI**: Includes a clean interface with interactive waveforms, slider tuning, and batch folder processing.

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
├── app.py                 # Local Gradio Web UI server
├── README.md              # Project documentation
└── client_secrets.json    # Google OAuth credentials (place here once downloaded)
```

---

## 🚀 Easy Setup Guide

### Step 1: Install Python & FFmpeg
1. **Python**: Ensure you have Python installed. Verify with `python --version` in your terminal.
2. **FFmpeg**: RVC requires FFmpeg on your system's PATH to process audio. 
   - *If not installed, download it from [ffmpeg.org](https://ffmpeg.org/download.html) and add its `/bin` directory to your system Environment Variables.*

### Step 2: Install Dependencies
Double-click [setup.bat](file:///c:/Users/skfir/Desktop/Serious/Freelance_Contest/setup.bat). This will:
- Set up a local Python virtual environment (`venv`).
- Download and install all required modules (PyTorch, Gradio, Google API modules, and `rvc-python`).

### Step 3: Google Drive API Setup (Simplified)
To allow the app to automatically upload files to your Google Drive, follow these steps:

1. **Enable the Drive API:**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project, search for the **Google Drive API**, and click **Enable**.

2. **Configure OAuth Consent Screen:**
   - On the left sidebar menu, click **APIs & Services** > **OAuth consent screen** (or Google Auth Platform).
   - If prompted, click **Get Started** / select **External**, and click **Create**.
   - **App Information**: Fill in an **App name** (e.g. `RVC Uploader`) and select your email under **User support email** and **Developer contact email**. Click Save/Continue.
   - **Test Users**: Under this step, click **+ ADD USERS**, type in your personal Gmail address, click **Add/Save**, and then finish the wizard. *(This ensures Google allows you to log in while the app is in development).*

3. **Download your Credentials file:**
   - On the left sidebar, click **Credentials**.
   - Click the **+ CREATE CREDENTIALS** button at the top and select **OAuth client ID**.
   - Choose **Desktop app** as the Application type, give it a name, and click **Create**.
   - A popup will show saying "OAuth client created". Click **Download JSON**.
   - Rename that downloaded file to exactly: **`client_secrets.json`**
   - Place it directly into your main project folder:
     `c:\Users\skfir\Desktop\Serious\Freelance_Contest\client_secrets.json`

---

## 🖥️ How to Run the App

1. Double-click [run.bat](file:///c:/Users/skfir/Desktop/Serious/Freelance_Contest/run.bat).
2. Open your web browser and go to `http://127.0.0.1:7860`.
3. Drag and drop any male audio file into the box, adjust sliders if necessary, and click **Convert & Upload**.
4. **First-time Login (Only once):**
   - A browser tab will automatically open asking you to sign in with your Google account.
   - Log in using the **same Gmail address** you added to the "Test Users" list.
   - Google will show a warning screen saying "Google hasn't verified this app". Click **Advanced** (bottom-left) and then click **Go to [App Name] (unsafe)** to bypass.
   - Click **Allow / Continue** to confirm permissions.
   - *This creates a local `token.json` file. All future uploads will now happen silently in the background.*
5. View the converted female waveform in the browser and copy the shareable Google Drive links.

---

## 🎛️ Tuning for Natural Quality

* **Spectral Denoising (Enabled by default)**: Removes background hum/noise without clipping. Essential for ASMR and quiet whisper tracks.
* **Pitch Shift (`--pitch`)**: Shifting by **`+13` semitones** delivers the most natural feminine tone for male whispering inputs.
* **Index Rate (`--index_rate`)**: Controls voice timbre retrieval. A value of **`0.5`** is the optimized default for keeping word pronunciation clean.
* **Protect Rate (`--protect`)**: Protects voiceless consonants and breath sounds. A value of **`0.33`** ensures that whispers and breath sounds are preserved naturally.
