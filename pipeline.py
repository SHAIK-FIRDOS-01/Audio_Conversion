import os
import sys
import time
import zipfile
import shutil
import argparse
import urllib.request
from pathlib import Path

import torch
try:
    from fairseq.data.dictionary import Dictionary
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Dictionary])
except Exception as e:
    pass

# Google API imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    GDRIVE_LIBS_AVAILABLE = True
except ImportError:
    GDRIVE_LIBS_AVAILABLE = False

# RVC Python imports
try:
    import rvc_python
    from rvc_python.infer import RVCInference
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False


# ==========================================
# 1. Downloader Utilities
# ==========================================

def download_file(url, dest_path):
    """Downloads a file showing progress and download speed in real-time."""
    print(f"[INFO] Downloading {os.path.basename(dest_path)}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1MB
            downloaded = 0
            
            with open(dest_path, 'wb') as out_file:
                start_time = time.time()
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB) - Speed: {speed:.2f} MB/s", end="", flush=True)
                    else:
                        print(f"\rDownloaded: {downloaded / 1024 / 1024:.1f} MB", end="", flush=True)
            print("\n[INFO] Download completed successfully.")
            return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {url}: {e}")
        return False


def setup_rvc_base_models():
    """Locates the internal rvc_python package and ensures base models are downloaded."""
    if not RVC_AVAILABLE:
        print("[WARNING] rvc-python package is not installed yet. Base models setup will be skipped until run.")
        return False
        
    pkg_dir = os.path.dirname(rvc_python.__file__)
    base_model_dir = os.path.join(pkg_dir, "base_model")
    os.makedirs(base_model_dir, exist_ok=True)
    
    hubert_path = os.path.join(base_model_dir, "hubert_base.pt")
    rmvpe_path = os.path.join(base_model_dir, "rmvpe.pt")
    
    # Standard RVC base URLs
    hubert_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
    rmvpe_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    
    success = True
    if not os.path.exists(hubert_path):
        print("[INFO] RVC base model hubert_base.pt is missing.")
        success = success and download_file(hubert_url, hubert_path)
    else:
        print(f"[INFO] Found base model: {hubert_path}")
        
    if not os.path.exists(rmvpe_path):
        print("[INFO] RVC pitch model rmvpe.pt is missing.")
        success = success and download_file(rmvpe_url, rmvpe_path)
    else:
        print(f"[INFO] Found base model: {rmvpe_path}")
        
    return success


def download_and_extract_voice_model(model_url, model_name):
    """Downloads a target voice model (zip) and extracts .pth and .index files."""
    dest_dir = os.path.join("models", model_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    # Check if files already exist
    pth_files = list(Path(dest_dir).rglob("*.pth"))
    index_files = list(Path(dest_dir).rglob("*.index"))
    
    if pth_files:
        print(f"[INFO] RVC model '{model_name}' already exists locally.")
        pth_path = str(pth_files[0])
        index_path = str(index_files[0]) if index_files else ""
        return pth_path, index_path
        
    zip_path = os.path.join("models", f"{model_name}.zip")
    if not download_file(model_url, zip_path):
        raise RuntimeError(f"Could not download model zip from {model_url}")
        
    print(f"[INFO] Extracting voice model zip...")
    extract_dir = os.path.join("models", f"{model_name}_temp")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Search for .pth and .index files
    temp_pth_files = list(Path(extract_dir).rglob("*.pth"))
    temp_index_files = list(Path(extract_dir).rglob("*.index"))
    
    if not temp_pth_files:
        # Clean up
        shutil.rmtree(extract_dir)
        os.remove(zip_path)
        raise RuntimeError("No .pth model weight file found in the zip archive.")
        
    # Copy files to clean model folder
    final_pth = os.path.join(dest_dir, os.path.basename(temp_pth_files[0]))
    shutil.copy2(temp_pth_files[0], final_pth)
    
    final_index = ""
    if temp_index_files:
        final_index = os.path.join(dest_dir, os.path.basename(temp_index_files[0]))
        shutil.copy2(temp_index_files[0], final_index)
        
    # Clean up temp files
    shutil.rmtree(extract_dir)
    os.remove(zip_path)
    
    print(f"[INFO] Voice model extracted to '{dest_dir}'.")
    return final_pth, final_index


# ==========================================
# 2. Google Drive API Upload
# ==========================================

def get_gdrive_service():
    """Initializes Google Drive service using Service Account or OAuth credentials."""
    if not GDRIVE_LIBS_AVAILABLE:
        print("[ERROR] Google Client libraries are missing. Run setup.bat first.")
        return None
        
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None
    
    # Method 1: Service Account JSON (Fully Headless)
    if os.path.exists('service_account.json'):
        print("[INFO] Authenticating using local service_account.json key...")
        try:
            creds = service_account.Credentials.from_service_account_file(
                'service_account.json', scopes=SCOPES
            )
            return build('drive', 'v3', credentials=creds)
        except Exception as e:
            print(f"[ERROR] Service account login failed: {e}")
            
    # Method 2: OAuth Client Credentials (token.json/client_secrets.json)
    if os.path.exists('token.json'):
        print("[INFO] Loading credentials from token.json...")
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[INFO] Credentials expired. Refreshing token...")
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
                
        if creds is None:
            if not os.path.exists('client_secrets.json'):
                print("[ERROR] Neither 'service_account.json' nor 'client_secrets.json' was found.")
                print("[INFO] Please place your credentials file in this folder to enable automated uploads.")
                return None
                
            print("[INFO] Initializing browser-based OAuth flow using client_secrets.json...")
            flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save local token
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return build('drive', 'v3', credentials=creds)


def upload_to_gdrive(file_path, folder_id=None):
    """Uploads file to Google Drive, sets sharing to public, and returns links."""
    service = get_gdrive_service()
    if not service:
        print("[WARNING] Skipping Google Drive upload due to missing credentials.")
        return None, None
        
    print(f"[INFO] Uploading {os.path.basename(file_path)} to Google Drive...")
    file_metadata = {
        'name': os.path.basename(file_path)
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]
        
    media = MediaFileUpload(file_path, mimetype='audio/wav', resumable=True)
    try:
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id, webViewLink'
        ).execute()
        file_id = file.get('id')
        web_view_link = file.get('webViewLink')
        
        # Change permissions to anyone with the link can view
        print("[INFO] Setting file permissions to public read...")
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(fileId=file_id, body=permission).execute()
        
        # Direct download link construction
        direct_link = f"https://docs.google.com/uc?export=download&id={file_id}"
        print(f"[INFO] Upload completed successfully.")
        print(f"[INFO] Web Link: {web_view_link}")
        print(f"[INFO] Direct Link: {direct_link}")
        return web_view_link, direct_link
    except Exception as e:
        print(f"[ERROR] Google Drive upload failed: {e}")
        return None, None


# ==========================================
# 3. Main Processing Engine
# ==========================================

def apply_noise_gate(input_path, output_path, threshold_db=-50.0, attack_ms=10, release_ms=100):
    """Applies a smooth block-based noise gate to the audio file."""
    import soundfile as sf
    import numpy as np
    
    if threshold_db is None or threshold_db <= -99.0:
        # Bypassed
        shutil.copy2(input_path, output_path)
        return
        
    data, samplerate = sf.read(input_path)
    is_stereo = len(data.shape) > 1
    
    if is_stereo:
        mono_data = np.mean(data, axis=1)
    else:
        mono_data = data
        
    threshold = 10 ** (threshold_db / 20)
    
    # 5ms blocks
    block_size = int(samplerate * 0.005)
    if block_size == 0:
        block_size = 1
    num_blocks = len(mono_data) // block_size
    
    if num_blocks == 0:
        shutil.copy2(input_path, output_path)
        return
        
    block_rms = []
    for i in range(num_blocks):
        block = mono_data[i*block_size : (i+1)*block_size]
        block_rms.append(np.sqrt(np.mean(block**2) + 1e-10))
    block_rms = np.array(block_rms)
    
    block_gains = np.ones_like(block_rms)
    current_gain = 1.0
    
    block_dt = 0.005 # 5ms
    alpha_attack = np.exp(-block_dt / (attack_ms / 1000.0))
    alpha_release = np.exp(-block_dt / (release_ms / 1000.0))
    
    for i in range(num_blocks):
        rms = block_rms[i]
        target_gain = 1.0 if rms >= threshold else 0.0
        
        if target_gain > current_gain:
            current_gain = alpha_attack * current_gain + (1.0 - alpha_attack) * target_gain
        else:
            current_gain = alpha_release * current_gain + (1.0 - alpha_release) * target_gain
            
        block_gains[i] = current_gain
        
    sample_gains = np.interp(
        np.arange(len(mono_data)),
        np.arange(num_blocks) * block_size + block_size // 2,
        block_gains,
        left=block_gains[0],
        right=block_gains[-1]
    )
    
    if is_stereo:
        gated_data = data * sample_gains[:, np.newaxis]
    else:
        gated_data = data * sample_gains
        
    sf.write(output_path, gated_data, samplerate)
    print(f"[INFO] Applied noise gate (threshold: {threshold_db} dB) to: {os.path.basename(input_path)}")


def apply_spectral_noise_reduction(input_path, output_path, noise_threshold=2.5, noise_floor_sec=0.5):
    """Applies spectral noise subtraction to remove stationary background noise."""
    import soundfile as sf
    import numpy as np
    import scipy.signal
    
    data, sr = sf.read(input_path)
    is_stereo = len(data.shape) > 1
    if is_stereo:
        channels = [data[:, i] for i in range(data.shape[1])]
    else:
        channels = [data]
        
    denoised_channels = []
    for ch in channels:
        nperseg = 2048
        noverlap = nperseg // 4
        # STFT
        f, t, Zxx = scipy.signal.stft(ch, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        # Estimate noise from the first noise_floor_sec (assumed quiet/silence)
        noise_frames = int(noise_floor_sec * sr / (nperseg - noverlap))
        noise_frames = max(5, min(noise_frames, Zxx.shape[1]))
        
        noise_profile = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1, keepdims=True)
        
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Subtract noise magnitude
        mag_clean = mag - noise_threshold * noise_profile
        mag_clean = np.maximum(mag_clean, 0.0)
        
        # ISTFT
        Zxx_clean = mag_clean * np.exp(1j * phase)
        _, ch_clean = scipy.signal.istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        if len(ch_clean) < len(ch):
            ch_clean = np.pad(ch_clean, (0, len(ch) - len(ch_clean)))
        else:
            ch_clean = ch_clean[:len(ch)]
        denoised_channels.append(ch_clean)
        
    if is_stereo:
        clean_data = np.stack(denoised_channels, axis=1)
    else:
        clean_data = denoised_channels[0]
        
    sf.write(output_path, clean_data, sr)
    print(f"[INFO] Applied spectral noise reduction (threshold: {noise_threshold}) to: {os.path.basename(input_path)}")


def apply_normalization(file_path, target_db=-3.0):
    """Normalizes the peak amplitude of the WAV file to target_db."""
    import soundfile as sf
    import numpy as np
    
    if target_db is None or target_db >= 0.0:
        # Bypassed
        return
        
    data, samplerate = sf.read(file_path)
    peak = np.max(np.abs(data))
    if peak == 0:
        return
        
    target_amplitude = 10 ** (target_db / 20)
    gain = target_amplitude / peak
    normalized_data = data * gain
    sf.write(file_path, normalized_data, samplerate)
    print(f"[INFO] Normalized peak of {os.path.basename(file_path)} to {target_db} dB (gain applied: {20 * np.log10(gain):.2f} dB)")


def convert_voice(input_path, output_path, model_pth, model_index, pitch_shift=13, index_rate=0.5, protect_index=0.33, rms_mix_rate=0.25):
    """Runs local RVC Voice Conversion on an audio file."""
    if not RVC_AVAILABLE:
        raise RuntimeError("RVC libraries are not loaded. Run setup.bat first.")
        
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    print(f"[INFO] Converting voice on: {input_path}")
    print(f"       Using model: {os.path.basename(model_pth)}")
    print(f"       Pitch Shift: {pitch_shift} semitones")
    print(f"       Index Rate: {index_rate} | Protect Consonants: {protect_index} | RMS Mix Rate: {rms_mix_rate}")
    
    start_time = time.time()
    # Call RVCInference class for execution
    try:
        print("[INFO] Loading model as RVC v2...")
        rvc = RVCInference(device="cpu:0", model_path=model_pth, index_path=model_index, version="v2")
    except Exception as e:
        if "size mismatch" in str(e) or "shape" in str(e).lower() or "emb_phone" in str(e):
            print("[INFO] Model shape mismatch. Falling back to RVC v1...")
            rvc = RVCInference(device="cpu:0", model_path=model_pth, index_path=model_index, version="v1")
        else:
            raise e
            
    rvc.set_params(
        f0up_key=pitch_shift,
        f0method="rmvpe",
        index_rate=index_rate,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=rms_mix_rate,
        protect=protect_index
    )
    
    try:
        result = rvc.infer_file(input_path, output_path)
    except Exception as e:
        if "size mismatch" in str(e) or "shape" in str(e).lower() or "emb_phone" in str(e):
            print("[INFO] Model shape mismatch during inference. Retrying as RVC v1...")
            rvc = RVCInference(device="cpu:0", model_path=model_pth, index_path=model_index, version="v1")
            rvc.set_params(
                f0up_key=pitch_shift,
                f0method="rmvpe",
                index_rate=index_rate,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=rms_mix_rate,
                protect=protect_index
            )
            result = rvc.infer_file(input_path, output_path)
        else:
            raise e
    elapsed = time.time() - start_time
    print(f"[INFO] Voice conversion complete in {elapsed:.1f}s. Saved to: {output_path}")
    return result


def process_audio_pipeline(input_path, model_url=None, pitch_shift=13, index_rate=0.5, protect_index=0.33, rms_mix_rate=0.25, noise_gate_db=-100.0, normalize_db=-3.0, folder_id=None, spectral_denoise=False, spectral_denoise_threshold=2.5):
    """Integrates downloading models, voice conversion, and drive upload."""
    global RVC_AVAILABLE, GDRIVE_LIBS_AVAILABLE
    
    # Dynamically re-import in case setup just finished
    if not RVC_AVAILABLE:
        try:
            import rvc_python
            from rvc_python.infer import RVCInference
            RVC_AVAILABLE = True
        except ImportError:
            pass
            
    if not GDRIVE_LIBS_AVAILABLE:
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google.oauth2 import service_account
            GDRIVE_LIBS_AVAILABLE = True
        except ImportError:
            pass
            
    if not RVC_AVAILABLE:
        print("[ERROR] RVC libraries are not installed. Please run setup.bat first.")
        return None
        
    # 1. Setup Base RVC Models
    setup_rvc_base_models()
    
    # 2. Get Voice Model
    if not model_url:
        # Default high-quality Ayaka RVC voice model
        model_url = "https://huggingface.co/ArkanDash/rvc-genshin-impact/resolve/main/prezipped/v1/ayaka-jp%20100%20epochs%2040k.zip"
        model_name = "ayaka-jp"
    else:
        model_name = "custom_voice"
        
    model_pth, model_index = download_and_extract_voice_model(model_url, model_name)
    
    # 3. Handle Single File vs Batch Directories
    input_files = []
    if os.path.isdir(input_path):
        extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
        input_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if os.path.splitext(f.lower())[1] in extensions
        ]
        if not input_files:
            print(f"[WARNING] No audio files found in directory: {input_path}")
            return {}
        print(f"[INFO] Found {len(input_files)} files to process in batch mode.")
    else:
        if not os.path.exists(input_path):
            print(f"[ERROR] Input audio file does not exist: {input_path}")
            return {}
        input_files = [input_path]
        
    results = {}
    for idx, file in enumerate(input_files):
        print(f"\n--- Processing [{idx + 1}/{len(input_files)}]: {os.path.basename(file)} ---")
        
        # Create output name
        base_name = os.path.splitext(os.path.basename(file))[0]
        output_name = f"{base_name}_female.wav"
        output_file = os.path.join("outputs", output_name)
        os.makedirs("outputs", exist_ok=True)
        
        # Apply pre-conversion noise reduction or noise gate
        temp_input = file
        is_temp = False
        if spectral_denoise:
            temp_input = os.path.join("outputs", f"{base_name}_denoised_temp.wav")
            try:
                apply_spectral_noise_reduction(file, temp_input, noise_threshold=spectral_denoise_threshold)
                is_temp = True
            except Exception as e:
                print(f"[WARNING] Failed to apply spectral denoise: {e}. Processing original file instead.")
                temp_input = file
                is_temp = False
        elif noise_gate_db is not None and noise_gate_db > -99.0:
            temp_input = os.path.join("outputs", f"{base_name}_gated_temp.wav")
            try:
                apply_noise_gate(file, temp_input, threshold_db=noise_gate_db)
                is_temp = True
            except Exception as e:
                print(f"[WARNING] Failed to apply noise gate: {e}. Processing original file instead.")
                temp_input = file
                is_temp = False
        
        # Run conversion
        try:
            convert_voice(
                input_path=temp_input,
                output_path=output_file,
                model_pth=model_pth,
                model_index=model_index,
                pitch_shift=pitch_shift,
                index_rate=index_rate,
                protect_index=protect_index,
                rms_mix_rate=rms_mix_rate
            )
            
            # Clean up temp gated file
            if is_temp and os.path.exists(temp_input):
                try:
                    os.remove(temp_input)
                except Exception:
                    pass
            
            # Apply peak normalization
            if normalize_db is not None and normalize_db < 0.0:
                try:
                    apply_normalization(output_file, target_db=normalize_db)
                except Exception as e:
                    print(f"[WARNING] Failed to normalize output: {e}")
            
            # Upload to Google Drive
            web_link, direct_link = upload_to_gdrive(output_file, folder_id)
            results[file] = {
                "output_file": output_file,
                "web_link": web_link,
                "direct_link": direct_link
            }
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
            if is_temp and os.path.exists(temp_input):
                try:
                    os.remove(temp_input)
                except Exception:
                    pass
            results[file] = {
                "error": str(e)
            }
            
    print("\n===================================================")
    print("                 PIPELINE RESULTS                  ")
    print("===================================================")
    for src, res in results.items():
        name = os.path.basename(src)
        if "error" in res:
            print(f"{name}: [FAILED] - {res['error']}")
        else:
            print(f"{name}: [SUCCESS]")
            print(f"  Local file:  {res['output_file']}")
            if res['web_link']:
                print(f"  Drive Link:  {res['web_link']}")
                print(f"  Direct Link: {res['direct_link']}")
            else:
                print("  Drive Link:  [SKIPPED / AUTH MISSING]")
    print("===================================================")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Automated RVC Male-to-Female voice conversion and Google Drive upload pipeline.")
    parser.add_argument("--input", type=str, default="inputs", help="Path to input audio file or inputs folder (default: inputs)")
    parser.add_argument("--model_url", type=str, default=None, help="Hugging Face zip URL of RVC model (default: Ayaka JP RVC v2)")
    parser.add_argument("--pitch", type=int, default=13, help="Semitone pitch shift, male-to-female defaults to +13 (default: 13)")
    parser.add_argument("--index_rate", type=float, default=0.5, help="Retrieval index influence rate, 0.0-1.0 (default: 0.5)")
    parser.add_argument("--protect", type=float, default=0.33, help="Voiceless consonant and breath protection index, 0.0-0.5 (default: 0.33)")
    parser.add_argument("--rms_mix_rate", type=float, default=0.25, help="RMS mix rate, 0.0-1.0 (default: 0.25)")
    parser.add_argument("--noise_gate_db", type=float, default=-100.0, help="Noise gate threshold in dB (set to -100 to disable, default: -100.0)")
    parser.add_argument("--normalize_db", type=float, default=-3.0, help="Peak normalization in dB (set to 0.0 to skip, default: -3.0)")
    parser.add_argument("--folder_id", type=str, default=None, help="Optional Google Drive folder ID to upload files directly to")
    parser.add_argument("--spectral_denoise", action="store_true", help="Enable the high-quality spectral noise reduction")
    parser.add_argument("--spectral_denoise_threshold", type=float, default=2.5, help="Spectral noise subtraction threshold multiplier (default: 2.5)")
    parser.set_defaults(spectral_denoise=False)
    
    args = parser.parse_args()
    
    # Make sure default folders exist
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    process_audio_pipeline(
        input_path=args.input,
        model_url=args.model_url,
        pitch_shift=args.pitch,
        index_rate=args.index_rate,
        protect_index=args.protect,
        rms_mix_rate=args.rms_mix_rate,
        noise_gate_db=args.noise_gate_db,
        normalize_db=args.normalize_db,
        folder_id=args.folder_id,
        spectral_denoise=args.spectral_denoise,
        spectral_denoise_threshold=args.spectral_denoise_threshold
    )
