import os
import shutil
import gradio as gr
import torch
try:
    from fairseq.data.dictionary import Dictionary
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Dictionary])
except Exception as e:
    pass
from pipeline import process_audio_pipeline, RVC_AVAILABLE, GDRIVE_LIBS_AVAILABLE

# Ensure folders exist
os.makedirs("inputs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Default Model URL (Ayaka JP RVC)
DEFAULT_MODEL_URL = "https://huggingface.co/ArkanDash/rvc-genshin-impact/resolve/main/prezipped/v1/ayaka-jp%20100%20epochs%2040k.zip"


def check_system_status():
    """Returns html describing missing requirements to guide the user."""
    rvc_status = "<span style='color: green; font-weight: bold;'>✓ Installed</span>" if RVC_AVAILABLE else "<span style='color: red; font-weight: bold;'>✗ Missing (Run setup.bat)</span>"
    gdrive_status = "<span style='color: green; font-weight: bold;'>✓ Ready</span>" if GDRIVE_LIBS_AVAILABLE else "<span style='color: orange; font-weight: bold;'>⚠ Missing (Optional, install for Drive)</span>"
    
    # Check for credentials
    creds_status = "<span style='color: red; font-weight: bold;'>✗ Missing (Uploads Disabled)</span>"
    if os.path.exists('service_account.json'):
        creds_status = "<span style='color: green; font-weight: bold;'>✓ Service Account Key Found</span>"
    elif os.path.exists('token.json'):
        creds_status = "<span style='color: green; font-weight: bold;'>✓ OAuth Token Found</span>"
    elif os.path.exists('client_secrets.json'):
        creds_status = "<span style='color: orange; font-weight: bold;'>✓ Client Secrets Found (Will authenticate on run)</span>"
        
    return f"""
    <div style='background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin-bottom: 20px;'>
        <h3 style='margin-top: 0;'>System Requirements Status</h3>
        <ul style='list-style-type: none; padding-left: 0; margin-bottom: 0;'>
            <li style='margin-bottom: 8px;'>🤖 <b>RVC Inference Engine</b>: {rvc_status}</li>
            <li style='margin-bottom: 8px;'>📁 <b>Google API Libraries</b>: {gdrive_status}</li>
            <li style='margin-bottom: 0;'>🔑 <b>Google Drive Credentials</b>: {creds_status}</li>
        </ul>
    </div>
    """


def gradio_single_convert(input_audio, model_url, pitch, index_rate, protect, rms_mix_rate, noise_gate_db, normalize_db, folder_id, spectral_denoise, spectral_denoise_threshold):
    if not input_audio:
        return None, "Error: Please upload an audio file.", "", ""
    
    # Copy file to inputs directory to process
    filename = os.path.basename(input_audio)
    dest_input = os.path.join("inputs", filename)
    try:
        shutil.copy2(input_audio, dest_input)
    except Exception as e:
        return None, f"Error caching file: {e}", "", ""
    
    # Run the automated pipeline
    results = process_audio_pipeline(
        input_path=dest_input,
        model_url=model_url.strip() if model_url.strip() else None,
        pitch_shift=int(pitch),
        index_rate=float(index_rate),
        protect_index=float(protect),
        rms_mix_rate=float(rms_mix_rate),
        noise_gate_db=float(noise_gate_db),
        normalize_db=float(normalize_db),
        folder_id=folder_id.strip() if folder_id.strip() else None,
        spectral_denoise=bool(spectral_denoise),
        spectral_denoise_threshold=float(spectral_denoise_threshold)
    )
    
    if dest_input not in results:
        return None, "Error: Processing pipeline failed.", "", ""
        
    res = results[dest_input]
    if "error" in res:
        return None, f"Error during voice conversion: {res['error']}", "", ""
        
    output_audio = res["output_file"]
    web_link = res["web_link"] if res["web_link"] else "Skipped (Credentials missing)"
    direct_link = res["direct_link"] if res["direct_link"] else "Skipped (Credentials missing)"
    
    status_msg = "Voice conversion completed successfully!"
    if not res["web_link"]:
        status_msg += "\nNote: Google Drive upload skipped. Place your 'service_account.json' or 'client_secrets.json' in the workspace folder to enable automated uploads."
        
    return output_audio, status_msg, web_link, direct_link


def gradio_batch_convert(input_dir, model_url, pitch, index_rate, protect, rms_mix_rate, noise_gate_db, normalize_db, folder_id, spectral_denoise, spectral_denoise_threshold):
    if not input_dir or not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return f"Error: Folder '{input_dir}' does not exist or is not a directory.", []
        
    # Run the automated batch pipeline
    results = process_audio_pipeline(
        input_path=input_dir,
        model_url=model_url.strip() if model_url.strip() else None,
        pitch_shift=int(pitch),
        index_rate=float(index_rate),
        protect_index=float(protect),
        rms_mix_rate=float(rms_mix_rate),
        noise_gate_db=float(noise_gate_db),
        normalize_db=float(normalize_db),
        folder_id=folder_id.strip() if folder_id.strip() else None,
        spectral_denoise=bool(spectral_denoise),
        spectral_denoise_threshold=float(spectral_denoise_threshold)
    )
    
    if not results:
        return "No files found or processed in the folder.", []
        
    table_data = []
    log_msg = f"Batch processing completed. Processed {len(results)} files.\n\n"
    for src, res in results.items():
        name = os.path.basename(src)
        if "error" in res:
            table_data.append([name, "FAILED", res["error"]])
            log_msg += f"❌ [{name}]: Failed - {res['error']}\n"
        else:
            drive_status = res["web_link"] if res["web_link"] else "Local only (no credentials)"
            table_data.append([name, "SUCCESS", drive_status])
            log_msg += f"✅ [{name}]: Success -> {res['output_file']}\n"
            
    return log_msg, table_data


# Build Gradio UI
with gr.Blocks(title="RVC Voice Conversion Automator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ RVC Male-to-Female Voice Conversion Automator
    This local pipeline uses Retrieval-based Voice Conversion (RVC) and the Google Drive API to convert male audio to a natural female voice, and upload it automatically to Google Drive.
    """)
    
    # System status component
    status_html = gr.HTML(check_system_status())
    
    with gr.Tabs():
        # TAB 1: Single Audio File Conversion
        with gr.TabItem("Single File Conversion"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Upload Male Audio")
                    audio_input = gr.Audio(type="filepath", label="Source Audio (Male)")
                    
                    gr.Markdown("### 2. Configure RVC parameters")
                    model_url_input = gr.Textbox(
                        value=DEFAULT_MODEL_URL,
                        label="Hugging Face Model Link (.zip)",
                        placeholder="Leave blank for default female voice model"
                    )
                    
                    with gr.Row():
                        pitch_shift = gr.Slider(minimum=-24, maximum=24, value=13, step=1, label="Pitch Shift (Octave Up = +12)")
                        index_rate = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Index Rate (Timbre Influence)")
                        
                    with gr.Row():
                        protect_rate = gr.Slider(
                            minimum=0.0, maximum=0.5, value=0.33, step=0.05, 
                            label="Consonant/Breath Protection (Keep breaths natural)"
                        )
                        rms_mix_rate = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.25, step=0.05, 
                            label="RMS Mix Rate (Volume Envelope Influence)"
                        )
                    
                    with gr.Row():
                        spectral_denoise = gr.Checkbox(
                            value=True,
                            label="Spectral Noise Reduction (Recommended for ASMR)"
                        )
                        spectral_denoise_threshold = gr.Slider(
                            minimum=0.0, maximum=5.0, value=2.5, step=0.1,
                            label="Spectral Noise Threshold (Default 2.5)"
                        )

                    with gr.Row():
                        noise_gate_db = gr.Slider(
                            minimum=-100.0, maximum=-30.0, value=-50.0, step=1.0, 
                            label="Noise Gate Threshold (dB, -100 to disable if using Spectral)"
                        )
                        normalize_db = gr.Slider(
                            minimum=-12.0, maximum=0.0, value=-3.0, step=0.5, 
                            label="Output Peak Normalization (dB, 0.0 to skip)"
                        )
                    
                    gdrive_folder = gr.Textbox(
                        value="",
                        label="Google Drive Folder ID (Optional)",
                        placeholder="Leave blank to upload to root folder"
                    )
                    
                    btn_convert = gr.Button("Convert & Upload", variant="primary")
                    
                with gr.Column():
                    gr.Markdown("### 3. Output Waveform (Female)")
                    audio_output = gr.Audio(type="filepath", label="Converted Audio (Female)")
                    
                    gr.Markdown("### 4. Status & Shareable Links")
                    status_output = gr.Textbox(label="Status Log", interactive=False)
                    
                    web_link_output = gr.Textbox(label="Google Drive Web View Link", interactive=True)
                    direct_link_output = gr.Textbox(label="Google Drive Direct Download Link", interactive=True)
                    
            btn_convert.click(
                fn=gradio_single_convert,
                inputs=[audio_input, model_url_input, pitch_shift, index_rate, protect_rate, rms_mix_rate, noise_gate_db, normalize_db, gdrive_folder, spectral_denoise, spectral_denoise_threshold],
                outputs=[audio_output, status_output, web_link_output, direct_link_output]
            )
            
        # TAB 2: Batch Directory Conversion
        with gr.TabItem("Batch Conversion"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Batch Processing Configuration")
                    input_folder = gr.Textbox(
                        value="inputs",
                        label="Inputs Directory Path",
                        placeholder="Path to folder containing audio files"
                    )
                    
                    batch_model_url = gr.Textbox(
                        value=DEFAULT_MODEL_URL,
                        label="Hugging Face Model Link (.zip)"
                    )
                    
                    with gr.Row():
                        batch_pitch = gr.Slider(minimum=-24, maximum=24, value=13, step=1, label="Pitch Shift")
                        batch_index = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Index Rate")
                        
                    with gr.Row():
                        batch_protect = gr.Slider(minimum=0.0, maximum=0.5, value=0.33, step=0.05, label="Consonant/Breath Protection")
                        batch_rms = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="RMS Mix Rate")
                    
                    with gr.Row():
                        batch_spectral_denoise = gr.Checkbox(
                            value=True,
                            label="Spectral Noise Reduction (Recommended for ASMR)"
                        )
                        batch_spectral_threshold = gr.Slider(
                            minimum=0.0, maximum=5.0, value=2.5, step=0.1,
                            label="Spectral Noise Threshold (Default 2.5)"
                        )

                    with gr.Row():
                        batch_noise_gate = gr.Slider(minimum=-100.0, maximum=-30.0, value=-50.0, step=1.0, label="Noise Gate Threshold (dB, -100 to disable)")
                        batch_normalize = gr.Slider(minimum=-12.0, maximum=0.0, value=-3.0, step=0.5, label="Output Normalization (dB)")
                        
                    batch_folder_id = gr.Textbox(value="", label="Google Drive Folder ID (Optional)")
                    
                    btn_batch = gr.Button("Start Batch Pipeline", variant="primary")
                    
                with gr.Column():
                    gr.Markdown("### Batch Conversion Log")
                    batch_status = gr.Textbox(label="Execution Log", interactive=False, lines=6)
                    
                    gr.Markdown("### Batch Results Table")
                    results_table = gr.Dataframe(
                        headers=["Filename", "Conversion Status", "Google Drive Link"],
                        datatype=["str", "str", "str"],
                        label="Processed Files"
                    )
                    
            btn_batch.click(
                fn=gradio_batch_convert,
                inputs=[input_folder, batch_model_url, batch_pitch, batch_index, batch_protect, batch_rms, batch_noise_gate, batch_normalize, batch_folder_id, batch_spectral_denoise, batch_spectral_threshold],
                outputs=[batch_status, results_table]
            )
            
    # Refresh system requirements status when page loads
    demo.load(fn=check_system_status, outputs=status_html)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
