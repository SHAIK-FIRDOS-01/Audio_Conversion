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
    # Connected status badges
    rvc_status = (
        "<span style='background-color: rgba(34, 197, 94, 0.15); color: #4ade80; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74, 222, 128, 0.25); display: inline-flex; align-items: center;'>✓ Installed</span>" 
        if RVC_AVAILABLE else 
        "<span style='background-color: rgba(239, 68, 68, 0.15); color: #f87171; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(248, 113, 113, 0.25); display: inline-flex; align-items: center;'>✗ Missing (Run setup.bat)</span>"
    )
    
    gdrive_status = (
        "<span style='background-color: rgba(34, 197, 94, 0.15); color: #4ade80; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74, 222, 128, 0.25); display: inline-flex; align-items: center;'>✓ Connected</span>" 
        if GDRIVE_LIBS_AVAILABLE else 
        "<span style='background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(251, 191, 36, 0.25); display: inline-flex; align-items: center;'>⚠ Missing API Libs (Local Mode Only)</span>"
    )
    
    # Check for credentials
    creds_status = "<span style='background-color: rgba(239, 68, 68, 0.15); color: #f87171; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(248, 113, 113, 0.25); display: inline-flex; align-items: center;'>✗ Uploads Disabled</span>"
    if os.path.exists('service_account.json'):
        creds_status = "<span style='background-color: rgba(34, 197, 94, 0.15); color: #4ade80; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74, 222, 128, 0.25); display: inline-flex; align-items: center;'>✓ Service Account Key Found</span>"
    elif os.path.exists('token.json'):
        creds_status = "<span style='background-color: rgba(34, 197, 94, 0.15); color: #4ade80; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(74, 222, 128, 0.25); display: inline-flex; align-items: center;'>✓ OAuth Token Found</span>"
    elif os.path.exists('client_secrets.json'):
        creds_status = "<span style='background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid rgba(251, 191, 36, 0.25); display: inline-flex; align-items: center;'>✓ Client Secrets Found (Will authenticate on run)</span>"
        
    return f"""
    <div style='background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); color: #f1f5f9;'>
        <h3 style='margin-top: 0; color: #818cf8; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 8px; font-weight: 600; font-size: 1.15rem;'>⚡ System Integration Status</h3>
        <ul style='list-style-type: none; padding-left: 0; margin-bottom: 0;'>
            <li style='margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;'>
                <span>🤖 <b>RVC Voice Engine</b>:</span>
                <span>{rvc_status}</span>
            </li>
            <li style='margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;'>
                <span>📁 <b>Google API Client</b>:</span>
                <span>{gdrive_status}</span>
            </li>
            <li style='margin-bottom: 0; display: flex; align-items: center; justify-content: space-between;'>
                <span>🔑 <b>OAuth/Drive Credentials</b>:</span>
                <span>{creds_status}</span>
            </li>
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
        status_msg += "\nNote: Google Drive upload skipped. Place your 'client_secrets.json' in the workspace folder to enable automated uploads."
        
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


# Custom non-intrusive CSS. Does not style global inputs/buttons to keep native components fully functional
custom_css = """
body, .gradio-container {
    background-color: #090d16 !important;
    background-image: radial-gradient(circle at top, #14113c 0%, #030712 100%) !important;
}

/* Custom premium hover and glow animations for primary button */
button.primary-btn {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    filter: brightness(1.1) !important;
}

button.primary-btn:active {
    transform: translateY(1px) !important;
}

/* Make sure Audio boxes have a solid, distinct outline and clean styling */
.custom-audio {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background-color: rgba(30, 41, 59, 0.3) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Make slider tracks consistent */
.gradio-slider input[type="range"] {
    accent-color: #818cf8 !important;
}
"""

def toggle_spectral(val):
    return gr.update(visible=val)

# Configure beautiful native theme variables (retains fully visible controls for complex components)
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate"
).set(
    # Applies high-contrast colors dynamically
    body_background_fill="*neutral_950",
    body_text_color="*neutral_100",
    
    background_fill_primary="*neutral_950",
    background_fill_secondary="*neutral_900",
    
    # Block containers (Cards)
    block_background_fill="rgba(15, 23, 42, 0.5)",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_width="1px",
    block_label_text_color="*neutral_300",
    block_shadow="0 8px 24px rgba(0, 0, 0, 0.25)",
    
    # Input Elements (scoped properly to standard textboxes natively)
    input_background_fill="rgba(15, 23, 42, 0.6)",
    input_border_color="rgba(255, 255, 255, 0.12)",
    
    # Button styling
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5 0%, #9333ea 100%)",
    button_primary_text_color="#ffffff",
)

# Javascript to force client browser to render in dark mode for high-contrast audio UI
force_dark_mode_js = """
function() {
    document.querySelector('body').classList.add('dark');
}
"""

# Configure Gradio blocks with custom theme, layout, and CSS
with gr.Blocks(title="AI Voice Conversion & Uploader", theme=custom_theme, css=custom_css) as demo:
    
    # Beautiful Header Card
    gr.HTML("""
    <div class="app-header" style="text-align: center; margin-bottom: 30px; padding: 30px 20px; background: rgba(30, 41, 59, 0.25); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.06); backdrop-filter: blur(10px); box-shadow: 0 4px 30px rgba(0,0,0,0.2);">
        <h1 style="font-size: 2.8rem; font-weight: 900; background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 12px 0; letter-spacing: -0.8px;">🎙️ AI Voice Conversion & Uploader</h1>
        <p style="color: #94a3b8; font-size: 1.15rem; max-width: 650px; margin: 0 auto; line-height: 1.6; font-weight: 400;">
            Convert male voice recordings to high-fidelity, expressive female vocals locally, and auto-upload directly to Google Drive with shareable links.
        </p>
    </div>
    """)
    
    # System status component
    status_html = gr.HTML(check_system_status())
    
    with gr.Tabs():
        # TAB 1: Single Audio File Conversion
        with gr.TabItem("⚡ Single File Conversion"):
            with gr.Row():
                # Input Controls Column
                with gr.Column(scale=11):
                    with gr.Group():
                        gr.Markdown("### 📂 1. Source Audio Input")
                        audio_input = gr.Audio(type="filepath", label="Source Audio (Male)", elem_classes=["custom-audio"])
                    
                    with gr.Group():
                        gr.Markdown("### ⚙️ 2. Core Configurations")
                        model_url_input = gr.Textbox(
                            value=DEFAULT_MODEL_URL,
                            label="Hugging Face RVC Model (.zip)",
                            placeholder="Leave blank to use default Voice Model"
                        )
                        gdrive_folder = gr.Textbox(
                            value="",
                            label="Google Drive Folder ID (Optional)",
                            placeholder="Leave blank to upload to root folder"
                        )
                    
                    with gr.Group():
                        gr.Markdown("### 🎛️ 3. RVC Voice Synthesis Tuning")
                        with gr.Row():
                            pitch_shift = gr.Slider(minimum=-24, maximum=24, value=13, step=1, label="Pitch Shift (Default: +13 for Male-to-Female)")
                            index_rate = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Index Rate (Timbre Strength)")
                            
                        with gr.Row():
                            protect_rate = gr.Slider(
                                minimum=0.0, maximum=0.5, value=0.33, step=0.05, 
                                label="Consonant/Breath Protection (Keep natural whispers)"
                            )
                            rms_mix_rate = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.25, step=0.05, 
                                label="RMS Mix Rate (Volume Profile)"
                            )
                    
                    with gr.Group():
                        gr.Markdown("### 🎚️ 4. Audio Quality & Preprocessing")
                        with gr.Row():
                            spectral_denoise = gr.Checkbox(
                                value=True,
                                label="Enable Spectral Noise Subtraction"
                            )
                            spectral_denoise_threshold = gr.Slider(
                                minimum=0.0, maximum=5.0, value=2.5, step=0.1,
                                label="Denoise Intensity (Threshold multiplier)"
                            )
                        
                        with gr.Row():
                            noise_gate_db = gr.Slider(
                                minimum=-100.0, maximum=-30.0, value=-50.0, step=1.0, 
                                label="Noise Gate Threshold (dB, -100 to disable)"
                            )
                            normalize_db = gr.Slider(
                                minimum=-12.0, maximum=0.0, value=-3.0, step=0.5, 
                                label="Output Peak Normalization (dB)"
                            )
                    
                    btn_convert = gr.Button("Convert & Upload 🚀", variant="primary", elem_classes=["primary-btn"])
                    
                # Output column
                with gr.Column(scale=9):
                    with gr.Group():
                        gr.Markdown("### 🎧 5. Output Preview (Female)")
                        audio_output = gr.Audio(type="filepath", label="Converted Audio (Female)", elem_classes=["custom-audio"])
                        
                    with gr.Group():
                        gr.Markdown("### 📊 6. Job Log & Shareable Links")
                        status_output = gr.Textbox(label="Status Log", interactive=False)
                        
                        web_link_output = gr.Textbox(label="Google Drive Web View Link", interactive=True)
                        direct_link_output = gr.Textbox(label="Google Drive Direct Download Link", interactive=True)
            
            # Interactive toggle for spectral denoise threshold
            spectral_denoise.change(
                fn=toggle_spectral,
                inputs=spectral_denoise,
                outputs=spectral_denoise_threshold
            )
            
            btn_convert.click(
                fn=gradio_single_convert,
                inputs=[audio_input, model_url_input, pitch_shift, index_rate, protect_rate, rms_mix_rate, noise_gate_db, normalize_db, gdrive_folder, spectral_denoise, spectral_denoise_threshold],
                outputs=[audio_output, status_output, web_link_output, direct_link_output]
            )
            
        # TAB 2: Batch Directory Conversion
        with gr.TabItem("📦 Batch Folder Processing"):
            with gr.Row():
                with gr.Column(scale=11):
                    with gr.Group():
                        gr.Markdown("### 📁 Batch Processing Settings")
                        input_folder = gr.Textbox(
                            value="inputs",
                            label="Inputs Directory Path (Local folder)",
                            placeholder="Path to folder containing audio files"
                        )
                        batch_model_url = gr.Textbox(
                            value=DEFAULT_MODEL_URL,
                            label="Hugging Face Model Link (.zip)"
                        )
                        batch_folder_id = gr.Textbox(
                            value="", 
                            label="Google Drive Folder ID (Optional)"
                        )
                        
                    with gr.Group():
                        gr.Markdown("### 🎛️ RVC Parameters")
                        with gr.Row():
                            batch_pitch = gr.Slider(minimum=-24, maximum=24, value=13, step=1, label="Pitch Shift")
                            batch_index = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Index Rate")
                            
                        with gr.Row():
                            batch_protect = gr.Slider(minimum=0.0, maximum=0.5, value=0.33, step=0.05, label="Consonant/Breath Protection")
                            batch_rms = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="RMS Mix Rate")
                    
                    with gr.Group():
                        gr.Markdown("### 🎚️ Audio Processing")
                        with gr.Row():
                            batch_spectral_denoise = gr.Checkbox(
                                value=True,
                                label="Enable Spectral Noise Subtraction"
                            )
                            batch_spectral_threshold = gr.Slider(
                                minimum=0.0, maximum=5.0, value=2.5, step=0.1,
                                label="Denoise Intensity"
                            )
                        
                        with gr.Row():
                            batch_noise_gate = gr.Slider(minimum=-100.0, maximum=-30.0, value=-50.0, step=1.0, label="Noise Gate Threshold (dB)")
                            batch_normalize = gr.Slider(minimum=-12.0, maximum=0.0, value=-3.0, step=0.5, label="Output Normalization (dB)")
                        
                    btn_batch = gr.Button("Start Batch Pipeline ⚙️", variant="primary", elem_classes=["primary-btn"])
                    
                with gr.Column(scale=9):
                    with gr.Group():
                        gr.Markdown("### 📊 Execution Log")
                        batch_status = gr.Textbox(label="Execution Log", interactive=False, lines=6)
                        
                    with gr.Group():
                        gr.Markdown("### 📋 Processed Files List")
                        results_table = gr.Dataframe(
                            headers=["Filename", "Conversion Status", "Google Drive Link"],
                            datatype=["str", "str", "str"],
                            label="Processed Files"
                        )
            
            # Interactive toggle for batch spectral denoise threshold
            batch_spectral_denoise.change(
                fn=toggle_spectral,
                inputs=batch_spectral_denoise,
                outputs=batch_spectral_threshold
            )
            
            btn_batch.click(
                fn=gradio_batch_convert,
                inputs=[input_folder, batch_model_url, batch_pitch, batch_index, batch_protect, batch_rms, batch_noise_gate, batch_normalize, batch_folder_id, batch_spectral_denoise, batch_spectral_threshold],
                outputs=[batch_status, results_table]
            )
            
    # Force client browser to render in dark mode on page load & refresh system status
    demo.load(fn=check_system_status, outputs=status_html, js=force_dark_mode_js)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
