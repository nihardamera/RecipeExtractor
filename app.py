# app.py
import gradio as gr
import yt_dlp
from pydub import AudioSegment
from transformers import pipeline
import os
import traceback
import torch

# --- Setup and Configuration ---

TRANSCRIBER_ID = "distil-whisper/distil-large-v2"

MAX_DURATION_SECONDS = 600

# --- Core Logic Functions with Improvements ---

def get_video_info(video_url: str):
    """
    Extracts thumbnail URL and title from a YouTube video for preview.
    """
    if not video_url:
        return None, None, gr.update(visible=False)
    ydl_opts = {'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            thumbnail_url = info_dict.get('thumbnail', None)
            video_title = info_dict.get('title', None)
            return thumbnail_url, video_title, gr.update(visible=True)
    except Exception as e:
        print(f"Could not fetch video info: {e}")
        return None, "Could not fetch video title.", gr.update(visible=False)

def download_and_trim_audio(video_url: str) -> str:
    """
    Downloads audio and trims it using Pydub.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio_full.mp3',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        filename = "audio_full.mp3.mp3"

        # Load the audio with Pydub
        audio = AudioSegment.from_mp3(filename)

        trimmed_audio = audio[:MAX_DURATION_SECONDS * 1000]

        trimmed_audio.export("audio_trimmed.mp3", format="mp3")

        os.remove(filename)
        return "audio_trimmed.mp3"

    except Exception as e:
        print(f"Error downloading or trimming audio: {e}")
        raise gr.Error(f"Failed to process audio. Error: {e}")


def transcribe_audio_optimized(audio_path: str) -> str:
    """
    Transcribes audio using the faster, distilled Whisper model via the transformers pipeline.
    """
    if not os.path.exists(audio_path):
        raise gr.Error("Audio file not found for transcription.")
    
    try:
        # Load the pipeline with the optimized model
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=TRANSCRIBER_ID,
            torch_dtype=torch.float16,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        
        result = transcriber(audio_path, return_timestamps=True)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise gr.Error(f"Failed to transcribe audio. The model may have failed to load. Error: {e}")

def generate_recipe_advanced(text: str, output_language: str) -> str:
    """
    Generates a recipe using an advanced, more reliable prompt.
    """
    try:
        summarizer = pipeline("summarization", model="google/flan-t5-base")
        
        prompt = (
            f"You are a helpful cooking assistant. Your task is to analyze the following transcript from a cooking video and generate a clear, accurate recipe in {output_language.upper()}. "
            "Follow these instructions carefully:\n"
            "1.  **Ingredients Section**: List all ingredients with precise measurements mentioned in the text. If a measurement is ambiguous (e.g., 'a pinch', 'a bit'), state it exactly as mentioned.\n"
            "2.  **Instructions Section**: Provide a step-by-step numbered list of instructions. \n"
            "3.  **Chef's Tips Section**: If the chef provides any specific tips, techniques, or reasons for a step, list them in a separate 'Chef's Tips' section.\n"
            "4.  **Strictness**: Only include ingredients and steps that are explicitly mentioned in the transcript. DO NOT add common ingredients like 'salt and pepper' unless they are mentioned.\n\n"
            f"Transcript:\n---\n{text}\n---"
        )
        
        recipe_text = summarizer(prompt, max_length=1024, min_length=200, do_sample=False)
        return recipe_text[0]['summary_text']
    except Exception as e:
        print(f"Error generating recipe: {e}")
        raise gr.Error(f"The AI model failed to generate a recipe from the text. Error: {e}")

def get_recipe_from_video(video_url: str, output_language: str, progress=gr.Progress(track_tqdm=True)):
    """
    Main pipeline function that orchestrates the improved recipe extraction process.
    """
    audio_file = ""
    try:
        progress(0, desc="Step 1/3: Downloading & Trimming Audio...")
        audio_file = download_and_trim_audio(video_url)

        progress(0.5, desc="Step 2/3: Transcribing Audio (using optimized model)...")
        transcribed_text = transcribe_audio_optimized(audio_file)

        progress(0.8, desc=f"Step 3/3: Generating Recipe in {output_language}...")
        recipe = generate_recipe_advanced(transcribed_text, output_language)
        
        return recipe

    except gr.Error as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        raise gr.Error(f"An unexpected error occurred: {e}")
        
    finally:
        if os.path.exists("audio_trimmed.mp3"):
            os.remove("audio_trimmed.mp3")
        if os.path.exists("audio_full.mp3"):
            os.remove("audio_full.mp3")
        print("Cleaned up temporary files.")

# --- Gradio Interface Definition ---

css = """
    .gradio-container {font-family: 'IBM Plex Sans', sans-serif;}
    footer {display: none !important;}
    .disclaimer {font-size: 0.9em; color: #666;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        # 🍳 AI Recipe Extractor from Video (v2.0)
        Turn any YouTube cooking video into a well-formatted recipe. Now faster and more reliable!
        """
    )
    
    with gr.Row():
        video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=...", scale=4)
        submit_button = gr.Button("Get Recipe", variant="primary", scale=1)

    with gr.Row():
        video_thumbnail_output = gr.Image(label="Video Thumbnail", width=240, interactive=False)
        video_title_output = gr.Textbox(label="Video Title", interactive=False)

    output_language_selector = gr.Radio(["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam"], label="Select Output Language", value="English")

    recipe_output = gr.Textbox(label="Generated Recipe", lines=15, interactive=False, show_copy_button=True)
    
    with gr.Accordion("Important Notes & Limitations", open=False):
        gr.Markdown(
            """
            - **Processing Time:** This app uses powerful AI models. Generating a recipe can take 1-3 minutes.
            - **Audio Quality is Key:** Results are best with clear narration and minimal background music.
            - **10-Minute Limit:** To ensure stability, only the first 10 minutes of the video are processed.
            - **AI Disclaimer:** The generated recipe is created by an AI and may contain inaccuracies. Always use your best judgment when cooking.
            """,
            elem_classes="disclaimer"
        )

    video_url_input.submit(fn=get_video_info, inputs=video_url_input, outputs=[video_thumbnail_output, video_title_output])
    submit_button.click(fn=get_recipe_from_video, inputs=[video_url_input, output_language_selector], outputs=recipe_output)
    
    gr.Examples(
        examples=[
            ["https://www.youtube.com/watch?v=uJp_G_bL_sA", "English"],
            ["https://www.youtube.com/watch?v=E1h__8s6_k4", "Hindi"],
            ["https://www.youtube.com/watch?v=k-uOf-2y4E4", "Telugu"],
        ],
        inputs=[video_url_input, output_language_selector],
        outputs=recipe_output,
        fn=get_recipe_from_video,
    )

if __name__ == "__main__":
    demo.launch()