# 🍳 AI Recipe Extractor from Video

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

An AI-powered web application that automatically extracts recipes from cooking videos. It understands multiple languages, displays a video thumbnail for confirmation, and generates a clean, formatted recipe in the **user's chosen language**.

*(Suggestion: Run the app locally, take a screenshot of the final UI with the thumbnail preview, and add it here.)*

---

## Features

-   **Video Thumbnail Preview**: Instantly displays the video's thumbnail and title upon entering a URL for confirmation.
-   **Selectable Output Language**: Users can choose to receive the final recipe in **English, Hindi, Telugu, Tamil, Kannada, or Malayalam**.
-   **Broad Input Language Support**: Natively understands and processes videos in all the languages listed above.
-   **Automated Recipe Generation**: Extracts ingredients and instructions from unstructured video dialogue.
-   **YouTube Integration**: Works directly with any public YouTube video URL.
-   **Interactive Web UI**: Simple, user-friendly interface built with Gradio.
-   **Deployable**: Ready to be deployed on platforms like Hugging Face Spaces.

---

## How It Works

1.  **User Input**: The user pastes a YouTube URL. The app immediately fetches and displays the **video thumbnail and title**. The user then **selects their desired output language**.
2.  **Audio Extraction**: `yt-dlp` downloads the audio stream from the video.
3.  **Multi-Language Speech-to-Text**: The `whisper` model auto-detects the language in the audio and transcribes it.
4.  **Cross-Lingual Recipe Generation**: The `flan-t5-base` model processes the transcript and generates a structured recipe in the **language selected by the user**.
5.  **Display**: The final recipe is presented to the user in the web interface.

---

## Tech Stack

-   **Core Framework**: Gradio
-   **Video/Audio Processing**: `yt-dlp`, `moviepy`
-   **Speech-to-Text**: `openai-whisper` via `SpeechRecognition`
-   **Language Model**: `google/flan-t5-base` via `transformers`
-   **Containerization**: Docker

---

## 🚀 Setup and Deployment

### A. Running Locally

**Prerequisites:**
* Python 3.9+
* Git
* `ffmpeg`:
    * **macOS**: `brew install ffmpeg`
    * **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
    * **Windows**: `choco install ffmpeg`

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ai-recipe-extractor.git](https://github.com/YOUR_USERNAME/ai-recipe-extractor.git)
    cd ai-recipe-extractor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    The app will be available at `http://127.0.0.1:7860`.

### B. Deploying to Hugging Face Spaces

This project is ready to be deployed for free on Hugging Face Spaces.

1.  **Create a Hugging Face Account**: If you don't have one, sign up at [huggingface.co](https://huggingface.co).
2.  **Create a New Space**:
    * Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    * Choose **Gradio** as the Space SDK and select **Public** visibility.
3.  **Upload Files**:
    * In your new Space repository, click on **Files** -> **Add file** -> **Upload files**.
    * Upload your `app.py`, `requirements.txt`, and `.gitignore` files.
    * The Space will automatically build and launch your application.

---

## Project Structure

├── .gitignore       # Specifies intentionally untracked files to ignore
├── app.py           # The core Gradio application script
├── Dockerfile       # Instructions to build the Docker container (optional)
├── README.md        # Project documentation
└── requirements.txt # Project dependencies