# 🎙️ Podcast Processing Pipeline

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

A reproducible audio-to-dataset pipeline that transforms long-form podcast audio into `(context → response)` training pairs for a target speaker. This document explains how to use the pipeline and provides a detailed look at its internal logic.

## ✨ Features

* **Automatic Transcription**: High-accuracy speech-to-text using **[WhisperX](https://github.com/m-bain/whisperx)**.
* **Word-Level Alignment**: Precise word and phoneme-level timestamps.
* **Speaker Diarization**: Determines "who spoke when" using **[pyannote.audio](https://pyannote.github.io/pyannote-audio/)**.
* **Target Speaker Identification**: Automatically finds the target speaker's voice using audio samples and **[Resemblyzer](https://github.com/resemble-ai/Resemblyzer)**.
* **Intelligent Pair Building**: Constructs clean `(context → response)` data by filtering overlaps, handling interruptions, and merging replies.
* **Workflow Caching**: Saves intermediate steps (transcription, alignment, etc.) for rapid iteration and debugging.
* **FFmpeg Auto-Setup**: Downloads a local copy of FFmpeg on Windows if it's not found in the system PATH.

---

## 🚀 Quickstart

### 1. Install Prerequisites

*   **Python**: Install **Python 3.11** (required).
*   **FFmpeg**:
    *   **Windows**: No action needed. The script will download it automatically if it's not found.
    *   **macOS**: Install via [Homebrew](https://brew.sh/): `brew install ffmpeg`
    *   **Linux (Debian/Ubuntu)**: `sudo apt-get update && sudo apt-get install ffmpeg`

### 2. Clone the Repo & Set Up

```bash
git clone <repository_url>
cd <repository_folder>
```

### 3. Get a Hugging Face Token

This pipeline requires a Hugging Face token to download pre-trained models for speaker diarization.
1.  Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to get one (you'll need a free account).
2.  Create a new **User Access Token**, giving it a name (e.g., "podcast_pipeline") and the `read` role.
3.  The first time you run the script, it will prompt you to enter this token.

### 4. Install Dependencies (PyTorch + Other Libraries)

It is highly recommended to install PyTorch *before* installing the other packages, as it has platform-specific builds.

1.  **Install PyTorch**: Follow the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your specific OS and GPU setup (CUDA/MPS/CPU).
    *   **Example (CPU-only)**:
        ```bash
        pip install torch torchvision torchaudio
        ```
    *   **Example (NVIDIA GPU with CUDA 11.8)**:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

2.  **Install Other Libraries**:
    *   **Windows**: Double-click `run_pipeline.bat`. It will create the virtual environment and install dependencies automatically.
    *   **macOS/Linux**: Run the setup script:
        ```bash
        bash run_pipeline.sh
        ```
    This will create a virtual environment in `sys/venv/` and install the remaining packages from `requirements.txt`.

### 5. Place Your Files

*   Place all your source audio files into the `audio_files/` folder.
*   Place your target speaker's voice samples into the `samples/` folder.

### 6. Run the Pipeline

*   **Windows**: Double-click `run_pipeline.bat`.
*   **macOS/Linux**:
    ```bash
    bash run_pipeline.sh
    ```

**Input**: `audio_files/` folder & `samples/` folder
**Output**: `results/{base_name}_results.jsonl` for each input file.


### Speaker Sample Guidelines

For the best speaker identification accuracy, follow these guidelines for the audio clips you place in the `samples/` folder:

* **Ideal Number**: Use **3 to 5 separate sample files**. This creates a more robust and stable average "voiceprint" for the target speaker.
* **Ideal Length**: Each sample clip should be between **10 and 30 seconds long**. This provides enough vocal data without being inefficient.
* **Quality is Key**: The most important factor is quality. Ensure the samples are **clean recordings of only the target speaker**, with no background music, noise, or other people talking.


## 🔑 Environment

* Python **3.11+**
* Torch (CUDA build for GPU acceleration)
* **Required**: A Hugging Face token (`HF_TOKEN`) to download the speaker diarization models. See the [Quickstart](#-quickstart) for instructions.
---

## ⚙️ Configuration Knobs

All primary settings are located at the top of `podcast_pipeline_gpu.py` for easy tuning.

* `KEEP_DEBUG_OUTPUTS` # If `True`, keeps detailed diagnostic files (e.g., `_pairs_flagged.jsonl`, `_pair_quality_metrics.csv`) for tuning and analysis. If `False`, all intermediate files will be deleted.

### Pair Building & Filtering
* `MERGE_GAP` # Defines the max silence (in seconds) between replies from the target speaker before they are merged into a single response.
* `MIN_WORDS` # Minimum word count for a reply. The pipeline is tuned to keep high-confidence single-word replies (e.g., "Exactly!"), so the default is `1`.
* `MIN_CONF` # Sets a minimum average word confidence for a reply. Can be set to `None` to disable.
* `OVERLAP_FRAC` # (fraction) If more than this % of a word overlaps with another speaker, drop it.
* `MAX_INPUT_WORDS` # Sets a hard limit on the number of words included in the input context to prevent excessively long prompts.
* `ADAPTIVE_STRATEGY` # Strategy for the adaptive confidence filter ("percentile" or "median").

### Speaker Identification
* `IDENTIFY_THRESHOLD` # How similar a voice sample must be (0.0-1.0) to a speaker in the podcast to be identified as the target. Higher is stricter.
* `MIN_EMB_DURATION` # Ignores diarized segments shorter than this (in seconds) when creating speaker voice profiles.

### Transcriptions **[WhisperX Docs](https://github.com/openai/whisper)**.
* `WHISPER_MODEL` # Model size (`"large-v2"`, `"medium"`, etc.). Larger models are more accurate but require more VRAM.
* `WHISPER_BATCH_SIZE` # Number of parallel segments to transcribe. Increase if you have a high-VRAM GPU.
* `USE_VAD` # If `True`, runs Voice Activity Detection to segment the audio before transcription, improving accuracy on audio with long silences.

### Quality Control & Auto-Flagging
* `QUALITY_PROFILE` # Active quality preset (`"balanced"`, `"conservative"`, `"relaxed"`). Controls VAD sensitivity and adaptive confidence filters.
* `OVERLAP_THRESHOLD` # Flags a pair for review if the percentage of overlapped words in the response exceeds this value.
* `LOW_CONFIDENCE_THRESHOLD` # Flags a pair for review if its average word confidence is below this value.
* `INPUT_TO_OUTPUT_RATIO` # Flags a pair if the input is excessively long compared to a short output (e.g., a 200-word question for a 3-word answer).

---

## 🏗️ Project Structure

This project uses a clean folder structure to separate inputs, outputs, and system files.

* **`audio_files/`**: Place your source audio files (e.g., `.mp3`, `.wav`) here.
* **`samples/`**: Place short, clean audio clips of your target speaker's voice here.
* **`results/`**: Contains the final, cleaned `.jsonl` training data after a successful run.
* **`cache/`**: Stores intermediate files to speed up subsequent runs (e.g., transcription, alignment). It also contains valuable diagnostic files for tuning (`_pairs_flagged.jsonl`, `_pair_quality_metrics.csv`). You can safely delete this folder to force a completely fresh run.
* **`sys/`**: Contains the core system files: the Python script, virtual environment (`venv`), and FFMPEG (on Windows). You should not need to modify this folder.
* **`run_pipeline.bat`**: The main script to execute the entire pipeline on **Windows**.

---

## 🧠 Pipeline Logic

### Critical Data Filtering Flow

The core of the pair-building logic follows this multi-stage filtering process. It's tuned to preserve the target speaker's unique conversational style for the parody chatbot goal.

1.  **Adaptive Confidence Filter**: Removes low-confidence words based on per-speaker statistical thresholds, which is more robust than a single global cutoff.

2.  **Neural VAD Filter**: Identifies all segments of active speech. Any transcribed words falling outside these segments (often hallucinations in silent parts) are discarded.

3.  **Short Word Reassignment**: Corrects common diarization errors where short interjections (e.g., "yes," "ok") from the target speaker are misattributed to another speaker.

4.  **Pair Construction & Reply Merging**: The pipeline constructs `(context -> response)` pairs and then stitches together consecutive replies from the target speaker to form longer, more coherent outputs.

5.  **Overlap Tagging (Instead of Dropping)**: To preserve the natural dynamics of conversation and interruption, words spoken during cross-talk are **not deleted**. Instead, they are wrapped in special tags (e.g., `"I <overlap>really</overlap> think so"`). This teaches the LLM the context of interruptions, a key part of a speaker's personality.

6.  **Final Filtering (with Nuance)**: The cleaned pairs are passed through final checks. This step is designed to **preserve punchy, high-confidence short replies** (like "Exactly!") which are often discarded by simpler filters but are crucial for capturing a speaker's character.

### Function Call Hierarchy

This shows the main logical flow of the script from entry point to the core functions.

* `main()`
    * `run_pipeline()`
        * `transcribe_with_vad()`
        * `safe_assign_word_speakers()`
            * `attach_speakers_to_transcript()`
        * `compute_speaker_embeddings()`
        * `identify_speaker()`
            * `load_reference_embeddings()`
        * **Pair Building & Filtering (Step 5)**
            * `adaptive_conf_filter()`
            * `run_vad_with_fallback()`
            * `reassign_short_target_words()`
            * `build_pairs_detailed()`
            * `merge_close_replies()`
            * `compute_overlap_and_write()`
                * `score_pair_quality()`
            * `auto_flag_low_quality_pairs()`
            * `postfilter_clean_pairs()`
