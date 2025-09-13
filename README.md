# 🎙️ Podcast Processing Pipeline

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

A reproducible audio-to-dataset pipeline that transforms long-form podcast audio into `(context → response)` training pairs for a target speaker. This document explains how to use the pipeline and provides a detailed look at its internal logic.

## ✨ Features

* **Automatic Transcription**: High-accuracy speech-to-text using **WhisperX**.
* **Word-Level Alignment**: Precise word and phoneme-level timestamps.
* **Speaker Diarization**: Determines "who spoke when" using **pyannote.audio**.
* **Target Speaker Identification**: Automatically finds the target speaker's voice using audio samples and **Resemblyzer**.
* **Intelligent Pair Building**: Constructs clean `(context → response)` data by filtering overlaps, handling interruptions, and merging replies.
* **Workflow Caching**: Saves intermediate steps (transcription, alignment, etc.) for rapid iteration and debugging.
* **FFmpeg Auto-Setup**: Downloads a local copy of FFmpeg on Windows if it's not found in the system PATH.

---

## 🚀 Quickstart

1.  **Clone the Repo**:
    ```bash
    git clone <your-repo-url>
    cd podcast-pipeline
    ```
2.  **Place Your Files**:
    * Place all your source audio files into the `audio_files/` folder.
    * Place your target speaker's voice samples into the `samples/` folder.
3.  **Run the Pipeline**:
    * On Windows, simply double-click `run_pipeline.bat`. It will create the environment, install dependencies, and start the process.

**Input**: `audio_files/` folder & `samples/` folder
**Output**: `results/{base_name}_results.jsonl` for each input file.

### Speaker Sample Guidelines

For the best speaker identification accuracy, follow these guidelines for the audio clips you place in the `samples/` folder:

* **Ideal Number**: Use **3 to 5 separate sample files**. This creates a more robust and stable average "voiceprint" for the target speaker.
* **Ideal Length**: Each sample clip should be between **10 and 30 seconds long**. This provides enough vocal data without being inefficient.
* **Quality is Key**: The most important factor is quality. Ensure the samples are **clean recordings of only the target speaker**, with no background music, noise, or other people talking.

---

## ⚙️ Configuration Knobs

All primary settings are located at the top of `podcast_pipeline_gpu.py` for easy tuning.

* `KEEP_DEBUG_OUTPUTS` # `True` keeps all intermediate files; `False` deletes them, saving only `results.jsonl`.

### Pair Building & Filtering
* `MERGE_GAP` # Defines the max silence (in seconds) between replies from the target speaker before they are merged into a single response.
* `MIN_WORDS` # Sets the minimum word count for a target's reply to be included in the dataset.
* `MIN_CONF` # Sets a minimum average word confidence for a reply. Can be set to `None` to disable.
* `OVERLAP_FRAC` # Defines the maximum fraction (e.g., 0.7 for 70%) that a target's word can be overlapped by another speaker before being discarded.
* `MAX_INPUT_WORDS` # Sets a hard limit on the number of words included in the input context to prevent excessively long prompts.
* `ADAPTIVE_STRATEGY` # Strategy for the adaptive confidence filter ("percentile" or "median").

### Speaker Identification
* `IDENTIFY_THRESHOLD` # How similar a voice sample must be (0.0-1.0) to a speaker in the podcast to be identified as the target. Higher is stricter.
* `MIN_EMB_DURATION` # Ignores diarized segments shorter than this (in seconds) when creating speaker voice profiles.

### Transcription (WhisperX)
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
* **`cache/`**: Contains all intermediate files: `.wav` conversions, `.json` caches, debug logs, and quality reports. This folder can be safely deleted to force a clean run.
* **`sys/`**: Contains the core system files: the Python script, virtual environment (`venv`), and FFMPEG. You should not need to modify this folder.
* **`run_pipeline.bat`**: The main script to execute the entire pipeline.

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
            