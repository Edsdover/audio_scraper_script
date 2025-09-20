# 🎙️ Audio Processing Pipeline

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

A reproducible audio-to-dataset pipeline that transforms audio into `(context → response)` training pairs for a target speaker. It uses a sophisticated, multi-factor quality scoring system to ensure the highest quality output.

For a deep dive into the tuning process and the rationale behind the quality model, see **[learnings.md](learnings.md)**.

## ✨ Features

*   **Automatic Transcription**: High-accuracy speech-to-text using **[WhisperX](https://github.com/m-bain/whisperx)**.
*   **Word-Level Alignment**: Precise word and phoneme-level timestamps.
*   **Speaker Diarization**: Determines "who spoke when" using **[pyannote.audio](https://pyannote.github.io/pyannote-audio/)**.
*   **Target Speaker Identification**: Automatically finds the target speaker's voice using audio samples and **[Resemblyzer](https://github.com/resemble-ai/Resemblyzer)**.
*   **Advanced Quality Scoring**: Uses a configurable, 8-factor scoring model (semantic similarity, confidence, speaker overlap, etc.) to filter for the highest quality conversational pairs.
*   **Intelligent Pair Building**: Constructs clean `(context → response)` data by filtering overlaps, handling interruptions, and merging replies.
*   **Workflow Caching**: Saves intermediate steps (transcription, alignment, etc.) for rapid iteration and debugging.
*   **FFmpeg Auto-Setup**: Downloads a local copy of FFmpeg on Windows if it's not found in the system PATH.

---

## 🚀 Quickstart

### 1. Install Prerequisites

*   **Python**: Install **Python 3.11** (required).
*   **FFmpeg**:
    *   **Windows**: No action needed. The script will download it automatically if it's not found.
    *   **macOS**: Install via [Homebrew](https://brew.sh/): `brew install ffmpeg`
    *   **Linux (Debian/Ubuntu)**: `sudo apt-get update && sudo apt-get install ffmpeg`

### 2. Get a Hugging Face Token

This pipeline requires a Hugging Face token to download pre-trained models for speaker diarization.
1.  Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to get one (you'll need a free account).
2.  Create a new **User Access Token**, giving it a name (e.g., "audio_pipeline") and the `read` role.
3.  The first time you run the script, it will prompt you to enter this token.

### 3. Install Dependencies (PyTorch + Other Libraries)

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
    *   **macOS/Linux**: A `run_pipeline.sh` script would need to be created, mirroring the logic in `run_pipeline.bat`.

### 4. Place Your Files

*   Place all your source audio files into the `audio_files/` folder.
*   Place your target speaker's voice samples into the `samples/` folder.

### 5. Run the Pipeline

*   **Windows**: Double-click `run_pipeline.bat`.
*   **macOS/Linux**: Execute the corresponding shell script.

**Input**: `audio_files/` folder & `samples/` folder
**Output**: `results/{base_name}_results.jsonl` for each input file.

### Speaker Sample Guidelines

For the best speaker identification accuracy, follow these guidelines for the audio clips you place in the `samples/` folder:

* **Ideal Number**: Use **3 to 5 separate sample files**. This creates a more robust and stable average "voiceprint" for the target speaker.
* **Ideal Length**: Each sample clip should be between **10 and 30 seconds long**.
* **Quality is Key**: The most important factor is quality. Ensure the samples are **clean recordings of only the target speaker**, with no background music, noise, or other people talking.

---

## ⚙️ Configuration

All primary settings are located at the top of `sys/audio_pipeline_gpu.py` for easy tuning. The recommended baseline values are set by default.

### Primary Filters
*   `MIN_QUALITY_SCORE`: The master threshold. A pair's final combined quality score must exceed this value to be kept. **Default: `0.55`**.
*   `MIN_WORDS`: A hard filter to drop replies with fewer than this many words. **Default: `5`**.
*   `MIN_CONF`: A hard filter to drop replies where the average word transcription confidence is below this value. **Default: `0.5`**.

### Pair Building
*   `MERGE_GAP`: The max silence (in seconds) between replies from the target speaker before they are merged into a single response. **Default: `3.0`**.
*   `MAX_CONTEXT`: The maximum number of preceding utterances to include as context for a pair. **Default: `10`**.

### Quality Scoring Weights
These weights control the "recipe" for the final quality score. `WEIGHT_SIMILARITY` is the most important.

*   `WEIGHT_SIMILARITY`: **(Core Metric)** The weight for semantic relevance between context and reply. **Default: `1.0`**.
*   `WEIGHT_OVERLAP`: The weight for penalizing speaker cross-talk. **Default: `1.0`**.
*   `WEIGHT_RATIO`: The weight for rewarding a balanced length between context and reply. **Default: `0.5`**.
*   `WEIGHT_DIVERSITY`: The weight for rewarding more unique speakers in the context. **Default: `0.5`**.
*   `WEIGHT_AVG_CONF`: The weight for rewarding higher transcription confidence. **Default: `0.2`**.
*   `WEIGHT_PUNC`: The weight for rewarding replies that contain sentence-ending punctuation. **Default: `0.2`**.
*   `WEIGHT_INPUT_LEN` / `WEIGHT_OUTPUT_LEN`: Weights for rewarding longer, more substantial conversations. **Default: `0.1`**.

### Speaker Identification
*   `IDENTIFY_THRESHOLD`: Similarity score needed to identify the target speaker in the audio. **Default: `0.70`**.
*   `CONTEXTUAL_REID_THRESHOLD`: Similarity score for re-assigning short, unattributed words to an adjacent speaker. **Default: `0.72`**.

---

## 🧠 Pipeline Logic

The pipeline processes audio in several stages to arrive at the final, high-quality dataset.

1.  **Transcription & Diarization**: The audio is transcribed to text with word-level timestamps using WhisperX, and speaker turns are identified using pyannote.

2.  **Target Speaker Identification**: The speaker whose voice best matches the audio clips in the `/samples` folder is designated as the target.

3.  **Pair Construction**: The script iterates through the transcript, creating initial `(context → response)` pairs where the response is an utterance from the target speaker.

4.  **Adaptive Pair Strategy**: The pipeline detects if the audio is a multi-speaker conversation or a single-speaker monologue. If the target speaker accounts for over 98% of the words, it switches to a special **Q&A mode** to extract question-answer pairs from the monologue. Otherwise, it builds conversational context from multiple speakers.

5.  **Pre-filtering**: A first pass removes pairs that are fundamentally unsuitable. Any reply that has fewer words than `MIN_WORDS` or an average transcription confidence below `MIN_CONF` is immediately discarded.

6.  **Multi-Factor Quality Scoring**: Each surviving pair is then passed to a sophisticated scoring engine. It is graded across **8 different metrics**: semantic similarity, speaker overlap, length ratio, speaker diversity, transcription confidence, punctuation, and input/output length. Each metric is weighted according to the `WEIGHT_*` parameters.

7.  **Final Selection**: The weighted scores are combined into a single `quality_score`. If this final score is greater than `MIN_QUALITY_SCORE`, the pair is saved to the final dataset. Otherwise, it is discarded.

This multi-stage process ensures that only the most contextually relevant, clean, and conversationally appropriate pairs make it to the final output.

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

## 🧠 Cache Invalidation Guide

To ensure your results always reflect your latest settings, it's critical to clear the cache when you change certain parameters. This guide explains which files to delete. When in doubt, deleting the entire `cache/` folder for the audio file you are processing is the safest way to force a complete re-run.

**Cache File Suffixes:**

*   `.wav`: The standardized audio file.
*   `.transcript.json`: Raw output from WhisperX transcription.
*   `.aligned.json`: Word-level timestamps from alignment.
*   `.diarization.json`: Speaker turn identification.
*   `.assigned.json`: Words with speakers assigned.
*   `_pairs_...jsonl`, `_pair_...csv`, etc.: Files related to pair building, scoring, and filtering.

---

### 1. Full Re-run Required

If you change **any** of the following parameters, you **must delete the entire `cache/` folder** for the audio file you are processing. These settings fundamentally alter the earliest, most computationally expensive stages of the pipeline.

*   **Audio Source Files:**
    *   Modifying the source audio in `audio_files/`.
*   **Transcription (`.transcript.json`):**
    *   `WHISPER_MODEL`
    *   `WHISPER_BATCH_SIZE`
    *   `USE_VAD`
    *   `LANGUAGE`
    *   Any values within the `PROFILES` dictionary (e.g., `VAD_MULT`).
*   **Diarization (`.diarization.json`):**
    *   Any changes to the `pyannote.audio` models or your `HF_TOKEN`.

---

### 2. Partial Re-run (Pair Building & Scoring)

If you **only** change the parameters below, you can keep the foundational cache files (`.wav`, `.transcript.json`, `.aligned.json`, `.diarization.json`, and `.assigned.json`).

*   **Delete:** All files in the `cache/` folder that start with your audio file's name and end with `_pairs_...` or `_pair_...` (e.g., `*_pairs_with_overlap_meta.jsonl`, `*_pair_quality_metrics.csv`, etc.).
*   **Applicable Parameters:**
    *   **Speaker Identification:**
        *   `IDENTIFY_THRESHOLD`
        *   `CONTEXTUAL_REID_THRESHOLD`
        *   Changing the sample files in `samples/`.
    *   **Pair Building & Filtering:**
        *   `MERGE_GAP`
        *   `MAX_CONTEXT`
        *   `MIN_WORDS`
        *   `MIN_CONF`
        *   `OVERLAP_FRAC`
    *   **Quality Scoring & Final Selection:**
        *   `MIN_QUALITY_SCORE`
        *   All `WEIGHT_*` parameters.

---

## 📊 Output Schema

The final `_results.jsonl` files contain a rich set of data for each training pair. This allows for powerful, nuanced filtering in downstream processing steps.

| Field | Type | Description |
| :--- | :--- | :--- |
| `seg_idx` | String | An internal identifier for the segment the pair was generated from. |
| `input` | String | The context/prompt for the response. |
| `output` | String | The response from the target speaker. |
| `input_len` | Integer | The number of words in the input. |
| `output_len` | Integer | The number of words in the output. |
| `output_speaker` | String | The identified speaker label for the person responding (e.g., `SPEAKER_01`). |
| `source_filename` | String | The name of the original audio file this pair was extracted from. |
| `pair_format` | String | The format of the pair. Either `conversational` or `qna`. |
| `target_speaker_word_ratio` | Float | The ratio of words spoken by the target speaker in the entire audio file. |
| `num_context_turns` | Integer | The number of distinct speaker turns in the `input` context. |
| `input_sentence_count` | Integer | The number of sentences in the `input`. |
| `output_sentence_count` | Integer | The number of sentences in the `output`. |
| `input_ttr` | Float | The Type-Token Ratio (lexical diversity) of the `input`. |
| `output_ttr` | Float | The Type-Token Ratio (lexical diversity) of the `output`. |
| `quality_score` | Float | The final, combined quality score for the pair (0.0 to 1.0). |
| `quality_metrics` | Object | A dictionary containing the 8 individual metrics that make up the quality score. |
| `avg_score` | Float | The average word-level transcription confidence from WhisperX for the `output`. |
| `tstart` / `tend` | Float | The start and end time (in seconds) of the `output` utterance in the audio. |
| `words` | Array | A detailed list of word objects for the `output`, including timestamps and confidence. |