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
2.  **Install Python**: Ensure you have **Python 3.11.x** installed.
3.  **Place Your Files**:
    * Put your source audio file in the root directory and name it `podcast.mp3`.
    * Create a `samples` folder and place one or more short audio clips (.wav, .mp3) of the target speaker's voice inside (10-200 sec clips).
4.  **Run the Pipeline**:
    * On Windows, simply double-click `run_pipeline.bat`. It will create a virtual environment, install dependencies, and start the process.

**Input**: `podcast.mp3` & `samples/` folder
**Output**: `results.jsonl` (final dataset)

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

## 📂 Files & Artifacts

The pipeline generates several files. Caching allows you to re-run the process without repeating heavy steps. **To re-run a specific stage, just delete its corresponding `.json` file.**

### Cache Files
* `podcast.transcript.json`: Raw transcription output from WhisperX.
* `podcast.aligned.json`: Transcript with word-level timestamps.
* `podcast.diarization.json`: Speaker diarization timeline.
* `podcast.assigned.json`: Transcript with a speaker assigned to each word.

### Output Files
* **`results.jsonl`**: The final, cleaned dataset of `(input, output)` pairs.
* `pairs_flagged.jsonl`: A list of low-quality pairs flagged for manual review.
* `pair_quality_metrics.csv`: A CSV file with quality scores for every generated pair.
* `base_pairs.jsonl`: Raw pairs generated before merging and advanced filtering.

### Debug Outputs (if `KEEP_DEBUG_OUTPUTS` is `True`)
* `pairs_merged.jsonl`: Pairs after the reply merging step.
* `pairs_with_overlap_meta.jsonl`: Pairs with detailed, per-word overlap data.
* `pairs_clean.jsonl`: Pairs after overlapped words have been removed from responses.

---

## 🧠 Pipeline Logic

### Critical Data Filtering Flow

The core of the pair-building logic follows this multi-stage filtering process to ensure high data quality:

1.  **Adaptive Confidence Filter**: Removes low-confidence words based on per-speaker statistical thresholds, which is more robust than a single global cutoff.
2.  **Neural VAD Filter**: Identifies all segments of active speech. Any transcribed words falling outside these segments (often hallucinations in silent parts) are discarded.
3.  **Short Word Reassignment**: Corrects common diarization errors where short interjections (e.g., "yes," "ok") from the target speaker are misattributed to the host.
4.  **Regrouping**: The filtered, word-level data is reassembled into contiguous speaker segments.
5.  **Pair Construction**: The pipeline iterates through segments, assigning non-target speech as "input" and target speech as "output."
6.  **Reply Merging**: Consecutive replies from the target speaker are stitched together to form longer, more coherent outputs.
7.  **Overlap Removal**: A detailed, per-word analysis removes any part of the target's speech that overlaps with other speakers, cleaning up cross-talk.
8.  **Final Filtering**: The cleaned pairs are passed through final checks for minimum length and confidence.

### Function Call Hierarchy

This shows the main logical flow of the script from entry point to the core functions.

* `main()`
    * `run_pipeline()`
        * `transcribe_with_vad()`
        * `attach_speakers_to_transcript()`
            * `safe_assign_word_speakers()`
        * `identify_speaker()`
            * `identify_best_speaker()`
                * `load_reference_embeddings()`
        * **Pair Building & Filtering (Step 5)**
            * `adaptive_conf_filter()`
            * `run_vad_with_fallback()`
            * `reassign_short_target_words()`
            * `build_pairs_detailed()`
            * `merge_close_replies()`
            * `compute_overlap_and_write()`
                * `score_pair_quality()`
            * `postfilter_clean_pairs()`
            * `auto_flag_low_quality_pairs()`