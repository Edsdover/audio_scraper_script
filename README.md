# 🎙️ Podcast Processing Pipeline

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

A reproducible audio-to-dataset pipeline that transforms long-form podcast audio into (context → response) training pairs for a target speaker.

## ✨ Features

* **Automatic transcription** with [WhisperX](https://github.com/m-bain/whisperx).
* **Word-level alignment** with phoneme timing.
* **Speaker diarization** using [pyannote.audio](https://github.com/pyannote/pyannote-audio).
* **Target speaker identification** with [Resemblyzer](https://github.com/resemble-ai/Resemblyzer).
* **Pair building**: creates clean `(context → response)` data from a single speaker.
* **Caching** of heavy intermediate steps for faster iteration.
* **FFmpeg auto-setup** (Windows-friendly, no PATH tweaks).

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone <your-repo-url>
cd podcast-pipeline

# 1) Install **Python 3.11** (required).

# 2) Run the pipeline
# Double click the script podcast_pipeline_gpu.py
```

Input: `podcast.mp3` (place in repo root)
Output: `results.jsonl` (final dataset)

---

## Quick facts & recommended versions
- **Tested with:** Windows 10/11 (developer logs show Windows), Python **3.11.x** (many wheels are `cp311`). Use Python 3.11 for best compatibility.
- **Torch:** The pinned wheel in `requirements_windows.txt` is `torch==2.2.2+cu118` (CUDA 11.8). If you don't have an NVIDIA GPU, install a CPU-only Torch variant or change `requirements_windows.txt`.
- **ffmpeg:** Required for audio I/O. The pipeline will attempt to download a local `ffmpeg-bin` if missing, but it's recommended to install `ffmpeg` and put it on your PATH (or put `ffmpeg.exe` in `./ffmpeg-bin/`).


## ⚙️ Config knobs (tunable constants)

All defined at the top of `podcast_pipeline_gpu.py`:

* `WHISPER_MODEL` → WhisperX model name (default: `large-v2`).
* `WHISPER_BATCH_SIZE` → batch size for transcription.
* `MERGE_GAP` → max silence (s) to merge replies.
* `MIN_WORDS` → min words per reply.
* `MIN_CONF` → min avg word confidence.
* `OVERLAP_FRAC` → max overlap with non-target before dropping.
* `IDENTIFY_THRESHOLD` → speaker similarity cutoff.
* `KEEP_DEBUG_OUTPUTS` → keep or clean intermediate JSONL.
* `OVERLAP_THRESHOLD` → fraction of words overlapped with another speaker that triggers a flag (default: 0.40).
* `LOW_CONFUDEBCE_THRESHOLD` → minimum average transcription confidence before a pair is flagged for review (default: 0.40).
* `INPUT_TO_OUTPUT_RATIO` → if input words / output words ≥ this ratio, flag as “long input vs. tiny output” (default: 4.0).


---

## 📂 Files & cache artifacts

* **Input:** `podcast.mp3`
* **Transcript:** `podcast.transcript.json`
* **Alignment:** `podcast.aligned.json`
* **Diarization:** `podcast.diarization.json`
* **Assigned speakers:** `podcast.assigned.json`
* **Final output:** `results.jsonl`
* **Debug pairs:** `pairs_merged.jsonl`, `pairs_with_overlap_meta.jsonl`, `pairs_clean.jsonl`

💡 To re-run a stage: delete the related cache file.

---

## 🔑 Environment

* Python **3.11+**
* Torch (CUDA build for GPU acceleration)
* Hugging Face token (`HF_TOKEN`) for WhisperX/pyannote
* FFmpeg (auto-handled by the

# 🎙️ Podcast Pipeline Overview

## Stages & Key Functions

1. **Entry**
   - `main` → `run_pipeline`

2. **Transcription**
   - `transcribe_with_vad`

3. **Alignment**
   - `attach_speakers_to_transcript`
     - uses `safe_assign_word_speakers`

4. **Diarization & Target Selection**
   - `parse_diarization_json_to_segments`
   - `identify_speaker` → `identify_best_speaker` → `load_reference_embeddings`

5. **Pair Building & Filtering**
   - `adaptive_conf_filter` (confidence cleanup)
   - `run_vad_with_fallback` → `run_energy_vad` (speech activity)
   - `overlaps_vad` (keep words inside speech)
   - `reassign_short_target_words` → `contextual_reid_with_embeddings` (fix mis-assignments)
   - `regroup_words_to_segments`
   - `build_pairs` → `make_pair`, `clean_input_words`
   - `build_pairs_detailed`

6. **Pair Processing**
   - Phase A: `merge_close_replies`
   - Phase B: `compute_overlap_and_write` → `score_pair_quality`
   - Phase C: `postfilter_clean_pairs`
   - Flagging: `auto_flag_low_quality_pairs`

7. **Finalization**
   - Write `results.jsonl` + metrics
   - `print_timings_summary`
   - `mark_step` used for timing

---

## 🔑 Critical Flow
# adaptive_conf_filter → VAD → reassign_short_target_words → regroup_words_to_segments → build_pairs → merge_close_replies → compute_overlap_and_write (+ scoring) → postfilter_clean_pairs

`main
 └─ run_pipeline
     ├─ transcribe_with_vad
     ├─ attach_speakers_to_transcript
     │    └─ safe_assign_word_speakers
     ├─ parse_diarization_json_to_segments
     ├─ identify_speaker
     │    └─ identify_best_speaker
     │         └─ load_reference_embeddings
     └─ STEP 5: Build pairs
          ├─ adaptive_conf_filter
          ├─ run_vad_with_fallback
          │    └─ run_energy_vad
          ├─ overlaps_vad
          ├─ reassign_short_target_words
          │    └─ contextual_reid_with_embeddings
          ├─ regroup_words_to_segments
          ├─ build_pairs
          │    ├─ make_pair
          │    └─ clean_input_words
          ├─ build_pairs_detailed
          ├─ merge_close_replies
          ├─ compute_overlap_and_write
          │    └─ score_pair_quality
          ├─ postfilter_clean_pairs
          └─ auto_flag_low_quality_pairs`
