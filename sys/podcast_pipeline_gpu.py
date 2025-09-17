import io
import json
import logging
import os
import shutil
import sys
import time
import zipfile
import tempfile
import re
import csv
import traceback
import warnings
from tqdm import tqdm
import math

import numpy as np
import requests
import statistics
import torch
import whisperx
import nltk

from pyannote.audio import Pipeline
from numpy.linalg import norm
from numpy import dot
from pydub import AudioSegment
from pyannote.core import Annotation, Segment
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# ============================================================ 
# 🎙️ Pipeline Configuration
# All the knobs you might want to tweak live here.
# Adjust these values to control how the pipeline behaves.
# ============================================================ 
os.environ['PYTHONIOENCODING'] = 'utf-8'
KEEP_DEBUG_OUTPUTS = True # True = keep merged/clean/overlap files, False = only save final cleaned_data.jsonl
LANGUAGE = "en"
# WhisperX Supports the following for transcription: Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Bashkir, Basque, Belarusian, Bengali, Bosnian, Breton, Bulgarian, Burmese, Castilian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Faroese, Finnish, Flemish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hausa, Hawaiian, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Lao, Latin, Latvian, Letzeburgesch, Lithuanian, Luxembourgish, Macedonian, Malagasy, Malay, Malayalam, Maltese, Maori, Marathi, Moldavian, Moldovan, Mongolian, Nepali, Norwegian, Nynorsk, Occitan, Pashto, Persian, Polish, Portuguese, Punjabi, Pushto, Romanian, Russian, Sanskrit, Serbian, Shona, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Tibetan, Turkish, Turkmen, Ukrainian, Urdu, Uzbek, Valencian, Vietnamese, Welsh, Yiddish, Yoruba.
# --- Pair building & filtering ---
MERGE_GAP = 3.0 # (seconds) Merge Target’s replies if the silence between them is shorter than this.
MIN_WORDS = 5 # (words) Minimum length of Target’s reply to keep it. Shorter replies are dropped.
MIN_CONF = 0.5 # (0.0–1.0 or None) Drop replies with avg confidence below this. Set None to disable.
OVERLAP_FRAC = 0.80 # (fraction) If more than this % of a word overlaps with another speaker, drop it.
MIN_QUALITY_SCORE = 0.55 # (0.0-1.0) Minimum quality score to keep a pair.

# --- Speaker identification ---
IDENTIFY_THRESHOLD = 0.70 # (0.0–1.0) How strict to be when matching Target’s voice to a diarized speaker.
CONTEXTUAL_REID_THRESHOLD = 0.72 # (0.0-1.0) Similarity threshold for re-assigning short words.

# --- Quality scoring weights ---
WEIGHT_AVG_CONF = 0.5
WEIGHT_OVERLAP = 1.0
WEIGHT_RATIO = 1.0
WEIGHT_SIMILARITY = 1.0
WEIGHT_PUNC = 0.5
WEIGHT_DIVERSITY = 0.5
WEIGHT_INPUT_LEN = 0.1
WEIGHT_OUTPUT_LEN = 0.1

# --- Advanced fine-tuning ---
REASSIGN_WINDOW = 1.0 # (seconds) Time window to check for neighboring speakers when reassigning short words.
SHORT_REPLY_MIN_CONF = 0.85 # (0.0-1.0) Confidence threshold to keep a single-word reply (e.g., "Exactly!").

# --- Embedding extraction ---
MIN_EMB_DURATION = 1.0 # (seconds) Ignore very short diarization segments when building speaker embeddings.

# --- Transcription (WhisperX) ---
WHISPER_MODEL = "large-v2"  # Options:
# |  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
# |:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
# |  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
# |  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
# | small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
# | medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
# | large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
# | turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |
#---------------------------------------------------------------------------------------------------

WHISPER_BATCH_SIZE = 16     # increase if you have GPU memory (32 or 16)
USE_VAD = True              # If True, split audio with VAD and transcribe segments individually

# --- Auto-flagging thresholds (for review dataset) ---
OVERLAP_THRESHOLD = 0.80         # (fraction) Flag a pair if ≥ this % of its words overlap with another speaker
LOW_CONFIDENCE_THRESHOLD = 0.30  # (0.0–1.0) Flag a pair if avg confidence is below this
INPUT_TO_OUTPUT_RATIO = 10.0      # (ratio) Flag a pair if input_len / output_len exceeds this value (long input → tiny output)

# --- Constants ---
MAX_INPUT_WORDS = 50 # Cap host inputs (truncate if longer)
MAX_RATIO = 4.0 # Filter out pairs where input_len / output_len > this value

# --- File paths ---
CACHE_FOLDER = "cache"
RESULTS_FOLDER = "results"
SYS_FOLDER = "sys"
# Generate a timestamped log file for each run
run_timestamp = time.strftime("%Y%m%d-%H%M%S")
LOG_FILE = os.path.join(CACHE_FOLDER, f"pipeline_{run_timestamp}.log")

# ============================================== 
# 🔧 Quality Control Configurations
# ============================================== 

# Choose which profile to use: "balanced" or "conservative"
QUALITY_PROFILE = "balanced"

PROFILES = {
    "relaxed": {
        # Very permissive — keep more data, tolerate some noise
        "ADAPTIVE_CONF_DELTA": 0.15,   # allow words further below median
        "ADAPTIVE_CONF_MIN": 0.20,     # lowest floor
        "VAD_MULT": 0.2,               # very loose VAD threshold
    },
    "balanced": {
        # Good tradeoff — clean but not too tiny
        "ADAPTIVE_CONF_DELTA": 0.10,
        "ADAPTIVE_CONF_MIN": 0.25,
        "VAD_MULT": 0.3,
    },
    "conservative": {
        # Very strict — gold-standard small set
        "ADAPTIVE_CONF_DELTA": 0.05,
        "ADAPTIVE_CONF_MIN": 0.30,
        "VAD_MULT": 0.5,
    }
}

CFG = PROFILES[QUALITY_PROFILE]
ADAPTIVE_STRATEGY = "percentile" # Strategy for adaptive confidence filter: "percentile" or "median" 

NON_TARGET_FILTER = {
    # Acknowledgements / backchannels
    "uh", "um", "ah", "eh", "er", "hmm",
    "mhm", "mm-hmm", "uh-huh", "huh",
    "ooh", "oh", "yeah", "ya", "ok", "okay", "alright",
    "yep", "yup", "nah", "for sure", "exactly", "totally",

    # Common conversational filler
    "like", "so", "you know", "i mean", "right", "well", "actually", "basically", "literally",
    "i guess", "i think", "sort of", "kind of", "you see", "i know", "believe me", "i'm sure",
    "of course",

    # Expletives (can be filtered if not desired in training data)
    # "whore", "slut", "bitch", "retard", "fuck", "shit",

    # Vague lead-ins or affirmations
    "i see", "gotcha", "cool", "wow", "oh wow",

    # Common questions that might not be part of the main content
    "what", "really", "thank you", "thanks",
}

# ------------------------- 
# Script Setup
# ------------------------- 
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception as e:
    logging.warning(f"[SETUP] NLTK data download failed: {e}. Proceeding with an empty stopword list.")
    STOPWORDS = set()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logging.error("[SETUP] CRITICAL: Hugging Face token (HF_TOKEN) not found.")
    logging.error("[SETUP] Please provide your token by running the script via `run_pipeline.bat` or by setting the HF_TOKEN environment variable.")
    sys.exit(1)

step_timings = {} # collect per-step durations
device = "cuda" if torch.cuda.is_available() else "cpu"

ffmpeg_dir = os.path.join(SYS_FOLDER, "ffmpeg-bin")
os.makedirs(ffmpeg_dir, exist_ok=True)

ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe.exe")

# ------------------------------ 
# FFMPEG bootstrap (download if missing)
# ------------------------------ 
if not os.path.exists(ffmpeg_exe) or not os.path.exists(ffprobe_exe):
    print("[FFMPEG] ffmpeg/ffprobe not found, downloading...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    r = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            if name.endswith("ffmpeg.exe"):
                with z.open(name) as src, open(ffmpeg_exe, "wb") as dst:
                    dst.write(src.read())
            if name.endswith("ffprobe.exe"):
                with z.open(name) as src, open(ffprobe_exe, "wb") as dst:
                    dst.write(src.read())


# Update PATH so subprocess + pydub can see them
os.environ["PATH"] += os.pathsep + ffmpeg_dir

# Tell pydub explicitly
AudioSegment.converter = ffmpeg_exe
AudioSegment.ffprobe = ffprobe_exe

# ------------------------------------------------- 
# TQDM-Aware Logging & Full Output Capture
# ------------------------------------------------- 
class TqdmLoggingHandler(logging.Handler):
    """Redirects logging messages to tqdm.write() to avoid progress bar corruption."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

def configure_loggers():
    """Set up the root logger and configure library loggers."""
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    # Create handlers
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(logging.INFO)
    tqdm_handler.setFormatter(formatter)

    # Add handlers to the root logger
    log.addHandler(file_handler)
    log.addHandler(tqdm_handler)

    # Configure library loggers to use the same handlers
    # This prevents them from creating their own handlers and writing directly to console
    libraries_to_configure = ["pyannote", "whisperx", "speechbrain", "numba"]
    for lib_name in libraries_to_configure:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.propagate = True # Propagate to root logger
        for handler in lib_logger.handlers[:]:
            lib_logger.removeHandler(handler)

configure_loggers()
log = logging.getLogger(__name__)

log.info("=== Pipeline started ===")
log.info(f"[FFMPEG] Using ffmpeg: {ffmpeg_exe}")
log.info(f"[FFMPEG] Using ffprobe: {ffprobe_exe}")


# ------------------------- 
# Function Definitions
# ------------------------- 
def mark_step(step_name, start_time):
    """Record elapsed time for a pipeline step."""
    elapsed = time.time() - start_time
    step_timings[step_name] = elapsed
    return time.time()  # reset timer for next step

def regroup_words_to_segments(words):
    """Wrap flat word list back into one segment for build_pairs."""
    if not words:
        return []
    return [{
        "speaker": None,  # actual speaker info is inside each word
        "words": words,
        "start": words[0].get("start"),
        "end": words[-1].get("end"),
    }]

def clean_input_words(words, target_label, filler_words=None):
    """
    Clean up non-target input words.
    Removes filler words (default = NON_TARGET_FILTER).
    Ensures only dict entries with text-like content are kept.
    Returns (cleaned_input, shifted_output) where shifted_output is empty
    (kept for consistency with build_pairs).
    """
    if filler_words is None:
        filler_words = NON_TARGET_FILTER  # use your global set

    cleaned = []
    shifted_output = []

    for w in words:
        if not isinstance(w, dict):
            continue
        txt = (w.get("text") or w.get("word") or w.get("token") or "").lower().strip()
        if not txt:
            continue
        if txt in filler_words:
            continue
        cleaned.append(w)

    return cleaned, shifted_output

def score_pair(pair, all_words, target_label):
    """
    Compute a soft quality score for a training pair.
    - Confidence: mean score of output words
    - Length ratio: input vs output balance
    - Completeness: output ends with sentence punctuation
    Returns a dict with components and final score.
    """
    out_text = pair.get("output") or ""
    output_words = [
        w for w in all_words
        if w.get("speaker") == target_label and w.get("word") in out_text
    ]
    if not output_words:
        return {"confidence": 0.0, "ratio": 0.0, "complete": 0.0, "score": 0.0}

    conf = sum(w.get("score", 0.0) for w in output_words) / len(output_words)
    input_len = len(pair.get("input", "").split())
    output_len = len(pair.get("output", "").split())
    ratio = min(input_len / output_len if output_len > 0 else 0, 4.0)
    out_text = (pair.get("output") or "").strip()
    complete = 1.0 if out_text and out_text[-1] in ".?!" else 0.0
    
    final = 0.5 * conf + 0.3 * ratio + 0.2 * complete
    return {"confidence": conf, "ratio": ratio, "complete": complete, "score": final}

def compute_speaker_embeddings(audio_file, diarization, min_duration=MIN_EMB_DURATION):
    """
    Compute averaged embeddings for each diarized speaker.
    Skips very short segments (default < 1s).
    This implementation extracts short wav files and reuses _embed_audio_segment
    (which calls preprocess_wav on the file path) so it matches Resemblyzer expectations.
    """
    encoder = VoiceEncoder()
    speaker_embeddings = {}

    # Load full audio once (pydub uses milliseconds)
    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        log.error(f"[EMBED] Cannot load audio file {audio_file}: {e}")
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        i = 0
        # Accept pyannote.Annotation via itertracks or fallback to list/dict
        if hasattr(diarization, "itertracks"):
            tracks = list(diarization.itertracks(yield_label=True))
            for track in tqdm(diarization.itertracks(yield_label=True), desc="Computing embeddings"):
                seg, _, label = track
                dur = seg.end - seg.start
                if dur < min_duration:
                    continue
                s_ms = int(seg.start * 1000); e_ms = int(seg.end * 1000)
                tmp_path = os.path.join(tmpdir, f"seg_embed_{i}_{label}.wav")
                try:
                    audio[s_ms:e_ms].export(tmp_path, format="wav")
                    emb = _embed_audio_segment(tmp_path, encoder)
                    if emb is not None:
                        speaker_embeddings.setdefault(label, []).append(emb)
                except Exception as e:
                    log.debug(f"[EMBED] Failed embedding segment {label} {seg}: {e}")
                i += 1
        else:
            # fallback if diarization is a list/dict of segments
            entries = diarization if isinstance(diarization, list) else (diarization.get("content") or diarization.get("segments") or [])
            for idx, entry in enumerate(entries):
                seg = entry.get("segment") if isinstance(entry, dict) else None
                label = entry.get("label") or entry.get("speaker")
                if not seg or label is None:
                    continue
                start = float(seg.get("start", 0.0)); end = float(seg.get("end", start))
                dur = end - start
                if dur < min_duration:
                    continue
                s_ms = int(start * 1000); e_ms = int(end * 1000)
                tmp_path = os.path.join(tmpdir, f"seg_embed_{idx}_{label}.wav")
                try:
                    audio[s_ms:e_ms].export(tmp_path, format="wav")
                    emb = _embed_audio_segment(tmp_path, encoder)
                    if emb is not None:
                        speaker_embeddings.setdefault(label, []).append(emb)
                except Exception as e:
                    log.debug(f"[EMBED] Failed embedding segment {label} {start}->{end}: {e}")

    # Average + normalize per speaker
    averaged_embeddings = {}
    for spk, embs in speaker_embeddings.items():
        if not embs:
            continue
        embs = [e / np.linalg.norm(e) for e in embs if np.linalg.norm(e) > 0]
        avg = np.mean(embs, axis=0)
        avg /= np.linalg.norm(avg)
        averaged_embeddings[spk] = avg
        log.info(f"[EMBED] Speaker {spk}: {len(embs)} segments used.")

    log.info(f"[EMBED] Computed embeddings for {len(averaged_embeddings)} speakers.")
    return averaged_embeddings

def overlaps_vad(word, vad_segments):
    """
    Return True if the word [start,end] overlaps any VAD segment.
    Robust: returns False if timing is missing or invalid.
    vad_segments is a list of (start,end) tuples or objects convertible to floats.
    Caveat: this function is conservative: if times are missing it
    will treat the word as non-speech for VAD filtering (safe).
    """
    # normalize start/end
    start = word.get("start")
    end = word.get("end")
    if start is None or end is None:
        return False
    try:
        start_f = float(start)
        end_f = float(end)
    except Exception:
        return False

    for seg in vad_segments:
        # seg might be (s,e) tuple or object with start/end floats
        try:
            if isinstance(seg, (tuple, list)) and len(seg) >= 2:
                s = float(seg[0]); e = float(seg[1])
            else:
                # attempt object-like access
                s = float(getattr(seg, "start", seg.get("start") if isinstance(seg, dict) else None))
                e = float(getattr(seg, "end", seg.get("end") if isinstance(seg, dict) else None))
        except Exception:
            continue

        # overlap test
        if not (end_f <= s or start_f >= e):
            return True
    return False

def _embed_audio_segment(wav_path, encoder):
    """Return embedding or None on failure."""
    try:
        wav = preprocess_wav(wav_path)
        emb = encoder.embed_utterance(wav)
        return emb
    except Exception as e:
        log.debug(f"[IDENTIFY] Embedding failed for {wav_path}: {e}")
        return None
    
def load_reference_embeddings(samples_folder):
    """
    Load reference embedding(s) from the provided samples_folder.
    This expects at least one audio file in the folder; returns (list_of_ref_embs, target_avg_emb).
    Raises a clear Exception if no valid sample files found.
    """
    encoder = VoiceEncoder()
    ref_embs = []
    if not os.path.isdir(samples_folder):
        raise FileNotFoundError(f"Reference samples folder not found: {samples_folder}")
    # accept common audio extensions
    exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    files = [os.path.join(samples_folder, f) for f in os.listdir(samples_folder) if f.lower().endswith(exts)]
    if not files:
        raise FileNotFoundError(f"No audio files found in samples folder: {samples_folder}")
    for fpath in files:
        try:
            emb = _embed_audio_segment(fpath, encoder)
            if emb is not None:
                ref_embs.append(emb)
        except Exception as e:
            log.warning(f"[IDENTIFY] Skipping sample {fpath}: {e}")
    if not ref_embs:
        raise ValueError("No valid reference embeddings found in samples folder.")
    # average to single target embedding
    target_avg_emb = np.mean(np.stack(ref_embs, axis=0), axis=0)
    return ref_embs, target_avg_emb

def annotation_to_df(annotation):
    """
    Converts a pyannote.core.annotation.Annotation object to a pandas DataFrame.
    """
    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {"start": segment.start, "end": segment.end, "speaker": speaker}
        )
    return pd.DataFrame(segments)

def print_final_batch_summary(setup_time, per_file_times, total_time):
    """Prints a final, comprehensive summary for the entire batch run."""
    log.info("\n" + "="*50)
    log.info("BATCH PROCESSING SUMMARY")
    log.info("="*50)

    # Format and print setup time
    s_mins, s_secs = divmod(int(setup_time), 60)
    log.info(f"Script Setup Time : {s_mins}m {s_secs}s")
    log.info("-"*50)

    # Format and print per-file times
    log.info("Per-File Processing Times:")
    if not per_file_times:
        log.info("  No files were processed.")
    else:
        for filename, duration in per_file_times.items():
            if duration == -1:
                log.info(f"  - {filename}: FAILED")
            else:
                f_mins, f_secs = divmod(int(duration), 60)
                log.info(f"  - {filename}: {f_mins}m {f_secs}s")
    log.info("-"*50)
    
    # Format and print total time
    t_mins, t_secs = divmod(int(total_time), 60)
    log.info(f"Total Run Time    : {t_mins}m {t_secs}s")
    log.info("="*50 + "\n")

def safe_assign_word_speakers(diarize_segments, result, diar_cache_path, base_name):
    """
    Try to assign speakers using WhisperX's built-in method.
    If it fails (due to serialization, version, or overlap issues),
    fall back to a manual attachment method.
    """
    try:
        log.info("[ASSIGN] Attempting whisperx.assign_word_speakers...")
        diarization_df = annotation_to_df(diarize_segments)
        assigned = whisperx.assign_word_speakers(diarization_df, result)
        log.info("[ASSIGN] Successfully assigned speakers with whisperx.")
        return assigned
    except Exception as e:
        log.warning(f"[ASSIGN] whisperx.assign_word_speakers failed. Traceback:")
        log.warning(traceback.format_exc())
        log.info("[ASSIGN] Falling back to custom attach_speakers_to_transcript...")
        try:
            assigned = attach_speakers_to_transcript(diarize_segments, result, diarization_json_path=diar_cache_path, base_name=base_name)
            log.info("[ASSIGN] Successfully assigned speakers with fallback method.")
            return assigned
        except Exception as inner_e:
            log.error(f"[ASSIGN] Fallback attach_speakers_to_transcript also failed: {inner_e}")
            raise
    
def attach_speakers_to_transcript(diarization, aligned_result, diarization_json_path, base_name):
    """
    Robustly attach speaker labels to each word in an aligned transcript.
    - If 'diarization' has itertracks(), prefer that (pyannote Annotation).
    - Otherwise, read diarization from `diarization_json_path` and perform max-overlap.
    Returns aligned_result with word['speaker'] set.
    """
    import json
    log.info("[ATTACH] Attaching speakers to aligned transcript...")

    # Helper: attach using pyannote Annotation if present
    if hasattr(diarization, "itertracks"):
        diar_segments = []
        for item in diarization.itertracks(yield_label=True):
            if len(item) == 3:
                seg, _, label = item
            elif len(item) == 2:
                seg, label = item
            else:
                continue
            diar_segments.append((float(seg.start), float(seg.end), label))
        log.info(f"[ATTACH] Using pyannote Annotation -> {len(diar_segments)} segments")
    else:
        # Fallback: load diarization JSON saved by pipeline
        try:
            with open(diarization_json_path, "r", encoding="utf-8") as f:
                diar_json = json.load(f)
            diar_segments = []
            content = diar_json.get("content") or diar_json.get("segments") or []
            for entry in content:
                seg = entry.get("segment") if isinstance(entry, dict) else None
                if seg:
                    start = seg.get("start")
                    end = seg.get("end")
                    label = entry.get("label") or entry.get("speaker")
                    if start is None or end is None or label is None:
                        continue
                    diar_segments.append((float(start), float(end), label))
            diar_segments.sort(key=lambda x: x[0])
            log.info(f"[ATTACH] Loaded {len(diar_segments)} diarization segments from JSON")
        except Exception as e:
            log.error(f"[ATTACH] Failed to load diarization JSON: {e}")
            diar_segments = []

    # Attach speaker label to each word by picking diarization label with max overlap
    attached = 0
    total_words = 0
    for seg in aligned_result.get("segments", []):
        for word in seg.get("words", []):
            total_words += 1
            wstart = float(word.get("start") or 0.0)
            wend = float(word.get("end") or wstart)
            best_label = None
            best_overlap = 0.0
            for dstart, dend, label in diar_segments:
                overlap = max(0.0, min(wend, dend) - max(wstart, dstart))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label
            if best_label and best_overlap > 0.0:
                word["speaker"] = best_label
                attached += 1
            else:
                word["speaker"] = "unknown"

    # Logging summary
    counts = {}
    for seg in aligned_result.get("segments", []):
        for w in seg.get("words", []):
            counts[w.get("speaker", "unknown")] = counts.get(w.get("speaker","unknown"), 0) + 1
    log.info(f"[ATTACH] Attached {attached}/{total_words} words. Per-speaker sample: {dict(list(counts.items())[:10])}")

    # Optionally write a debug copy of the attached alignment
    try:
        debug_path = os.path.join(CACHE_FOLDER, f"{base_name}.aligned.attached.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(aligned_result, f, ensure_ascii=False, indent=2)
        log.info(f"[ATTACH] Wrote {os.path.basename(debug_path)} for inspection.")
    except Exception:
        pass

    return aligned_result

def adaptive_conf_filter(words, cutoff_strategy=ADAPTIVE_STRATEGY, min_cutoff=0.4, grace=0.1):
    """
    Adaptive filter:
    - Uses per-speaker cutoff (percentile or median)
    - Lower min_cutoff (default 0.4 instead of 0.6)
    - Keeps words slightly below cutoff if within grace zone
    """
    by_speaker = defaultdict(list)
    for w in words:
        if isinstance(w.get("score"), (int, float)):
            by_speaker[w.get("speaker")].append(w["score"])

    cutoffs = {}
    for spk, scores in by_speaker.items():
        if not scores:
            cutoffs[spk] = min_cutoff
        elif cutoff_strategy == "percentile":
            # 30th percentile = less strict than median
            cutoffs[spk] = max(min_cutoff, float(np.percentile(scores, 30)))
        elif cutoff_strategy == "median":
            cutoffs[spk] = max(min_cutoff, statistics.median(scores))
        else:
            cutoffs[spk] = min_cutoff

    kept, removed = [], []
    for w in words:
        score = w.get("score")
        cutoff = cutoffs.get(w.get("speaker"), min_cutoff)
        if score is None or score >= cutoff - grace:  # Keep words that are only slightly below the cutoff
            kept.append(w)
        else:
            removed.append(w)

    return kept, removed, cutoffs

def run_energy_vad(wav_path, frame_ms=30, vad_mult=CFG["VAD_MULT"]):
    import wave, contextlib, numpy as np, math, statistics
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        sr = wf.getframerate()
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)/32768.0
    frame_len = int(sr*frame_ms/1000)
    energies = [math.sqrt(float((samples[i*frame_len:(i+1)*frame_len]**2).mean()))
                for i in range(len(samples)//frame_len)]
    mean_e, stdev_e = statistics.mean(energies), statistics.pstdev(energies)
    thresh = mean_e + vad_mult*stdev_e
    flags = [e>thresh for e in energies]
    # merge frames to segments
    segs, cur = [], None
    for i,f in enumerate(flags):
        t0=i*frame_ms/1000; t1=(i+1)*frame_ms/1000
        if f: cur=[t0,t1] if cur is None else [cur[0],t1] # merge
        elif cur: segs.append(tuple(cur)); cur=None # end segment
    if cur: segs.append(tuple(cur)) # add last segment if active
    return segs

def run_vad_with_fallback(wav_path, vad_mult=CFG["VAD_MULT"], hf_token_env="HF_HUB_TOKEN"):
    """
    Try neural VAD (pyannote pipeline) first. If anything fails, fall back to energy VAD.
    Returns list of (start,end) voice segments.
    """
    # try pyannote pipeline
    try:
        from pyannote.audio import Pipeline
        hf_token = os.environ.get(hf_token_env) or HF_TOKEN if 'HF_TOKEN' in globals() else None
        pipe = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=hf_token)
        diar = pipe(wav_path)
        vad_segments = []
        for speech in diar.get_timeline().support():
            vad_segments.append((speech.start, speech.end))
        log.info(f"[VAD-NEURAL] pyannote returned {len(vad_segments)} segments")
        if vad_segments:
            return vad_segments
    except Exception as e:
        log.info(f"[VAD-NEURAL] neural VAD failed/absent, falling back: {e}")

    # fallback: energy-based
    return run_energy_vad(wav_path, vad_mult=vad_mult)

def merge_close_replies(pairs, gap_threshold=MERGE_GAP):
    """
    Merge successive replies by same speaker when gap < gap_threshold (seconds).
    Expects each pair to have 'tstart' and 'tend'; if missing, those pairs stay separate.
    Returns merged list (keeps all detailed fields; updates output, output_len, tstart/tend, avg_score weighted).
    """
    if not pairs:
        return []
    # sort by tstart (None goes last)
    # pairs_sorted = sorted(pairs, key=lambda p: (p.get("tstart") is None, p.get("tstart") or 0.0))
    merged = []
    cur = pairs[0]
    # cur["output_len"] = cur.get("output_len", len(cur.get("output", "").split()))

    for p in pairs[1:]:
        # Merge only if same speaker and gap < threshold
        if (
            cur.get("output_speaker") == p.get("output_speaker")
            and cur.get("tend") is not None
            and p.get("tstart") is not None
            and (p["tstart"] - cur["tend"]) <= gap_threshold
        ):
            # Merge text outputs
            cur["output"] = (cur.get("output", "") + " " + p.get("output", "")).strip()
            cur["output_len"] = cur.get("output_len", 0) + p.get("output_len", 0)
            cur["tend"] = p.get("tend") or cur.get("tend")

            # Weighted avg_score if present
            if cur.get("avg_score") is not None and p.get("avg_score") is not None:
                cur["avg_score"] = (cur["avg_score"] + p["avg_score"]) / 2.0

            # ✅ Merge word-level lists too (critical for overlap checks)
            cur["words"] = (cur.get("words", []) or []) + (p.get("words", []) or [])
        else:
            merged.append(cur)
            cur = p

    merged.append(cur)
    log.info(
        f"[PAIRS-MERGE] Merged {len(pairs)} -> {len(merged)} pairs with gap_threshold={gap_threshold:.1f}s"
    )
    return merged

def make_pair(input_words, output_words, seg_idx=None, run_idx=None, target_label=None):
    """
    Turn lists of word dicts into a training pair dict.
    Enforces MAX_INPUT_WORDS by keeping the *last* MAX_INPUT_WORDS words
    from the input context (closest to the response).
    """
    # Build raw text
    input_text = " ".join((w.get("text") or w.get("word") or w.get("token") or "") for w in input_words).strip()
    output_text = " ".join((w.get("text") or w.get("word") or w.get("token") or "") for w in output_words).strip()

    # Compute lengths
    input_words_list = input_text.split()
    input_len = len(input_words_list)
    output_len = len(output_text.split())

    # Enforce MAX_INPUT_WORDS: keep the most recent words (end of context)
    try:
        max_in = int(MAX_INPUT_WORDS) if MAX_INPUT_WORDS is not None else None
    except Exception:
        max_in = None

    if max_in and input_len > max_in:
        # keep last max_in words
        input_words_list = input_words_list[-max_in:]
        input_text = " ".join(input_words_list)
        input_len = len(input_words_list)
        log.debug(f"[PAIR] Truncated input for seg_idx={seg_idx} to {input_len} words (MAX_INPUT_WORDS={max_in})")

    # tstart/tend from output words
    tstart = output_words[0].get("start") if output_words else None
    tend = output_words[-1].get("end") if output_words else None

    scores = [w.get("score") for w in output_words if isinstance(w.get("score"), (int, float))]
    avg_score = statistics.mean(scores) if scores else None

    # seg_idx: segment index; if run_idx is given, append it
    seg_id = f"{seg_idx}-{run_idx}" if run_idx is not None else seg_idx

    return {
        "seg_idx": seg_id,
        "input": input_text,
        "input_len": input_len,
        "output": output_text,
        "output_len": output_len,
        "output_speaker": target_label,
        "avg_score": avg_score,
        "tstart": tstart,
        "tend": tend,
        "words": output_words,
    }

def reassign_short_target_words(words, target_label, window=REASSIGN_WINDOW, audio_segment=None, target_emb=None, encoder=None):
    """
    Fix diarization mis-assignments for short words likely belonging to target speaker.
    If a candidate is borderline we optionally call contextual_reid_with_embeddings to verify.
    """
    corrected = []
    for i, w in enumerate(words):
        spk = w.get("speaker")
        if spk == target_label:
            corrected.append(w)
            continue

        word_text = (w.get("word") or w.get("text") or "").strip().lower()
        if len(word_text) <= 3:  # e.g. "yes", "no", "ok" 
            start = w.get("start", 0.0) or 0.0
            # Collect neighbors in window
            neighbors = [u for u in words if abs((u.get("start") or 0.0) - start) <= window]
            target_votes = sum(1 for u in neighbors if u.get("speaker") == target_label)
            if target_votes > len(neighbors) / 2:
                # if we have a target_emb available, do a quick embedding sanity check for borderline cases
                if target_emb is not None and audio_segment is not None:
                    sim, accept = contextual_reid_with_embeddings(
                        audio_segment, start, w.get("end", start + 0.05),
                        target_emb, encoder=encoder
                    )
                    # accept if similarity agrees (or if sim is very high)
                    if accept or sim > 0.82:
                        w["speaker"] = target_label
                else:
                    # fallback: majority vote
                    w["speaker"] = target_label
        corrected.append(w)
    return corrected

def build_pairs(segments, target_label, non_target_filter=None):
    """
    Build training pairs robustly by splitting segments into contiguous speaker runs
    (using per-word speaker labels), then applying merging, filtering, and shifting.
    Returns (pairs, dropped_non_target_count).
    """
    if non_target_filter is None:
        non_target_filter = NON_TARGET_FILTER

    pairs = []
    input_words = []
    output_words = []
    last_end = None
    dropped_non_target = 0
    leading_shifted = 0

    def split_into_runs(seg):
        words = seg.get("words", []) if isinstance(seg.get("words", []), list) else []
        runs, cur_run, cur_speaker = [], [], None
        for w in words:
            if not isinstance(w, dict):
                continue
            sp = w.get("speaker") or seg.get("speaker")
            if cur_speaker is None:
                cur_speaker, cur_run = sp, [w]
            elif sp == cur_speaker:
                cur_run.append(w)
            else:
                runs.append({
                    "speaker": cur_speaker,
                    "words": cur_run,
                    "start": cur_run[0].get("start"),
                    "end": cur_run[-1].get("end"),
                })
                cur_speaker, cur_run = sp, [w]
        if cur_run:
            runs.append({
                "speaker": cur_speaker,
                "words": cur_run,
                "start": cur_run[0].get("start"),
                "end": cur_run[-1].get("end"),
            })
        return runs

    for seg_idx, seg in enumerate(segments):
        runs = split_into_runs(seg)
        for run_idx, run in enumerate(runs):
            speaker, words = run.get("speaker"), run.get("words", [])
            start, end = run.get("start"), run.get("end")

            # Shift leading target words from non-target runs
            if speaker != target_label and words:
                leading_target = []
                while words and words[0].get("speaker") == target_label:
                    leading_target.append(words.pop(0))
                if leading_target:
                    output_words.extend(leading_target)
                    leading_shifted += len(leading_target)

            if speaker == target_label:
                if output_words and last_end is not None and start is not None and (start - last_end > MERGE_GAP):
                    cleaned_input, shifted_output = clean_input_words(input_words, target_label)
                    pairs.append(make_pair(cleaned_input,
                                           shifted_output + output_words,
                                           seg_idx=f"{seg_idx}-{run_idx}",
                                           target_label=target_label))
                    input_words, output_words = [], []
                output_words.extend(words)
                last_end = end
            else:
                if output_words:
                    cleaned_input, shifted_output = clean_input_words(input_words, target_label)
                    pairs.append(make_pair(cleaned_input,
                                           shifted_output + output_words,
                                           seg_idx=f"{seg_idx}-{run_idx}",
                                           target_label=target_label))
                    input_words, output_words = [], []
                for w in words:
                    txt = (w.get("text") or w.get("word") or w.get("token") or "").lower().strip()
                    if txt and txt in non_target_filter:
                        dropped_non_target += 1
                        continue
                    input_words.append(w)
                last_end = end

    if output_words:
        cleaned_input, shifted_output = clean_input_words(input_words, target_label)
        pairs.append(make_pair(cleaned_input,
                               shifted_output + output_words,
                               seg_idx=f"{seg_idx}-{run_idx}",
                               target_label=target_label))

    log.info(f"[FILTER] Dropped {dropped_non_target} non-target filler words using NON_TARGET_FILTER={list(non_target_filter)}")
    log.info(f"[BUILD_PAIRS] Leading target words shifted: {leading_shifted}")

    return pairs, dropped_non_target

def build_qna_pairs(assigned_data, target_speaker):
    """
    Builds training pairs from a single-speaker recording by identifying
    question-and-answer patterns in the monologue.
    """
    log.info("[PAIRS-QNA] Building pairs for single-speaker Q&A scenario.")
    
    words = []
    for seg in assigned_data.get("segments", []):
        words.extend(w for w in seg.get("words", []) if isinstance(w, dict))

    if not words:
        return []

    # Reconstruct sentences from words based on punctuation
    sentences = []
    current_sentence_words = []
    for word in words:
        word_text = (word.get("word") or word.get("text") or "").strip()
        current_sentence_words.append(word)
        if word_text.endswith(('.', '?', '!')):
            sentences.append(current_sentence_words)
            current_sentence_words = []
    if current_sentence_words:
        sentences.append(current_sentence_words)

    # Identify questions and build pairs (Question -> Subsequent non-questions)
    qna_pairs = []
    current_question_words = []
    current_answer_words = []
    in_answer = False

    interrogative_words = {"what", "who", "when", "where", "why", "how", "is", "are", "do", "does", "did", "can", "could", "will", "would", "should"}

    for sentence_words in sentences:
        sentence_text = " ".join((w.get("word") or w.get("text") or "") for w in sentence_words).strip()
        first_word = sentence_text.split(' ', 1)[0].lower().strip(".,?!")
        
        is_question = sentence_text.endswith('?') or first_word in interrogative_words

        if is_question:
            # If we were in an answer, the previous pair is complete
            if in_answer and current_question_words and current_answer_words:
                qna_pairs.append((current_question_words, current_answer_words))
            
            # Start a new question
            current_question_words = sentence_words
            current_answer_words = []
            in_answer = True # We are now looking for an answer
        elif in_answer:
            # If we are in an answer, append the current sentence
            current_answer_words.extend(sentence_words)

    # Add the last collected pair if it exists
    if current_question_words and current_answer_words:
        qna_pairs.append((current_question_words, current_answer_words))

    # Format into the standard pair structure
    final_pairs = []
    for i, (q_words, a_words) in enumerate(qna_pairs):
        input_text = " ".join((w.get("word") or w.get("text") or "") for w in q_words).strip()
        output_text = " ".join((w.get("word") or w.get("text") or "") for w in a_words).strip()
        
        scores = [w.get("score") for w in a_words if isinstance(w.get("score"), (int, float))]
        avg_score = statistics.mean(scores) if scores else None
        
        tstart = a_words[0].get("start") if a_words else None
        tend = a_words[-1].get("end") if a_words else None

        final_pairs.append({
            "seg_idx": f"qna-{i}",
            "input": input_text,
            "input_len": len(input_text.split()),
            "output": output_text,
            "output_len": len(output_text.split()),
            "output_speaker": target_speaker,
            "avg_score": avg_score,
            "tstart": tstart,
            "tend": tend,
            "words": a_words,
        })
        
    log.info(f"[PAIRS-QNA] Built {len(final_pairs)} Q&A pairs.")
    return final_pairs

def build_pairs_detailed(assigned_data, target_speaker):
    """
    Like build_pairs but returns detailed pairs:
    - seg_idx, input, output, output_len, avg_score, tstart, tend, words (list of dicts)
    """
    pairs = []
    context_buffer = []
    is_first_pair = True # Track if we are creating the first pair of the session

    for seg_idx, segment in enumerate(assigned_data.get("segments", [])):
        normalized = []
        for w in segment.get("words", []):
            if not isinstance(w, dict):
                continue
            txt = w.get("text") or w.get("word") or w.get("token")
            if txt is None:
                continue
            normalized.append({
                "text": txt.strip(),
                "speaker": w.get("speaker", "unknown"),
                "start": w.get("start"),
                "end": w.get("end"),
                "score": w.get("score")
            })
        if not normalized:
            continue

        if any(w["speaker"] == target_speaker for w in normalized):
            reply_words = [w for w in normalized if w["speaker"] == target_speaker]
            # When deciding whether to keep a reply (output), allow short outputs if avg confidence is high.
            reply_text = " ".join(w['text'] for w in reply_words if not w.get('overlapped'))
            reply_len = len(reply_text.split())
            avg_conf = statistics.mean([w.get('score',0.0) for w in reply_words]) if reply_words else 0.0

            # Keep if reply_len >= min_words OR (reply_len >= 1 and avg_conf >= 0.70)
            keep = (reply_len >= MIN_WORDS) or (reply_len >= 1 and avg_conf >= 0.70)
            if keep:
                starts = [w["start"] for w in reply_words if w.get("start") is not None]
                ends = [w["end"] for w in reply_words if w.get("end") is not None]
                tstart = min(starts) if starts else None
                tend = max(ends) if ends else None
                scores = [w["score"] for w in reply_words if isinstance(w.get("score"), (int, float))]
                avg_score = statistics.mean(scores) if scores else None
                # Build structured, turn-based context
                structured_context = []
                if context_buffer:
                    current_speaker = context_buffer[0]['speaker']
                    current_text = []
                    for word in context_buffer:
                        if word['speaker'] == current_speaker:
                            current_text.append(word['text'])
                        else:
                            if current_text:
                                structured_context.append(f"{current_speaker}: \"{ ' '.join(current_text) }\"")
                            current_speaker = word['speaker']
                            current_text = [word['text']]
                    if current_text:
                        structured_context.append(f"{current_speaker}: \"{ ' '.join(current_text) }\"")
                    context_text = " ".join(structured_context)
                else:
                    context_text = ""
                
                context_words = context_text.split()
                context_len = len(context_words)

                # Enforce MAX_INPUT_WORDS on the detailed builder as well
                try:
                    max_in = int(MAX_INPUT_WORDS) if MAX_INPUT_WORDS is not None else None
                except Exception:
                    max_in = None

                if max_in and context_len > max_in:
                    context_words = context_words[-max_in:]
                    context_text = " ".join(context_words)
                    context_len = len(context_words)
                    log.debug(f"[PAIRS-D] Truncated detailed context for seg_idx={seg_idx} to {context_len} words (MAX_INPUT_WORDS={max_in})")

                pair_data = {
                    "seg_idx": seg_idx,
                    "input": context_text,
                    "input_len": context_len,
                    "output": reply_text,
                    "output_len": reply_len,
                    "output_speaker": target_speaker,
                    "avg_score": avg_score,
                    "tstart": tstart,
                    "tend": tend,
                    "words": reply_words
                }

                # If this is the first pair and it has no preceding context, flag it as the introduction.
                if is_first_pair and context_len == 0:
                    pair_data["is_first_utterance"] = True
                
                pairs.append(pair_data)
                is_first_pair = False # Unset flag after processing the first potential pair

            # reset context
            context_buffer = []
        else:
            context_buffer.extend(normalized)

    log.info(f"[PAIRS-D] Built {len(pairs)} detailed pairs for {target_speaker}.")
    return pairs

def parse_diarization_json_to_segments(diarization_json_path="podcast.diarization.json"):
    """
    Tolerant parser for diarization JSON files (pyannote .to_json() or other shapes).
    Returns list of dicts: {'start':float,'end':float,'speaker':label}
    """

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    segments = []
    try:
        with open(diarization_json_path, "r", encoding="utf-8") as f:
            dj = json.load(f)
    except Exception:
        return segments

    # common pyannote shape: {'uri':..., 'content': [{'segment':{'start':..., 'end':...}, 'label': ...}, ...]}}
    if isinstance(dj, dict) and "content" in dj and isinstance(dj["content"], list):
        for item in dj["content"]:
            if isinstance(item, dict):
                seg = item.get("segment")
                label = item.get("label") or item.get("speaker")
                if isinstance(seg, dict):
                    s = safe_float(seg.get("start"))
                    e = safe_float(seg.get("end"))
                    if s is not None and e is not None and label is not None:
                        segments.append({"start": s, "end": e, "speaker": label})
    # fallback: an object with 'segments' list
    if not segments and isinstance(dj, dict) and "segments" in dj and isinstance(dj["segments"], list):
        for item in dj["segments"]:
            if isinstance(item, dict) and "segment" in item and isinstance(item["segment"], dict):
                seg = item["segment"]
                s = safe_float(seg.get("start"))
                e = safe_float(seg.get("end"))
                label = item.get("label") or item.get("speaker")
                if s is not None and e is not None and label is not None:
                    segments.append({"start": s, "end": e, "speaker": label})
    # deep traversal fallback: find nested segment dicts anywhere
    if not segments:
        def traverse(obj):
            if isinstance(obj, dict):
                if "segment" in obj and isinstance(obj["segment"], dict):
                    seg = obj["segment"]
                    label = obj.get("label") or obj.get("speaker")
                    s = safe_float(seg.get("start"))
                    e = safe_float(seg.get("end"))
                    if s is not None and e is not None and label is not None:
                        segments.append({"start": s, "end": e, "speaker": label})
                for v in obj.values():
                    traverse(v)
            elif isinstance(obj, list):
                for v in obj:
                    traverse(v)
        traverse(dj)

    # sort by start
    segments = sorted(segments, key=lambda x: x["start"])
    log.info(f"[DIAR-PARSE] Parsed {len(segments)} diarization segments from {diarization_json_path}")
    return segments

def auto_flag_low_quality_pairs(pairs_meta_path, flagged_out,
                                overlap_pair_threshold=OVERLAP_THRESHOLD, low_conf_threshold=LOW_CONFIDENCE_THRESHOLD, long_input_small_output_ratio=INPUT_TO_OUTPUT_RATIO):
    """
    Inspect pairs_with_overlap_meta.jsonl and create a flagged list of pairs that meet
    heuristic 'badness' tests. Writes flagged_out (jsonl) and returns list of flagged seg_idx.
    Heuristics:
      - pair overlap fraction (overlapped words / total words) > overlap_pair_threshold => high_overlap
      - avg_score is None or < low_conf_threshold => low_conf
      - input_len / max(1, output_len) > long_input_small_output_ratio => long_input_small_output
    """
    flagged = []
    flagged_ids = set()

    if not os.path.exists(pairs_meta_path):
        log.info("[FLAG] No pairs_meta file found (%s) — skipping auto-flag.", pairs_meta_path)
        return flagged_ids

    with open(pairs_meta_path, "r", encoding="utf-8") as f_in, open(flagged_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            p = json.loads(line)
            seg_idx = p.get("seg_idx")
            words = p.get("words", [])
            total = len(words)
            overlapped = sum(1 for w in words if w.get("overlapped"))
            overlap_rate = overlapped / total if total else 0.0
            reasons = []

            # high overlap
            if overlap_rate >= overlap_pair_threshold:
                reasons.append("high_overlap")

            # low confidence
            avg = p.get("avg_score")
            if avg is None or (avg < low_conf_threshold):
                reasons.append("low_conf")

            # input/output shape: long input with tiny output
            input_len = p.get("input_len") or (len(p.get("input","" ).split()) if p.get("input") else 0)
            output_len = p.get("output_len") or (len(p.get("output","" ).split()) if p.get("output") else 0)
            if output_len > 0 and (input_len / max(1, output_len) > long_input_small_output_ratio):
                reasons.append("long_input_small_output")

            if reasons:
                flagged_item = {
                    "seg_idx": seg_idx,
                    "input": p.get("input"),
                    "input_len": input_len,
                    "output": p.get("output"),
                    "output_len": output_len,
                    "overlap_rate": overlap_rate,
                    "avg_score": avg,
                    "reasons": reasons
                }
                f_out.write(json.dumps(flagged_item, ensure_ascii=False) + "\n")
                flagged.append(flagged_item)
                flagged_ids.add(seg_idx)

    log.info(f"[FLAG] Auto-flagged {len(flagged)} pairs to {flagged_out}.")
    return flagged_ids

def compute_overlap_and_write(pairs_detailed, diarization_object, overlap_word_frac, meta_out_path, clean_out_path, overlap_summary_out_path, model=None):
    """
    Tag per-word overlap fraction (relative to non-target speakers),
    write pairs_with_overlap_meta.jsonl and pairs_clean.jsonl (clean=drop overlapped words).
    """
    # Use the passed-in pyannote Annotation object directly
    diar_segments = [{"start": seg.start, "end": seg.end, "speaker": label} for seg, _, label in diarization_object.itertracks(yield_label=True)]

    # fast index is just the sorted list; we'll scan through it
    total_words = 0
    overlapped_words = 0
    pairs_meta = []
    pairs_clean = []

    def overlap_fraction(word_start, word_end, target_label):
        if word_start is None or word_end is None:
            return 0.0
        dur = word_end - word_start
        if dur <= 0:
            return 0.0
        ov = 0.0
        # sum overlap with diar segments that are not the target label
        for seg in diar_segments:
            if seg["end"] <= word_start:
                continue
            if seg["start"] >= word_end:
                break
            if seg["speaker"] != target_label:
                local_ov = max(0.0, min(word_end, seg["end"]) - max(word_start, seg["start"]))
                ov += local_ov
        return ov / dur if dur > 0 else 0.0

    for p in tqdm(pairs_detailed, desc="Analyzing Overlaps", leave=False):
        words = p.get("words", [])
        for w in words:
            w_start = w.get("start"); w_end = w.get("end")
            ov_frac = overlap_fraction(w_start, w_end, p.get("output_speaker"))
            w["overlap_frac"] = ov_frac
            # gather competing word scores overlapping the same timespan
            competing_scores = []
            for other_w in words:
                if other_w is w: continue
                if other_w.get("speaker") != p.get("output_speaker") and not (other_w.get("end",0) <= w_start or other_w.get("start",0) >= w_end):
                    if isinstance(other_w.get("score"), (int, float)):
                        competing_scores.append(other_w.get("score") )
            max_comp = max(competing_scores) if competing_scores else 0.0

            # dominance rule: only mark overlapped if overlap_frac > threshold AND we are not dominant
            if ov_frac > overlap_word_frac:
                # allow keeping slightly-lower-scored words if they beat competing speaker by margin_delta
                margin_delta = 0.05
                if (w.get("score", 0.0) + margin_delta) >= max_comp:
                    w["overlapped"] = False
                else:
                    w["overlapped"] = True
            else:
                w["overlapped"] = False

        total_words += len(words)
        overlapped_words += sum(1 for w in words if w.get("overlapped"))

        # compute quality score and attach
        try:
            qscore, qmetrics = score_pair_quality(p, model=model)
            p["quality_score"] = qscore
            p["quality_metrics"] = qmetrics
        except Exception as e:
            log.debug(f"[SCORE] Failed scoring pair {p.get('seg_idx')}: {e}")
            p["quality_score"] = 0.0
            p["quality_metrics"] = {}

        pairs_meta.append(p)

        # build clean pair (wrap overlapped words in tags instead of dropping)
        clean_tokens = []
        for w in words:
            text = (w.get("text") or w.get("word") or "").strip()
            if w.get("overlapped"):
                clean_tokens.append(f"<overlap>{text}</overlap>")
            else:
                clean_tokens.append(text)
        
        clean_text = " ".join(clean_tokens).strip()
        
        # The clean pair still gets all original words for scoring purposes
        if words:
            scores = [w.get("score") for w in words if isinstance(w.get("score"), (int, float))]
            avg_score = statistics.mean(scores) if scores else p.get("avg_score")
            pairs_clean.append({
                "seg_idx": p.get("seg_idx"),
                "input": p.get("input"),
                "input_len": p.get("input_len"),
                "output": clean_text,
                "output_len": len(clean_text.split()),
                "output_speaker": p.get("output_speaker"),
                "avg_score": avg_score,
                "tstart": p.get("tstart"),
                "tend": p.get("tend"),
                "quality_score": p.get("quality_score")
            })

    # write files
    with open(meta_out_path, "w", encoding="utf-8") as f:
        for pm in pairs_meta:
            f.write(json.dumps(pm, ensure_ascii=False) + "\n")
    with open(clean_out_path, "w", encoding="utf-8") as f:
        for pc in pairs_clean:
            f.write(json.dumps(pc, ensure_ascii=False) + "\n")

    overlap_rate = overlapped_words / total_words if total_words > 0 else 0.0
    summary = {
        "pairs_meta_count": len(pairs_meta),
        "pairs_clean_count": len(pairs_clean),
        "total_words": total_words,
        "overlapped_words": overlapped_words,
        "overlap_rate": overlap_rate,
        "clean_drop_rate": 1 - (len(pairs_clean) / len(pairs_meta) if pairs_meta else 1)
    }
    with open(overlap_summary_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"[OVERLAP] Wrote pairs_with_overlap_meta.jsonl ({len(pairs_meta)}) and pairs_clean.jsonl ({len(pairs_clean)}).")
    log.info(f"[OVERLAP] Summary: {summary}")
    return summary


def score_pair_quality(pair, model, weights=None, punc_bonus=0.1, ratio_floor=0.3, ratio_ceiling=4.0):
    """
    Compute a soft quality score for a single pair_meta dict.
    Returns (score, metrics_dict).
    Metrics included: avg_conf, overlap_rate, input_len, output_len, ratio, punc_ok, speaker_diversity, semantic_similarity.
    Weight defaults tuned conservatively.
    """
    # This function now uses the global WEIGHT_* constants
    
    words = pair.get("words", [])
    # avg confidence (use any score/confidence in words)
    confs = [w.get("score") for w in words if isinstance(w.get("score"), (int, float))]
    avg_conf = float(statistics.mean(confs)) if confs else float(pair.get("avg_score") or 0.0)

    # overlap: fraction of words flagged overlapped
    overlap_rate = sum(1 for w in words if w.get("overlapped")) / len(words) if words else 0.0
    overlap_score = 1 - overlap_rate

    # input / output lengths
    input_len = int(pair.get("input_len") or 0)
    output_len = int(pair.get("output_len") or 0)
    
    # Ratio score
    ratio = (input_len / output_len) if output_len > 0 else 0
    ratio_score = 1 / (1 + ratio)

    input_len_score = min(input_len / MAX_INPUT_WORDS, 1.0)
    output_len_score = 1 / (1 + math.exp(-(output_len - 5)))


    # Granular punctuation score based on sentence completion
    output_text = (pair.get("output") or "").strip()
    if not output_text:
        punc_score = 0.0
    else:
        sentences = sent_tokenize(output_text)
        if not sentences:
            punc_score = 0.0
        else:
            punctuated_sentences = sum(1 for s in sentences if s.strip().endswith(('.', '?', '!')))
            punc_score = punctuated_sentences / len(sentences)

    # speaker diversity of input: prefer at least 2 input speakers (normalized)
    input_speakers = set()
    for w in words:
        if w.get("speaker") and w.get("speaker") != pair.get("output_speaker"):
            input_speakers.add(w.get("speaker") )
    diversity_score = min(1.0, len(input_speakers) / 2.0)

    # Semantic Similarity
    input_text = pair.get("input", "")
    semantic_similarity = 0.0
    if model is not None and input_text and output_text:
        embeddings = model.encode([input_text, output_text])
        semantic_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Combine scores using a weighted average.
    total_weight = (WEIGHT_AVG_CONF + WEIGHT_OVERLAP + WEIGHT_RATIO + WEIGHT_SIMILARITY + 
                    WEIGHT_PUNC + WEIGHT_DIVERSITY + WEIGHT_INPUT_LEN + WEIGHT_OUTPUT_LEN)
    
    # Handle potential division by zero if all weights are zero
    if total_weight == 0:
        score = 0.0
    else:
        score = (
            avg_conf * WEIGHT_AVG_CONF +
            overlap_score * WEIGHT_OVERLAP +
            ratio_score * WEIGHT_RATIO +
            semantic_similarity * WEIGHT_SIMILARITY +
            punc_score * WEIGHT_PUNC +
            diversity_score * WEIGHT_DIVERSITY +
            input_len_score * WEIGHT_INPUT_LEN +
            output_len_score * WEIGHT_OUTPUT_LEN
        ) / total_weight


    # clamp to [0,1]
    score = max(0.0, min(1.0, score))

    metrics = {
        "avg_conf": avg_conf,
        "overlap_rate": overlap_rate,
        "input_len": input_len,
        "output_len": output_len,
        "ratio": ratio,
        "ratio_score": ratio_score,
        "punc_score": punc_score,
        "diversity_score": diversity_score,
        "semantic_similarity": semantic_similarity
    }

    return float(score), metrics

def is_semantically_rich(text):
    """
    Returns True if text has at least 1 unique non-stopword tokens.
    """
    tokens = re.findall(r"\w+", text.lower())
    content = [t for t in tokens if t not in STOPWORDS]
    return len(set(content)) >= 1

def postfilter_clean_pairs(clean_pairs, min_words=MIN_WORDS, min_conf=MIN_CONF, min_quality_score=MIN_QUALITY_SCORE):
    """
    Consolidated final filter.
    - Filters by length, keeping high-confidence short replies.
    - Filters by overall confidence.
    - Filters by a minimum quality score.
    Returns the final list of pairs and a summary dictionary.
    """
    final_pairs = []
    dropped_len, dropped_conf, dropped_quality = 0, 0, 0
    short_reply_conf_threshold = SHORT_REPLY_MIN_CONF

    for p in tqdm(clean_pairs, desc="Applying final filters", leave=False):
        output_len = p.get("output_len", 0)
        avg_score = p.get("avg_score")
        quality_score = p.get("quality_score")

        # 1. Length and High-Confidence Short Reply Check
        keep_by_len = (output_len >= min_words) or \
                      (output_len >= 1 and avg_score is not None and avg_score >= short_reply_conf_threshold)
        if not keep_by_len:
            dropped_len += 1
            continue
        
        # 2. General Confidence Check
        if (min_conf is not None) and (avg_score is not None) and (avg_score < min_conf):
            dropped_conf += 1
            continue
            
        # 3. Quality Score Check
        if quality_score is None or quality_score < min_quality_score:
            dropped_quality += 1
            continue
        
        final_pairs.append(p)

    summary = {
        "params": {"min_words": min_words, "min_conf": min_conf, "short_reply_conf": SHORT_REPLY_MIN_CONF, "min_quality_score": min_quality_score},
        "counts": {
            "before": len(clean_pairs), 
            "after": len(final_pairs), 
            "dropped_by_length": dropped_len, 
            "dropped_by_conf": dropped_conf,
            "dropped_by_quality": dropped_quality
        }
    }
    log.info(f"[POSTFILTER] Filtering complete. Kept {len(final_pairs)}/{len(clean_pairs)} pairs.")
    return final_pairs, summary


def identify_speaker(samples_folder, averaged_speaker_embeddings, threshold=IDENTIFY_THRESHOLD):
    """
    Robust speaker identification:
      - samples_folder: folder of reference audio (one or more files)
      - audio_file: full podcast path (used for extraction via pydub)
      - diarization: pyannote Annotation or fallback list of segments
      - threshold: similarity threshold (0..1), where higher is more strict

    Returns: (best_speaker_label, best_similarity) OR (None, None) if not found.
    """
        # Load reference embedding (guarded)
    try:
        _, target_avg_emb = load_reference_embeddings(samples_folder)
    except Exception as e:
        log.error(f"[IDENTIFY] Could not load reference embeddings: {e}")
        return None, None

    # Use the pre-computed embeddings passed into the function
    avg_embs = averaged_speaker_embeddings

    if not avg_embs:
        log.warning("[IDENTIFY] Averaging per-speaker embeddings failed or produced no results.")
        return None, None

    # similarity: 1 - cosine_distance  (range ~ -1..1 but for normalized embeddings ~0..1)
    best_speaker = None
    best_similarity = -999.0
    for sp, emb in avg_embs.items():
        try:
            dist = cosine(emb, target_avg_emb)  # smaller distance = closer
            similarity = 1.0 - dist
        except Exception as e:
            log.debug(f"[IDENTIFY] cosine failed for speaker {sp}: {e}")
            continue
        log.info(f"[IDENTIFY] speaker {sp} similarity={similarity:.4f} (dist={dist:.4f})")
        if similarity > best_similarity:
            best_similarity = similarity
            best_speaker = sp

    # threshold check: require some minimum similarity
    if best_speaker is None or (best_similarity < threshold):
        log.warning(f"[IDENTIFY] Best speaker {best_speaker} has similarity {best_similarity:.4f} < threshold {threshold}. Rejecting.")
        return None, None

    log.info(f"[IDENTIFY] Selected target speaker {best_speaker} (similarity={best_similarity:.4f})")
    return best_speaker, float(best_similarity)

def transcribe_with_vad(model, audio_path, wav_path, language=LANGUAGE, batch_size=WHISPER_BATCH_SIZE,
                        condition_on_previous_text=False):
    """
    Split the audio into non-silent chunks using the pipeline's neural VAD, 
    transcribe each chunk, then stitch the segments back together into a 
    single Whisper-style result dict with adjusted segment timestamps.
    """
    log.info("[TRANSCRIBE-VAD] Using neural VAD for audio segmentation...")

    # Use the pipeline's superior neural VAD with fallback to get speech segments
    # This is more robust than the previous energy-based pydub method.
    if not os.path.exists(wav_path):
        log.error(f"[TRANSCRIBE-VAD] WAV file not found at {wav_path}. Cannot perform VAD. Please ensure it's created before this step.")
        # Fallback to full-file transcription if WAV is missing
        return model.transcribe(audio_path, batch_size=batch_size, language=language)
        
    nonsilent_intervals = run_vad_with_fallback(wav_path, vad_mult=CFG.get("VAD_MULT", 0.3))
    
    # Convert segments from seconds to milliseconds for pydub
    nonsilent_intervals_ms = [(int(start * 1000), int(end * 1000)) for start, end in nonsilent_intervals]

    audio = AudioSegment.from_file(audio_path)

    # If VAD found nothing, fall back to full-file transcription.
    if not nonsilent_intervals_ms:
        log.warning("[TRANSCRIBE-VAD] VAD returned no voiced intervals — falling back to full-file transcription.")
        return model.transcribe(audio_path, batch_size=batch_size, language=language)

    merged_result = {"language": None, "segments": []} # Initialize merged_result
    
    with tempfile.TemporaryDirectory() as td:
        for i, (start_ms, end_ms) in enumerate(tqdm(nonsilent_intervals_ms, desc="Transcribing segments", leave=False)):
            # Export segment
            seg = audio[start_ms:end_ms]
            tmp_path = os.path.join(td, f"vad_seg_{i}.wav")
            seg.export(tmp_path, format="wav")

            # Transcribe this segment
            try:
                # remove unsupported kwargs when calling model.transcribe
                res = model.transcribe(tmp_path, batch_size=batch_size, language=language)
            except Exception as e:
                log.warning(f"[TRANSCRIBE-VAD] Transcription failed for segment {i} [{start_ms}..{end_ms}]: {e}")
                continue

            # set language if not set
            if merged_result["language"] is None:
                merged_result["language"] = res.get("language")

            # adjust timings of each returned segment by start_ms/1000
            offset = start_ms / 1000.0
            for seg_item in res.get("segments", []):
                seg_adj = seg_item.copy()
                seg_adj["start"] = (seg_item.get("start", 0.0) or 0.0) + offset
                seg_adj["end"] = (seg_item.get("end", 0.0) or 0.0) + offset
                merged_result["segments"].append(seg_adj)

    if "language" not in merged_result or merged_result["language"] is None:
        log.warning("[LANG] Transcript missing language — forcing 'en'")
        merged_result["language"] = "en"

    # sort segments by start time to be safe
    merged_result["segments"] = sorted(merged_result["segments"], key=lambda s: s.get("start", 0.0))
    log.info("[TRANSCRIBE-VAD] Transcribed %d voiced intervals into %d segments.",
                 len(nonsilent_intervals_ms), len(merged_result["segments"]))

    return merged_result

def contextual_reid_with_embeddings(audio_segment, word_start, word_end, target_emb, encoder=None, pad=0.05, similarity_threshold=CONTEXTUAL_REID_THRESHOLD):
    """
    Embed a short time span by exporting a tiny wav file and comparing embeddings.
    Returns (similarity, accept_bool).
    """
    try:
        enc = encoder if encoder is not None else VoiceEncoder()
        start = max(0.0, (word_start or 0.0) - pad)
        end = max(start + 0.01, float(word_end or start) + pad)
        duration = end - start

        # Export small slice to temporary wav so preprocess_wav gets a filepath
        with tempfile.TemporaryDirectory() as td:
            audio = audio_segment
            if audio is None:
                log.error("[REID] audio_segment object not provided to contextual_reid_with_embeddings.")
                return 0.0, False
            s_ms = int(start * 1000); e_ms = int(end * 1000)
            tmp_path = os.path.join(td, "ctx_reid.wav")
            audio[s_ms:e_ms].export(tmp_path, format="wav")
            emb = _embed_audio_segment(tmp_path, enc)
            if emb is None:
                return 0.0, False
            # cosine similarity (normalized embeddings expected)
            sim = 1.0 - cosine(emb, target_emb) if target_emb is not None else 0.0
            accept = sim >= similarity_threshold
            return float(sim), bool(accept)
    except Exception as e:
        log.debug(f"[REID] contextual reid failed: {e}")
        return 0.0, False
    
def cleanup_redundant_files(base_name):
    """Remove temporary files that are less useful for debugging to reduce clutter."""
    if not KEEP_DEBUG_OUTPUTS:
        log.info("[CLEANUP] Skipping cleanup since KEEP_DEBUG_OUTPUTS is False.")
        return

    log.info("[CLEANUP] Removing redundant intermediate cache files...")
    files_to_delete = [
        os.path.join(CACHE_FOLDER, f"{base_name}_pairs_merged.jsonl"),
        os.path.join(CACHE_FOLDER, f"{base_name}_pairs_clean.jsonl"),
    ]
    for f_path in files_to_delete:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                log.info(f"[CLEANUP]   - Deleted: {os.path.basename(f_path)}")
            except OSError as e:
                log.warning(f"[CLEANUP]   - Failed to delete {os.path.basename(f_path)}: {e}")

# ==================================================================
# 🚀 Main execution block
# ==================================================================
def run_pipeline(input_audio_path, models):
    """
    Main processing function for a single audio file.
    Accepts pre-loaded models to avoid reloading in a batch process.
    """
    # --- Dynamic Path Generation ---
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    log.info(f"--- Starting pipeline for: {base_name} ---")

    # Define all paths dynamically, placing them in the correct folders
    wav_path = os.path.join(CACHE_FOLDER, f"{base_name}.wav")
    transcript_cache = os.path.join(CACHE_FOLDER, f"{base_name}.transcript.json")
    alignment_cache = os.path.join(CACHE_FOLDER, f"{base_name}.aligned.json")
    diar_cache = os.path.join(CACHE_FOLDER, f"{base_name}.diarization.json")
    assign_cache = os.path.join(CACHE_FOLDER, f"{base_name}.assigned.json")
    
    # Define DYNAMIC output filenames to prevent overwriting
    PAIRS_MERGED_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_pairs_merged.jsonl")
    PAIRS_META_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_pairs_with_overlap_meta.jsonl")
    PAIRS_CLEAN_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_pairs_clean.jsonl")
    PAIRS_FLAGGED_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_pairs_flagged.jsonl")
    PAIRS_TMP_PREFIX_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_pairs_tmp")
    POSTFILTER_SUMMARY_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_pairs_postfilter_summary.json")
    OVERLAP_SUMMARY_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_overlap_phaseB_summary.json")
    METRICS_CSV_OUT = os.path.join(CACHE_FOLDER, f"{base_name}_pair_quality_metrics.csv")
    FINAL_OUTPUT = os.path.join(RESULTS_FOLDER, f"{base_name}_results.jsonl")
    
    start_time = time.time()

    # ------------------------------
    # STEP 1: Transcribe audio → text with WhisperX
    # ------------------------------
    log.info("[STEP] 1 ... Transcribe audio -> text with WhisperX")

    # Ensure we have a WAV copy for VAD and libraries needing PCM
    audio_duration_seconds = 0
    if not os.path.exists(wav_path):
        try:
            audio = AudioSegment.from_file(input_audio_path)
            audio.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")
            audio_duration_seconds = len(audio) / 1000.0
            log.info(f"[AUDIO] Exported WAV copy for VAD: {wav_path}")
        except Exception as e:
            log.warning(f"[AUDIO] Could not export WAV copy: {e}")
    else:
        try:
            audio_duration_seconds = len(AudioSegment.from_file(wav_path)) / 1000.0
        except Exception:
            pass # Will be logged later if it fails

    result = None  # Initialize result to be safe
    if os.path.exists(transcript_cache):
        log.info(f"[CACHE] Loading cached transcript from {transcript_cache}...")
        with open(transcript_cache, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        log.info("[TRANSCRIBE] Running WhisperX transcription...")
        if USE_VAD:
            result = transcribe_with_vad(
                models['whisper'],
                input_audio_path,
                wav_path=wav_path,
                batch_size=WHISPER_BATCH_SIZE,
            )
        else:
            result = models['whisper'].transcribe(
                input_audio_path,
                batch_size=WHISPER_BATCH_SIZE,
                language="en"
            )
        
        # Only save to cache if transcription was successful
        if result and result.get("segments"):
            with open(transcript_cache, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # After attempting to load or create, check if the result is valid
    if not result or not result.get("segments"):
        log.error(f"Transcription failed or produced no segments for {base_name}. Skipping this file.")
        return -1  # Return error code

    if not result.get("language"):
        log.warning("[LANG] Transcript missing language — forcing 'en'")
        result["language"] = "en"

    start_time = mark_step(f"Transcription for {base_name}", start_time)

    # ------------------------------
    # STEP 2: Align transcript to audio (word-level timestamps)
    # ------------------------------
    log.info("[STEP] 2 ... Align transcript to audio")
    if os.path.exists(alignment_cache):
        log.info(f"[CACHE] Loading cached alignment from {alignment_cache}...")
        with open(alignment_cache, "r", encoding="utf-8") as f:
            result_aligned = json.load(f)
    else:
        log.info("[ALIGN] Running WhisperX alignment...")
        with tqdm(total=1, desc="Aligning transcript") as pbar:
            result_aligned = whisperx.align(result["segments"], models['align'], models['align_meta'], input_audio_path, device)
            pbar.update(1)
        with open(alignment_cache, "w", encoding="utf-8") as f:
            json.dump(result_aligned, f, ensure_ascii=False, indent=2)

    start_time = mark_step(f"Alignment for {base_name}", start_time)

    # ------------------------------
    # STEP 3: Speaker diarization (who spoke when)
    # ------------------------------
    log.info("[STEP] 3 ... Speaker diarization (who spoke when)")
    if os.path.exists(diar_cache):
        log.info(f"[CACHE] Loading cached diarization from {diar_cache}...")
        with open(diar_cache, "r", encoding="utf-8") as f:
            diarization_json = json.load(f)

        diarization = Annotation(uri=diarization_json.get("uri", base_name))
        for entry in diarization_json.get("content", []):
            seg = entry["segment"]
            diarization[Segment(seg["start"], seg["end"])] = entry["label"]
    else:
        log.info("[CACHE] No cached diarization found, generating anew...")
        try:
            with tqdm(total=1, desc="Diarizing speakers") as pbar:
                diarization_df = models['diarize'](input_audio_path)
                pbar.update(1)
        except Exception as e:
            log.error(f"[DIARIZE] Diarization failed. This is often due to an invalid Hugging Face token or not accepting the model's terms of use.")
            log.error(f"[DIARIZE] Please ensure you have a valid token and have accepted the terms at:")
            log.error(f"[DIARIZE] https://huggingface.co/pyannote/speaker-diarization")
            log.error(f"[DIARIZE] https://huggingface.co/pyannote/segmentation")
            log.error(f"[DIARIZE] Original error: {e}", exc_info=True)
            raise

        diarization = Annotation(uri=base_name)
        for _, row in diarization_df.iterrows():
            diarization[Segment(row["start"], row["end"])] = row["speaker"]

        diarization_dict = {
            "uri": base_name,
            "content": [
                {"segment": {"start": row["start"], "end": row["end"]}, "label": row["speaker"]} 
                for _, row in diarization_df.iterrows()
            ],
        }
        with open(diar_cache, "w", encoding="utf-8") as f:
            json.dump(diarization_dict, f, ensure_ascii=False, indent=2)
        log.info(f"[CACHE] Saved new diarization to {diar_cache}")
    
    if not isinstance(diarization, Annotation):
        raise TypeError(f"Expected pyannote.core.Annotation, got {type(diarization)}")
    start_time = mark_step(f"Diarization for {base_name}", start_time)

    # ------------------------------
    # STEP 4: Assign speakers to transcript words
    # ------------------------------
    log.info("[STEP] 4 ... Assign speakers to transcript words")
    if os.path.exists(assign_cache):
        log.info(f"[CACHE] Loading cached speaker assignment from {assign_cache}...")
        with open(assign_cache, "r", encoding="utf-8") as f:
            result_aligned = json.load(f)
    else:
        log.info("[ASSIGN] Running safe speaker assignment...")
        result_aligned = safe_assign_word_speakers(diarization, result_aligned, diar_cache_path=diar_cache, base_name=base_name)
        
        with open(assign_cache, "w", encoding="utf-8") as f:
            json.dump(result_aligned, f, ensure_ascii=False, indent=2)

    start_time = mark_step(f"Speaker Assignment for {base_name}", start_time)

    # ------------------------------
    # STEP 5: Build training pairs
    # ------------------------------
    log.info("[STEP] 5 ... Build training pairs")
    log.info("[AUDIO] Loading main audio segment for processing...")
    try:
        main_audio_segment = AudioSegment.from_file(wav_path)
    except Exception as e:
        log.error(f"[AUDIO] Could not load main audio segment from {wav_path}: {e}")
        main_audio_segment = None
    
    # First, compute all speaker embeddings once. This is now the single source of truth.
    speaker_embs = compute_speaker_embeddings(wav_path, diarization)
    
    # Now, identify the target by comparing against the pre-computed embeddings.
    best_speaker, best_score = identify_speaker("samples", speaker_embs)

    if not best_speaker:
        log.error(f"[PIPELINE] No reference speaker identified for {base_name}. Skipping pair generation.")
        return -1

    # Load assigned segments
    with open(assign_cache, "r", encoding="utf-8") as f:
        assigned_data = json.load(f)
    assigned_segments = assigned_data.get("segments", [])

    # --- Speaker Dominance Check for Single-Speaker Scenario ---
    speaker_word_counts = defaultdict(int)
    total_word_count = 0
    for seg in assigned_data.get("segments", []):
        for w in seg.get("words", []):
            speaker = w.get("speaker")
            if speaker:
                speaker_word_counts[speaker] += 1
                total_word_count += 1

    is_single_speaker_scenario = False
    if best_speaker and total_word_count > 0:
        target_speaker_words = speaker_word_counts.get(best_speaker, 0)
        # If only one speaker was diarized OR the target speaks > 98% of the words
        if len(speaker_word_counts) == 1 or (target_speaker_words / total_word_count) > 0.98:
            is_single_speaker_scenario = True
            log.info(f"[PIPELINE] Single-speaker Q&A scenario detected for {base_name}.")

    # --- Scenario-based Pair Building ---
    if is_single_speaker_scenario:
        # For Q&A, we build pairs and then run a simplified finalization sequence
        qna_pairs = build_qna_pairs(assigned_data, best_speaker) 
        
        for p in qna_pairs:
            score, metrics = score_pair_quality(p, model=models['sentence_transformer'])
            p["quality_score"] = score
            p["quality_metrics"] = metrics

        final_pairs, _ = postfilter_clean_pairs(
            qna_pairs, min_words=MIN_WORDS, min_conf=MIN_CONF, min_quality_score=MIN_QUALITY_SCORE
        )
        
        log.info(f"[PIPELINE] Final output for {base_name} written to '{FINAL_OUTPUT}' ({len(final_pairs)} kept).")
        with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
            for p in final_pairs:
                # Q&A pairs should not have empty inputs, but handle defensively
                if not p.get("input", "").strip():
                    p["input"] = "[MISSING QUESTION]"
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        
        cleanup_redundant_files(base_name)
        return 0 # End processing for this file

    # --- Normalize words across all segments (Standard Multi-Speaker Path) ---
    flat_words = []
    for seg in assigned_segments:
        for w in seg.get("words", []):
            if "text" not in w:
                w["text"] = w.get("word") or w.get("token") or ""
            if w.get("score") is None:
                w["score"] = w.get("confidence")
            try:
                w["start"] = float(w["start"]) if w.get("start") is not None else None
            except Exception:
                w["start"] = None
            try:
                w["end"] = float(w["end"]) if w.get("end") is not None else None
            except Exception:
                w["end"] = None
            flat_words.append(w)

    # Adaptive confidence cutoff
    words_conf, removed_conf, cutoffs = adaptive_conf_filter(flat_words)
    log.info(f"[ADAPTIVE] Removed {len(removed_conf)} words below cutoffs: {cutoffs}")

    # Run neural VAD with fallback
    vad_segments = run_vad_with_fallback(wav_path, vad_mult=CFG.get("VAD_MULT", 1.0))

    # Keep only words that overlap speech according to VAD
    words_vad = [w for w in words_conf if overlaps_vad(w, vad_segments)]
    log.info(f"[VAD] Kept {len(words_vad)} words after speech activity check")

    # Compute speaker embeddings for contextual re-ID (optional, potentially slow)
    try:
        speaker_embs = compute_speaker_embeddings(wav_path, diarization)
        target_emb = speaker_embs.get(best_speaker)
    except Exception as e:
        log.warning(f"[EMBED] Could not compute speaker embeddings for re-ID: {e}")
        target_emb = None

    # Reassignment correction (short target words mis-assigned). 
    words_reassigned = reassign_short_target_words(words_vad, best_speaker, window=1.0,
                                                   audio_segment=main_audio_segment, target_emb=target_emb)
    log.info(f"[REASSIGN] Applied correction for short words near target speaker (reassigned -> {len([w for w in words_reassigned if w.get('speaker')==best_speaker])} target words)")

    # Rebuild segments from filtered + reassigned words
    segments_final = regroup_words_to_segments(words_reassigned)
    
    # To get the `dropped_non_target` count, we can do a quick pass on the flat word list
    dropped_non_target = 0
    for w in words_reassigned: # This is the final, cleaned list of words
        if w.get('speaker') != best_speaker:
            txt = (w.get("text") or "").lower().strip()
            if txt and txt in NON_TARGET_FILTER:
                dropped_non_target += 1

    # Build detailed pairs
    pairs_detailed = build_pairs_detailed(assigned_data, best_speaker)
    target_word_count = sum(
        1 for seg in assigned_data.get("segments", [])
        for w in seg.get("words", [])
        if w.get("speaker") == best_speaker
    )
    # Phase A: Merge
    pairs_merged = merge_close_replies(pairs_detailed, gap_threshold=MERGE_GAP)
    with open(PAIRS_MERGED_OUT, "w", encoding="utf-8") as f:
        for p in pairs_merged:
            f.write(json.dumps({
                "input": p.get("input", ""),
                "output": p.get("output", ""),
                "output_speaker": p.get("output_speaker")
            }, ensure_ascii=False) + "\n")

    # Phase B: Overlap filtering
    overlap_summary = compute_overlap_and_write(
        pairs_merged,
        diarization_object=diarization,
        overlap_word_frac=OVERLAP_FRAC,
        meta_out_path=PAIRS_META_OUT,
        clean_out_path=PAIRS_CLEAN_OUT,
        overlap_summary_out_path=OVERLAP_SUMMARY_OUT,
        model=models['sentence_transformer']
    )

    # Auto-flag suspicious pairs for manual review
    flagged_ids = auto_flag_low_quality_pairs(
        pairs_meta_path=PAIRS_META_OUT,
        flagged_out=PAIRS_FLAGGED_OUT,
        overlap_pair_threshold=OVERLAP_THRESHOLD,
        low_conf_threshold=LOW_CONFIDENCE_THRESHOLD,
        long_input_small_output_ratio=INPUT_TO_OUTPUT_RATIO
    )
    log.info(f"[FLAG] {len(flagged_ids)} pairs flagged for manual review ({base_name}_pairs_flagged.jsonl).")

    # Phase C: Post-filter
    clean_pairs_from_file = []
    try:
        with open(PAIRS_CLEAN_OUT, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                clean_pairs_from_file.append(json.loads(line))
    except Exception:
        log.warning(f"[POSTFILTER] {PAIRS_CLEAN_OUT} not found — skipping postfilter.")
    
    # Run the new, consolidated filtering function to get the final list directly.
    final_pairs, post_summary = postfilter_clean_pairs(
        clean_pairs_from_file,
        min_words=MIN_WORDS,
        min_conf=MIN_CONF,
        min_quality_score=MIN_QUALITY_SCORE
    )

    # The list is now final. Proceed directly to scoring and saving.

    # Re-score the final, post-filtered pairs using the best quality metric
    for p in final_pairs:
        # We need to re-read the detailed "meta" pairs to get the word-level data for scoring
        # This is a bit inefficient, but ensures we use the best score.
        # A more advanced refactor could pass this data through the pipeline.
        score, metrics = score_pair_quality(p, model=models['sentence_transformer'])
        p["quality_score"] = score
        p["quality_metrics"] = metrics

    # Now, filter based on the new, more accurate score
    clean_pairs = [p for p in final_pairs if p.get("quality_score", 0.0) >= 0.7]
    borderline_pairs = [p for p in final_pairs if 0.4 <= p.get("quality_score", 0.0) < 0.7]
    flagged_pairs = [p for p in final_pairs if p.get("quality_score", 0.0) < 0.4]

    log.info(f"[QUALITY] Clean={len(clean_pairs)}, Borderline={len(borderline_pairs)}, Flagged={len(flagged_pairs)}")
    
    # Write the detailed, accurate metrics to the CSV
    with open(METRICS_CSV_OUT, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["seg_idx", "quality_score"] + list(final_pairs[0].get("quality_metrics", {}).keys()) if final_pairs else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for p in final_pairs:
            row = {"seg_idx": p.get("seg_idx"), "quality_score": p.get("quality_score")}
            row.update(p.get("quality_metrics", {}))
            writer.writerow(row)

    def _is_empty_input(pair):
        inp = pair.get("input")
        return inp is None or (isinstance(inp, str) and not inp.strip())

    original_empty_input_count = sum(1 for p in final_pairs if _is_empty_input(p))
    original_empty_input_pct = (original_empty_input_count / len(final_pairs) * 100) if final_pairs else 0.0

    for p in final_pairs:
        if _is_empty_input(p):
            if p.get("is_first_utterance"):
                p["input"] = "[NO PROMPT]"
            else:
                p["input"] = "[CONTEXT CONTINUES]"

    final_output_path = FINAL_OUTPUT
    with open(final_output_path, "w", encoding="utf-8") as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    flagged_count = len(flagged_ids)

    tmp_summary_path = f"{PAIRS_TMP_PREFIX_OUT}_postfilter_summary.json"
    if os.path.exists(tmp_summary_path):
        try:
            shutil.move(tmp_summary_path, POSTFILTER_SUMMARY_OUT)
        except Exception:
            log.warning("[PIPELINE] Could not rename temp postfilter summary")

    log.info(f"[PIPELINE] Replaced {original_empty_input_count} empty inputs with [CONTEXT CONTINUES].")
    log.info(f"[PIPELINE] Final output for {base_name} written to '{os.path.join(RESULTS_FOLDER, base_name)}_results.jsonl' ({len(final_pairs)} kept).")

    # --- Calculate final metrics for summary ---
    target_speech_duration = sum(w['end'] - w['start'] for w in flat_words if w.get('speaker') == best_speaker and w.get('start') is not None and w.get('end') is not None)
    final_avg_conf = statistics.mean([p['avg_score'] for p in final_pairs if p.get('avg_score') is not None]) if final_pairs else 0.0
    post_filter_counts = post_summary.get('counts', {})

    # ==========================
    # FINAL PIPELINE SUMMARY
    # ==========================
    log.info("========================================")
    log.info(f"FINAL DATASET SUMMARY FOR: {base_name}")
    log.info("========================================")
    log.info(f"Audio duration             : {time.strftime('%H:%M:%S', time.gmtime(audio_duration_seconds))}")
    log.info(f"Target identified as       : {best_speaker} (score = {best_score:.4f})")
    log.info(f"Target speech time         : {time.strftime('%H:%M:%S', time.gmtime(target_speech_duration))} ({target_speech_duration/audio_duration_seconds:.1%})")
    log.info(f"Words attributed to Target : {target_word_count}")
    log.info(f"Raw pairs (pre-merge)      : {len(pairs_detailed)}")
    log.info(f"After merging (gap={MERGE_GAP}s)   : {len(pairs_merged)}")
    log.info(f"Overlaps tagged            : {overlap_summary.get('overlapped_words', 0)} words \n                 ({overlap_summary.get('overlap_rate', 0.0):.1%})")
    log.info(f"Clean pairs before filter  : {overlap_summary.get('pairs_clean_count', 0)}")
    log.info(f"Pairs dropped by filters   : {post_filter_counts.get('dropped_by_length', 0)} (length), {post_filter_counts.get('dropped_by_conf', 0)} (confidence), {post_filter_counts.get('dropped_by_semantic', 0)} (semantic), {post_filter_counts.get('dropped_by_semantic_similarity', 0)} (similarity)")
    log.info(f"Pairs flagged for review   : {flagged_count}")
    log.info(f"Final usable pairs         : {len(final_pairs)} (avg conf: {final_avg_conf:.2f})")
    log.info(f"Pairs with empty input     : {original_empty_input_count} ({original_empty_input_pct:.1f}%)")
    
    cleanup_redundant_files(base_name)
    log.info(f"--- Finished pipeline for: {base_name} ---")
    return 0 # Success

# ==================================================================
# 🚀 Main execution block
# ==================================================================
def main():
    """
    Main function to load models once and process all audio files in a batch.
    """
    overall_start_time = time.time()
    
    # --- Pre-load all models ---
    models = {}
    model_names = ["Whisper", "Alignment", "Diarization", "SentenceTransformer"]
    with tqdm(total=len(model_names), desc="Loading AI models") as pbar:
        try:
            pbar.set_postfix_str("Whisper...")
            models['whisper'] = whisperx.load_model(WHISPER_MODEL, device=device)
            pbar.update(1)

            pbar.set_postfix_str("Alignment...")
            # A bit of a hack to get the language, we assume the first transcript is representative
            # A better way would be to run transcription for all, then alignment for all.
            # For now, we'll just use the default language.
            models['align'], models['align_meta'] = whisperx.load_align_model(language_code=LANGUAGE, device=device)
            pbar.update(1)

            pbar.set_postfix_str("Diarization...")
            models['diarize'] = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
            pbar.update(1)

            pbar.set_postfix_str("SentenceTransformer...")
            models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            pbar.update(1)
        except Exception as e:
            log.error(f"Fatal error during model loading: {e}", exc_info=True)
            log.error("This is often due to an invalid Hugging Face token or network issues.")
            log.error("Please ensure you have a valid HF_TOKEN and have accepted the terms for pyannote models on HuggingFace.")
            return

    setup_time = time.time() - overall_start_time
    
    # --- Find and process audio files ---
    audio_folder = "audio_files"
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.lower().endswith((".wav", ".mp3", ".m4a"))]
    
    if not audio_files:
        log.warning(f"No audio files found in '{audio_folder}'. Nothing to do.")
        return

    log.info(f"Found {len(audio_files)} audio file(s) to process.")
    
    per_file_times = {} # Initialize per_file_times
    for audio_file in audio_files:
        file_start_time = time.time()
        # Use a try-except block to gracefully handle errors per file
        try:
            result_code = run_pipeline(audio_file, models)
            file_duration = time.time() - file_start_time
            if result_code == 0:
                per_file_times[os.path.basename(audio_file)] = file_duration
            else:
                per_file_times[os.path.basename(audio_file)] = -1 # Indicate failure
        except Exception as e:
            log.error(f"An unexpected error occurred while processing {audio_file}: {e}", exc_info=True)
            per_file_times[os.path.basename(audio_file)] = -1


    total_time = time.time() - overall_start_time
    print_final_batch_summary(setup_time, per_file_times, total_time)

if __name__ == "__main__":
    main()in()