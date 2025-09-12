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

import librosa
import numpy as np
import pandas as pd
import requests
import statistics
import torch
import whisperx
import nltk

from pydub.silence import detect_nonsilent
from pyannote.audio import Pipeline
from numpy.linalg import norm
from numpy import dot
from pydub import AudioSegment
from pyannote.core import Annotation, Segment
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from collections import defaultdict

# ============================================================
# 🎙️ Pipeline Configuration
# All the knobs you might want to tweak live here.
# Adjust these values to control how the pipeline behaves.
# ============================================================
KEEP_DEBUG_OUTPUTS = True # True = keep merged/clean/overlap files, False = only save final cleaned_data.jsonl

# --- Pair building & filtering ---
MERGE_GAP = 2.0 # (seconds) Merge Target’s replies if the silence between them is shorter than this.
MIN_WORDS = 4 # (words) Minimum length of Target’s reply to keep it. Shorter replies are dropped.
MIN_CONF = 0.6 # (0.0–1.0 or None) Drop replies with avg confidence below this. Set None to disable.
OVERLAP_FRAC = 0.30 # (fraction) If more than this % of a word overlaps with another speaker, drop it.

# --- Speaker identification ---
IDENTIFY_THRESHOLD = 0.75 # (0.0–1.0) How strict to be when matching Target’s voice to a diarized speaker.

# --- Embedding extraction ---
MIN_EMB_DURATION = 1.0 # (seconds) Ignore very short diarization segments when building speaker embeddings.

# --- Transcription (WhisperX) ---
WHISPER_MODEL = "large-v2"  # Options: "tiny","base","small","medium","large-v2"
WHISPER_BATCH_SIZE = 32     # increase if you have GPU memory (was 16)
USE_VAD = True              # If True, split audio with VAD and transcribe segments individually
VAD_MIN_SILENCE_LEN_MS = 700   # pydub: minimum silence length to consider a split (ms)
VAD_SILENCE_THRESH_DB = -40    # pydub: silence threshold (dBFS)

# --- Auto-flagging thresholds (for review dataset) ---
OVERLAP_THRESHOLD = 0.80         # (fraction) Flag a pair if ≥ this % of its words overlap with another speaker
LOW_CONFIDENCE_THRESHOLD = 0.30  # (0.0–1.0) Flag a pair if avg confidence is below this
INPUT_TO_OUTPUT_RATIO = 10.0      # (ratio) Flag a pair if input_len / output_len exceeds this value (long input → tiny output)

# --- Constants ---
MAX_INPUT_WORDS = 50 # Cap host inputs (truncate if longer)

# --- File paths ---
PAIRS_OUTPUT = "results.jsonl" # canonical final output (filtered)
INPUT_AUDIO_FILE = "podcast.mp3" # Main input audio file
LOG_FILE = "pipeline.log" # Where pipeline logs will be written

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

NON_TARGET_FILTER = {
    # Acknowledgements / backchannels
    "uh",
    "um",
    "like",
    "you know",
    "yeah",
    "for sure",
    "really",
    "what",
    "right",
    "okay",
    "ok",
    "alright",
    "mm-hm",
    "mhm",
    "yep",
    "yup",
    "nah",
    "totally",
    "exactly",

    # expletives
    "whore",
    "slut",
    "bitch",
    "retard",
    "fuck",
    "shit",

    # Prompt noise / vague lead-ins
    "so",
    "well",
    "i mean",
    "kinda",
    "sorta",
    "basically",

    # Host hedging / repeats
    "gotcha",
    "cool",
    "wow",
    "oh wow",
    "huh",
}

# Centralized filenames (use these everywhere in the script)
BASE_PAIRS = "base_pairs.jsonl" # raw dump of base pairs (unfiltered)
PAIRS_MERGED_OUTPUT = "pairs_merged.jsonl" # optional merged output (debug)
PAIRS_META_OUTPUT = "pairs_with_overlap_meta.jsonl" # detailed per-word overlap metadata (debug)
PAIRS_CLEAN_OUTPUT = "pairs_clean.jsonl" # clean pairs after overlap removal (debug)
PAIRS_TMP_PREFIX = "pairs_tmp" # prefix used for temporary postfilter outputs
POSTFILTER_SUMMARY = "pairs_postfilter_summary.json" # final postfilter summary
OVERLAP_SUMMARY = "overlap_phaseB_summary.json" # overlap summary file

# -------------------------
# Script Setup
# -------------------------
nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))
HF_TOKEN = os.getenv("HF_TOKEN") # safer than hardcoding, user must set env var
if HF_TOKEN is None:
    HF_TOKEN = "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"
    logging.warning("[SETUP] No HuggingFace token found in environment variable HF_TOKEN.")

step_timings = {} # collect per-step durations
device = "cuda" if torch.cuda.is_available() else "cpu"

ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg-bin")
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

print(f"[FFMPEG] Using ffmpeg: {ffmpeg_exe}")
print(f"[FFMPEG] Using ffprobe: {ffprobe_exe}")

# ------------------------------
# Logging setup: console + file
# ------------------------------
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])
logging.info("=== Pipeline started ===")

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

def identify_best_speaker(speaker_embeddings, target_avg_emb, threshold=0.75):
    """
    Identify Target's speaker label from diarized speaker embeddings.
    Returns (best_speaker, best_score, all_scores).
    """
    scores = {}
    for spk, emb in speaker_embeddings.items():
        sim = dot(emb, target_avg_emb) / (norm(emb) * norm(target_avg_emb))
        scores[spk] = sim
        logging.info(f"[IDENTIFY] Similarity {spk} = {sim:.4f}")

    if not scores:
        logging.error("[IDENTIFY] No speaker embeddings available.")
        return None, None, {}

    best_speaker = max(scores, key=scores.get)
    best_score = scores[best_speaker]

    if best_score >= threshold:
        logging.info(f"[IDENTIFY] Best match: {best_speaker} (score={best_score:.4f})")
        return best_speaker, best_score, scores
    else:
        logging.warning(f"[IDENTIFY] No speaker passed threshold {threshold:.2f}. Best was {best_speaker} ({best_score:.4f})")
        return None, best_score, scores
    
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
        logging.error(f"[EMBED] Cannot load audio file {audio_file}: {e}")
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        i = 0
        # Accept pyannote.Annotation via itertracks or fallback to list/dict
        if hasattr(diarization, "itertracks"):
            tracks = list(diarization.itertracks(yield_label=True))
            iterator = ((seg, label) if len(item := (seg, track, label))==3 else None for (seg, track, label) in tracks)  # readable form not used
            for track in diarization.itertracks(yield_label=True):
                seg = track[0]
                label = track[-1]
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
                    logging.debug(f"[EMBED] Failed embedding segment {label} {seg}: {e}")
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
                    logging.debug(f"[EMBED] Failed embedding segment {label} {start}->{end}: {e}")

    # Average + normalize per speaker
    averaged_embeddings = {}
    for spk, embs in speaker_embeddings.items():
        if not embs:
            continue
        embs = [e / np.linalg.norm(e) for e in embs if np.linalg.norm(e) > 0]
        avg = np.mean(embs, axis=0)
        avg /= np.linalg.norm(avg)
        averaged_embeddings[spk] = avg
        logging.info(f"[EMBED] Speaker {spk}: {len(embs)} segments used.")

    logging.info(f"[EMBED] Computed embeddings for {len(averaged_embeddings)} speakers.")
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
        logging.debug(f"[IDENTIFY] Embedding failed for {wav_path}: {e}")
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
            logging.warning(f"[IDENTIFY] Skipping sample {fpath}: {e}")
    if not ref_embs:
        raise ValueError("No valid reference embeddings found in samples folder.")
    # average to single target embedding
    target_avg_emb = np.mean(np.stack(ref_embs, axis=0), axis=0)
    return ref_embs, target_avg_emb

def print_timings_summary():
    """Print all timings clearly at the end."""
    print("\n" + "="*40)
    print("PIPELINE TIMING SUMMARY")
    print("="*40)
    total = 0
    for step, elapsed in step_timings.items():
        mins, secs = divmod(int(elapsed), 60)
        print(f"{step:25s}: {mins}m {secs}s")
        total += elapsed
    mins, secs = divmod(int(total), 60)
    print("-"*40)
    print(f"TOTAL                  : {mins}m {secs}s")
    print("="*40 + "\n")

def safe_assign_word_speakers(diarize_segments, result):
    """
    Try to assign speakers using WhisperX's built-in method.
    If it fails (due to serialization, version, or overlap issues),
    fall back to a manual attachment method.
    """
    try:
        logging.info("[ASSIGN] Attempting whisperx.assign_word_speakers...")
        assigned = whisperx.assign_word_speakers(diarize_segments, result)
        logging.info("[ASSIGN] Successfully assigned speakers with whisperx.")
        return assigned
    except Exception as e:
        logging.warning(f"[ASSIGN] whisperx.assign_word_speakers failed: {e}")
        logging.info("[ASSIGN] Falling back to custom attach_speakers_to_transcript...")
        try:
            assigned = attach_speakers_to_transcript(diarize_segments, result)
            logging.info("[ASSIGN] Successfully assigned speakers with fallback method.")
            return assigned
        except Exception as inner_e:
            logging.error(f"[ASSIGN] Fallback attach_speakers_to_transcript also failed: {inner_e}")
            raise
    
def attach_speakers_to_transcript(diarization, aligned_result, diarization_json_path="podcast.diarization.json"):
    """
    Robustly attach speaker labels to each word in an aligned transcript.
    - If 'diarization' has itertracks(), prefer that (pyannote Annotation).
    - Otherwise, read diarization from `diarization_json_path` and perform max-overlap.
    Returns aligned_result with word['speaker'] set.
    """
    import json
    logging.info("[ATTACH] Attaching speakers to aligned transcript...")

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
        logging.info(f"[ATTACH] Using pyannote Annotation → {len(diar_segments)} segments")
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
            logging.info(f"[ATTACH] Loaded {len(diar_segments)} diarization segments from JSON")
        except Exception as e:
            logging.error(f"[ATTACH] Failed to load diarization JSON: {e}")
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
    logging.info(f"[ATTACH] Attached {attached}/{total_words} words. Per-speaker sample: {dict(list(counts.items())[:10])}")

    # Optionally write a debug copy of the attached alignment
    try:
        with open("podcast.aligned.attached.json", "w", encoding="utf-8") as f:
            json.dump(aligned_result, f, ensure_ascii=False, indent=2)
        logging.info("[ATTACH] Wrote podcast.aligned.attached.json for inspection.")
    except Exception:
        pass

    return aligned_result

def adaptive_conf_filter(words, cutoff_strategy="percentile", min_cutoff=0.4, grace=0.1):
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
        if score is None or score >= cutoff - grace:  # 🔧 allow grace below cutoff
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
        if f: cur=[t0,t1] if cur is None else [cur[0],t1]
        elif cur: segs.append(tuple(cur)); cur=None
    if cur: segs.append(tuple(cur))
    return segs

def run_vad_with_fallback(wav_path, vad_mult=1.0, hf_token_env="HF_HUB_TOKEN"):
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
        logging.info(f"[VAD-NEURAL] pyannote returned {len(vad_segments)} segments")
        if vad_segments:
            return vad_segments
    except Exception as e:
        logging.info(f"[VAD-NEURAL] neural VAD failed/absent, falling back: {e}")

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
    logging.info(
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
        logging.debug(f"[PAIR] Truncated input for seg_idx={seg_idx} to {input_len} words (MAX_INPUT_WORDS={max_in})")

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

def reassign_short_target_words(words, target_label, window=1.0, wav_path=None, target_emb=None, encoder=None):
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
                if target_emb is not None and wav_path is not None:
                    sim, accept = contextual_reid_with_embeddings(
                        wav_path, start, w.get("end", start + 0.05),
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

    logging.info(f"[FILTER] Dropped {dropped_non_target} non-target filler words using NON_TARGET_FILTER={list(non_target_filter)}")
    logging.info(f"[BUILD_PAIRS] Leading target words shifted: {leading_shifted}")

    return pairs, dropped_non_target

def build_pairs_detailed(assigned_data, target_speaker, MIN_WORDS=1):
    """
    Like build_pairs but returns detailed pairs:
    - seg_idx, input, output, output_len, avg_score, tstart, tend, words (list of dicts)
    """
    pairs = []
    context_buffer = []

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
            reply_text = " ".join(w["text"] for w in reply_words).strip()
            reply_len = len(reply_text.split()) if reply_text else 0

            if reply_len >= MIN_WORDS:
                starts = [w["start"] for w in reply_words if w.get("start") is not None]
                ends = [w["end"] for w in reply_words if w.get("end") is not None]
                tstart = min(starts) if starts else None
                tend = max(ends) if ends else None
                scores = [w["score"] for w in reply_words if isinstance(w.get("score"), (int, float))]
                avg_score = statistics.mean(scores) if scores else None
                # Build raw context text
                context_text = " ".join([c["text"] for c in context_buffer]).strip()
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
                    logging.debug(f"[PAIRS-D] Truncated detailed context for seg_idx={seg_idx} to {context_len} words (MAX_INPUT_WORDS={max_in})")

                pairs.append({
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
                })
            # reset context
            context_buffer = []
        else:
            context_buffer.extend(normalized)

    logging.info(f"[PAIRS-D] Built {len(pairs)} detailed pairs for {target_speaker}.")
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

    # common pyannote shape: {'uri':..., 'content': [{'segment':{'start':..., 'end':...}, 'label': ...}, ...]}
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
    logging.info(f"[DIAR-PARSE] Parsed {len(segments)} diarization segments from {diarization_json_path}")
    return segments

def auto_flag_low_quality_pairs(pairs_meta_path=PAIRS_META_OUTPUT, flagged_out="pairs_flagged.jsonl",
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
        logging.info("[FLAG] No pairs_meta file found (%s) — skipping auto-flag.", pairs_meta_path)
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
            input_len = p.get("input_len") or (len(p.get("input","").split()) if p.get("input") else 0)
            output_len = p.get("output_len") or (len(p.get("output","").split()) if p.get("output") else 0)
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

    logging.info(f"[FLAG] Auto-flagged {len(flagged)} pairs to {flagged_out}.")
    return flagged_ids

def is_semantically_rich(text):
    """
    Returns True if text has at least 2 unique non-stopword tokens.
    """
    tokens = re.findall(r"\w+", text.lower())
    content = [t for t in tokens if t not in STOPWORDS]
    return len(set(content)) >= 2

def compute_overlap_and_write(pairs_detailed, diarization_json_path="podcast.diarization.json", overlap_word_frac=0.30):
    """
    Tag per-word overlap fraction (relative to non-target speakers),
    write pairs_with_overlap_meta.jsonl and pairs_clean.jsonl (clean=drop overlapped words).
    """
    diar_segments = parse_diarization_json_to_segments(diarization_json_path)

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

    for p in pairs_detailed:
        # annotate each word
        words = p.get("words", [])
        for w in words:
            w_start = w.get("start"); w_end = w.get("end")
            ov_frac = overlap_fraction(w_start, w_end, p.get("output_speaker"))
            w["overlap_frac"] = ov_frac
            w["overlapped"] = ov_frac > overlap_word_frac

        total_words += len(words)
        overlapped_words += sum(1 for w in words if w.get("overlapped"))

        # compute quality score and attach
        try:
            qscore, qmetrics = score_pair_quality(p)
            p["quality_score"] = qscore
            p["quality_metrics"] = qmetrics
        except Exception as e:
            logging.debug(f"[SCORE] Failed scoring pair {p.get('seg_idx')}: {e}")
            p["quality_score"] = 0.0
            p["quality_metrics"] = {}

        pairs_meta.append(p)

        # build clean pair (drop overlapped words)
        clean_words = [w for w in words if not w.get("overlapped")]
        if clean_words:
            clean_text = " ".join((w.get("text") or w.get("word") or "").strip() for w in clean_words).strip()
            scores = [w.get("score") for w in clean_words if isinstance(w.get("score"), (int, float))]
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
    with open(PAIRS_META_OUTPUT, "w", encoding="utf-8") as f:
        for pm in pairs_meta:
            f.write(json.dumps(pm, ensure_ascii=False) + "\n")
    with open(PAIRS_CLEAN_OUTPUT, "w", encoding="utf-8") as f:
        for pc in pairs_clean:
            f.write(json.dumps(pc, ensure_ascii=False) + "\n")
    # write pair-quality CSV
    try:
        csv_fields = ["seg_idx", "output_speaker", "input_len", "output_len", "avg_score", "quality_score", "overlap_rate"]
        with open("pair_quality_metrics.csv", "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_fields)
            writer.writeheader()
            for pm in pairs_meta:
                writer.writerow({
                    "seg_idx": pm.get("seg_idx"),
                    "output_speaker": pm.get("output_speaker"),
                    "input_len": pm.get("input_len"),
                    "output_len": pm.get("output_len"),
                    "avg_score": pm.get("avg_score"),
                    "quality_score": pm.get("quality_score"),
                    "overlap_rate": statistics.mean([w.get("overlap_frac", 0.0) for w in pm.get("words", [])]) if pm.get("words") else 0.0
                })
    except Exception as e:
        logging.warning(f"[CSV] Could not write pair quality CSV: {e}")

    summary = {
        "pairs_meta_count": len(pairs_meta),
        "pairs_clean_count": len(pairs_clean),
        "total_words": total_words,
        "overlapped_words": overlapped_words,
        "overlap_rate": overlapped_words / total_words if total_words else 0.0,
        "clean_drop_rate": 1 - (len(pairs_clean) / len(pairs_meta) if pairs_meta else 1)
    }
    with open(OVERLAP_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"[OVERLAP] Wrote pairs_with_overlap_meta.jsonl ({len(pairs_meta)}) and pairs_clean.jsonl ({len(pairs_clean)}).")
    logging.info(f"[OVERLAP] Summary: {summary}")
    return summary

def score_pair_quality(pair, weights=None, punc_bonus=0.1, ratio_floor=0.3, ratio_ceiling=4.0):
    """
    Compute a soft quality score for a single pair_meta dict.
    Returns (score, metrics_dict).
    Metrics included: avg_conf, overlap_rate, input_len, output_len, ratio, punc_ok, speaker_diversity.
    Weight defaults tuned conservatively.
    """
    if weights is None:
        weights = {
            "conf": 0.45,
            "overlap": 0.25,
            "ratio": 0.15,
            "punc": 0.10,
            "diversity": 0.05
        }

    words = pair.get("words", [])
    # avg confidence (use any score/confidence in words)
    confs = [w.get("score") for w in words if isinstance(w.get("score"), (int, float))]
    avg_conf = float(statistics.mean(confs)) if confs else float(pair.get("avg_score") or 0.0)

    # overlap: fraction of words flagged overlapped
    overlapped_count = sum(1 for w in words if bool(w.get("overlapped")))
    overlap_rate = overlapped_count / max(1, len(words))

    # input / output lengths
    input_len = int(pair.get("input_len") or 0)
    output_len = int(pair.get("output_len") or 0)
    # if output_len==0 then score should be low
    ratio = (input_len / output_len) if output_len > 0 else float('inf')

    # ratio score: close to 1 is ideal, extreme values penalized
    if output_len == 0:
        ratio_score = 0.0
    else:
        # map ratio to (0,1]: if between ratio_floor..ratio_ceiling -> ok
        if ratio < ratio_floor:
            ratio_score = ratio / ratio_floor  # small scale-up
        elif ratio > ratio_ceiling:
            ratio_score = max(0.0, 1.0 - ((ratio - ratio_ceiling) / (ratio_ceiling))) 
        else:
            ratio_score = 1.0

    # punctuation heuristic on output
    out_text = (pair.get("output") or "").strip()
    punc_ok = 1.0 if out_text.endswith((".", "?", "!")) else 0.0

    # speaker diversity of input: prefer at least 2 input speakers (normalized)
    input_speakers = set()
    for w in words:
        if w.get("speaker") and w.get("speaker") != pair.get("output_speaker"):
            input_speakers.add(w.get("speaker"))
    diversity = min(1.0, len(input_speakers) / 2.0)

    # combine weighted
    score = (
        weights["conf"] * avg_conf +
        weights["overlap"] * (1.0 - overlap_rate) +  # prefer low overlap
        weights["ratio"] * ratio_score +
        weights["punc"] * (punc_ok * punc_bonus + (1.0 - punc_bonus) * punc_ok) +  # small bonus if punctuated
        weights["diversity"] * diversity
    )

    # clamp to [0,1]
    score = max(0.0, min(1.0, score))

    metrics = {
        "avg_conf": avg_conf,
        "overlap_rate": overlap_rate,
        "input_len": input_len,
        "output_len": output_len,
        "ratio": ratio if output_len > 0 else None,
        "punc_ok": bool(punc_ok),
        "diversity": diversity
    }

    return float(score), metrics

def postfilter_clean_pairs(clean_pairs, min_words=3, min_conf=None, out_prefix="pairs"):
    """
    Post-filter clean pairs by minimum words and optional minimum avg confidence.
    Writes out_prefix + '_filtered.jsonl' and a summary JSON.
    """
    filtered = []
    dropped_len = 0
    dropped_conf = 0
    for p in clean_pairs:
        if p.get("output_len", 0) < min_words:
            dropped_len += 1
            continue
        if (min_conf is not None) and (p.get("avg_score") is not None) and (p.get("avg_score") < min_conf):
            dropped_conf += 1
            continue
        filtered.append(p)

    out_file = f"{out_prefix}_filtered.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for p in filtered:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "params": {"min_words": min_words, "min_conf": min_conf},
        "counts": {"before": len(clean_pairs), "after": len(filtered), "dropped_by_length": dropped_len, "dropped_by_conf": dropped_conf}
    }
    with open(f"{out_prefix}_postfilter_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"[POSTFILTER] Wrote {out_file} ({len(filtered)} kept). Summary: {summary}")
    return summary

def identify_speaker(samples_folder, audio_file, diarization, threshold=IDENTIFY_THRESHOLD):
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
        ref_embs, target_avg_emb = load_reference_embeddings(samples_folder)
    except Exception as e:
        logging.error(f"[IDENTIFY] Could not load reference embeddings: {e}")
        return None, None

    encoder = VoiceEncoder()  # will be used for segment embeddings
    speaker_embs = {}   # speaker_label -> list of embeddings
    skipped_short = 0
    skipped_error = 0
    extracted = 0

    # Ensure diarization is iterable: it could be pyannote Annotation or a list of dict entries
    diar_segments = []
    if hasattr(diarization, "itertracks"):
        # pyannote Annotation
        for segment, track, label in diarization.itertracks(yield_label=True):
            diar_segments.append((float(segment.start), float(segment.end), label))
    else:
        # fallback: expect diarization to be a dict (podcast.diarization.json loaded) or list of segments
        if isinstance(diarization, dict):
            content = diarization.get("content", []) or diarization.get("segments", [])
            for entry in content:
                seg = entry.get("segment") or {}
                start = seg.get("start")
                end = seg.get("end")
                label = entry.get("label") or entry.get("speaker")
                if start is None or end is None or label is None:
                    continue
                diar_segments.append((float(start), float(end), label))
        elif isinstance(diarization, list):
            for entry in diarization:
                seg = entry.get("segment")
                if seg:
                    diar_segments.append((float(seg.get("start")), float(seg.get("end")), entry.get("label") or entry.get("speaker")))

    if not diar_segments:
        logging.warning("[IDENTIFY] No diarization segments available.")
        return None, None

    # Use a tempdir for extracted samples — avoids permission issues with a hardcoded samples/ folder
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = None
        try:
            audio = AudioSegment.from_file(audio_file)
        except Exception as e:
            logging.error(f"[IDENTIFY] Cannot load audio file {audio_file}: {e}")
            return None, None

        for i, (start, end, label) in enumerate(diar_segments):
            dur = end - start
            if dur < MIN_EMB_DURATION:
                skipped_short += 1
                continue
            # pydub expects milliseconds
            s_ms = int(start * 1000)
            e_ms = int(end * 1000)
            try:
                seg_audio = audio[s_ms:e_ms]
                tmp_path = os.path.join(tmpdir, f"seg_{i}_{label}.wav")
                seg_audio.export(tmp_path, format="wav")
                emb = _embed_audio_segment(tmp_path, encoder)
                if emb is None:
                    skipped_error += 1
                    continue
                speaker_embs.setdefault(label, []).append(emb)
                extracted += 1
            except Exception as e:
                skipped_error += 1
                logging.debug(f"[IDENTIFY] Failed to extract/embed segment {label} [{start}->{end}]: {e}")
                continue

    # Summarize
    logging.info(f"[IDENTIFY] Extracted embeddings: {extracted}. Skipped short: {skipped_short}, errors: {skipped_error}. Per-speaker counts: { {k: len(v) for k,v in speaker_embs.items()} }")

    if not speaker_embs:
        logging.warning("[IDENTIFY] No speaker embeddings computed — diarization may be empty or too short.")
        return None, None

    # compute avg embedding per speaker, then similarity to the target reference embedding
    avg_embs = {}
    for sp, embs in speaker_embs.items():
        try:
            avg_emb = np.mean(np.stack(embs, axis=0), axis=0)
            avg_embs[sp] = avg_emb
        except Exception as e:
            logging.debug(f"[IDENTIFY] failed averaging speaker {sp}: {e}")

    if not avg_embs:
        logging.warning("[IDENTIFY] Averaging per-speaker embeddings failed or produced no results.")
        return None, None

    # similarity: 1 - cosine_distance  (range ~ -1..1 but for normalized embeddings ~0..1)
    best_speaker = None
    best_similarity = -999.0
    for sp, emb in avg_embs.items():
        try:
            dist = cosine(emb, target_avg_emb)  # smaller distance = closer
            similarity = 1.0 - dist
        except Exception as e:
            logging.debug(f"[IDENTIFY] cosine failed for speaker {sp}: {e}")
            continue
        logging.info(f"[IDENTIFY] speaker {sp} similarity={similarity:.4f} (dist={dist:.4f})")
        if similarity > best_similarity:
            best_similarity = similarity
            best_speaker = sp

    # threshold check: require some minimum similarity
    if best_speaker is None or (best_similarity < threshold):
        logging.warning(f"[IDENTIFY] Best speaker {best_speaker} has similarity {best_similarity:.4f} < threshold {threshold}. Rejecting.")
        return None, None

    logging.info(f"[IDENTIFY] Selected target speaker {best_speaker} (similarity={best_similarity:.4f})")
    return best_speaker, float(best_similarity)

def transcribe_with_vad(model, audio_path, batch_size=WHISPER_BATCH_SIZE,
                        condition_on_previous_text=False,
                        min_silence_len_ms=VAD_MIN_SILENCE_LEN_MS,
                        silence_thresh_db=VAD_SILENCE_THRESH_DB):
    """
    Split the audio into non-silent chunks using pydub, transcribe each chunk,
    then stitch the segments back together into a single Whisper-style result dict
    with adjusted segment timestamps.
    """
    logging.info("[TRANSCRIBE-VAD] Using VAD: min_silence_ms=%d silence_thresh_db=%d",
                 min_silence_len_ms, silence_thresh_db)

    audio = AudioSegment.from_file(audio_path)
    # detect_nonsilent returns list of [start_ms, end_ms] intervals
    nonsilent_intervals = detect_nonsilent(audio,
                                          min_silence_len=min_silence_len_ms,
                                          silence_thresh=silence_thresh_db)

    # If VAD found nothing, fall back to full-file transcription.
    # NOTE: remove unsupported kwargs (like condition_on_previous_text) here.
    if not nonsilent_intervals:
        logging.warning("[TRANSCRIBE-VAD] VAD returned no voiced intervals — falling back to full-file transcription.")
        return model.transcribe(audio_path, batch_size=batch_size, language="en")

    merged_result = {"language": None, "segments": []}

    with tempfile.TemporaryDirectory() as td:
        for i, (start_ms, end_ms) in enumerate(nonsilent_intervals):
            # Export segment
            seg = audio[start_ms:end_ms]
            tmp_path = os.path.join(td, f"vad_seg_{i}.wav")
            seg.export(tmp_path, format="wav")

            # Transcribe this segment
            try:
                # remove unsupported kwargs when calling model.transcribe
                res = model.transcribe(tmp_path, batch_size=batch_size, language="en")
            except Exception as e:
                logging.warning(f"[TRANSCRIBE-VAD] Transcription failed for segment {i} [{start_ms}..{end_ms}]: {e}")
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

    # fix: check merged_result, not undefined 'result'
    if "language" not in merged_result or merged_result["language"] is None:
        logging.warning("[TRANSCRIBE] No language detected. Forcing 'en'.")
        merged_result["language"] = "en"

    # sort segments by start time to be safe
    merged_result["segments"] = sorted(merged_result["segments"], key=lambda s: s.get("start", 0.0))
    logging.info("[TRANSCRIBE-VAD] Transcribed %d voiced intervals into %d segments.",
                 len(nonsilent_intervals), len(merged_result["segments"]))

    return merged_result

def contextual_reid_with_embeddings(wav_path, word_start, word_end, target_emb, encoder=None, pad=0.05, similarity_threshold=0.72):
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
            audio = AudioSegment.from_file(wav_path)
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
        logging.debug(f"[REID] contextual reid failed: {e}")
        return 0.0, False
    
# -------------------------
# Run pipeline
# -------------------------
def run_pipeline():
    start_time = time.time()

    # ------------------------------
    # STEP 1: Transcribe audio → text with WhisperX
    # ------------------------------
    logging.info("[STEP] 1 ... Transcribe audio → text with WhisperX")

    # Ensure we have a WAV copy for VAD and libraries needing PCM
    wav_path = INPUT_AUDIO_FILE.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(INPUT_AUDIO_FILE)
            audio.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")
            logging.info(f"[AUDIO] Exported WAV copy for VAD: {wav_path}")
        except Exception as e:
            logging.warning(f"[AUDIO] Could not export WAV copy: {e}")

    transcript_cache = "podcast.transcript.json"
    if os.path.exists(transcript_cache):
        logging.info("[CACHE] Loading cached transcript...")
        with open(transcript_cache, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        logging.info("[TRANSCRIBE] Running WhisperX transcription...")
        model = whisperx.load_model(WHISPER_MODEL, device=device)
        if USE_VAD:
            result = transcribe_with_vad(
                model,
                INPUT_AUDIO_FILE,
                batch_size=WHISPER_BATCH_SIZE,
                min_silence_len_ms=VAD_MIN_SILENCE_LEN_MS,
                silence_thresh_db=VAD_SILENCE_THRESH_DB,
            )
        else:
            result = model.transcribe(
                INPUT_AUDIO_FILE,
                batch_size=WHISPER_BATCH_SIZE,
                language="en"
            )
        with open(transcript_cache, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ✅ Now it's safe to check language
    if not result.get("language"):
        logging.warning("[LANG] Transcript missing language — forcing 'en'")
        result["language"] = "en"

    start_time = mark_step("Transcription", start_time)


    # TODO: Remove caching once pipeline is finalized 
    # NOTE: Safe because WhisperX returns a dict.

    # ------------------------------
    # STEP 2: Align transcript to audio (word-level timestamps)
    # ------------------------------
    logging.info("[STEP] 2 ... Align transcript to audio")
    alignment_cache = "podcast.aligned.json"
    if os.path.exists(alignment_cache):
        logging.info("[CACHE] Loading cached alignment...")
        with open(alignment_cache, "r", encoding="utf-8") as f:
            result_aligned = json.load(f)
    else:
        logging.info("[ALIGN] Running WhisperX alignment...")
        alignment_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], alignment_model, metadata, INPUT_AUDIO_FILE, device)
        with open(alignment_cache, "w", encoding="utf-8") as f:
            json.dump(result_aligned, f, ensure_ascii=False, indent=2)

    start_time = mark_step("Align transcript to audio", start_time)
    # TODO: Remove caching once pipeline is finalized
    # ------------------------------
    # STEP 3: Speaker diarization (who spoke when)
    # ------------------------------
    logging.info("[STEP] 3 ... Speaker diarization (who spoke when)")
    diar_cache = "podcast.diarization.json"

    if os.path.exists(diar_cache):
        logging.info("[CACHE] Loading cached diarization...")
        with open(diar_cache, "r", encoding="utf-8") as f:
            diarization_json = json.load(f)

        # Rebuild Annotation from cached JSON
        diarization = Annotation(uri=diarization_json.get("uri", None))
        for entry in diarization_json.get("content", []):
            seg = entry["segment"]
            diarization[Segment(seg["start"], seg["end"])] = entry["label"]

    else:
        logging.info("[CACHE] No cached diarization found, generating anew...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        # Returns pandas.DataFrame: start, end, speaker columns
        diarization_df = diarize_model(INPUT_AUDIO_FILE)

        # Convert DataFrame → pyannote.core.Annotation
        diarization = Annotation(uri="podcast")
        for _, row in diarization_df.iterrows():
            diarization[Segment(row["start"], row["end"])] = row["speaker"]

        # Build cache-friendly JSON
        diarization_dict = {
            "uri": "podcast",
            "content": [
                {"segment": {"start": row["start"], "end": row["end"]}, "label": row["speaker"]}
                for _, row in diarization_df.iterrows()
            ],
        }
        with open(diar_cache, "w", encoding="utf-8") as f:
            json.dump(diarization_dict, f, ensure_ascii=False, indent=2)
        logging.info(f"[CACHE] Saved new diarization to {diar_cache}")
    # Safety: ensure correct type
    if not isinstance(diarization, Annotation):
        raise TypeError(f"Expected pyannote.core.Annotation, got {type(diarization)}")
    start_time = mark_step("Diarization", start_time)
    # ------------------------------
    # STEP 4: Assign speakers to transcript words
    # ------------------------------
    
    logging.info("[STEP] 4 ... Assign speakers to transcript words")
    assign_cache = "podcast.assigned.json"
    if os.path.exists(assign_cache):
        logging.info("[CACHE] Loading cached speaker assignment...")
        with open(assign_cache, "r", encoding="utf-8") as f:
            result_aligned = json.load(f)
    else:
        logging.info("[ASSIGN] Attempting safe assign_word_speakers...")
        # load aligned_result (from alignment cache)
        with open(alignment_cache, "r", encoding="utf-8") as f:
            result_aligned = json.load(f)

        # attach
        result_aligned = attach_speakers_to_transcript(diarization, result_aligned, diarization_json_path="podcast.diarization.json")

        # save assignment cache
        with open(assign_cache, "w", encoding="utf-8") as f:
            json.dump(result_aligned, f, ensure_ascii=False, indent=2)

    start_time = mark_step("Speaker Assignment", start_time)

    # ------------------------------
    # STEP 5: Build training pairs
    # ------------------------------
    logging.info("[STEP] 5 ... Build training pairs")
    best_speaker, best_score = identify_speaker("samples", INPUT_AUDIO_FILE, diarization)
    if not best_speaker:
        logging.error("[PIPELINE] No reference speaker identified. Aborting pair generation.")
        return

    # Load assigned segments
    with open(assign_cache, "r", encoding="utf-8") as f:
        assigned_data = json.load(f)
    assigned_segments = assigned_data.get("segments", [])

    # --- Normalize words across all segments ---
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
    logging.info(f"[ADAPTIVE] Removed {len(removed_conf)} words below cutoffs: {cutoffs}")

    # Run neural VAD with fallback
    wav_path = INPUT_AUDIO_FILE.replace(".mp3", ".wav")
    vad_segments = run_vad_with_fallback(wav_path, vad_mult=CFG.get("VAD_MULT", 1.0))

    # Keep only words that overlap speech according to VAD
    words_vad = [w for w in words_conf if overlaps_vad(w, vad_segments)]
    logging.info(f"[VAD] Kept {len(words_vad)} words after speech activity check")

    # Compute speaker embeddings for contextual re-ID (optional, potentially slow)
    # We compute target embedding so we can use contextual_reid_with_embeddings for doubtful short words.
    wav_path = INPUT_AUDIO_FILE.replace(".mp3", ".wav")
    try:
        speaker_embs = compute_speaker_embeddings(wav_path, diarization)
        target_emb = speaker_embs.get(best_speaker)
    except Exception as e:
        logging.warning(f"[EMBED] Could not compute speaker embeddings for re-ID: {e}")
        target_emb = None

    # Reassignment correction (short target words mis-assigned).
    # Pass wav_path + target_emb so function can optionally call contextual_reid_with_embeddings.
    words_reassigned = reassign_short_target_words(words_vad, best_speaker, window=1.0,
                                                   wav_path=wav_path, target_emb=target_emb)
    logging.info(f"[REASSIGN] Applied correction for short words near target speaker (reassigned -> {len([w for w in words_reassigned if w.get('speaker')==best_speaker])} target words)")

    # Rebuild segments from filtered + reassigned words
    segments_final = regroup_words_to_segments(words_reassigned)

    # Build training pairs from the corrected segments
    pairs, dropped_non_target = build_pairs(segments_final, best_speaker)

    # Save base pairs
    with open(BASE_PAIRS, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logging.info(f"[ADAPTIVE] Removed {len(removed_conf)} words below cutoffs: {cutoffs}")
    logging.info(f"[VAD] Kept {len(words_vad)} words after speech activity check")

    # Use global config vars
    args = type("Args", (), {})()
    args.min_words = MIN_WORDS
    args.merge_gap = MERGE_GAP
    args.overlap_frac = OVERLAP_FRAC
    args.min_conf = MIN_CONF

    # Build detailed pairs
    pairs_detailed = build_pairs_detailed(assigned_data, best_speaker)
    target_word_count = sum(
        1 for seg in assigned_data.get("segments", [])
        for w in seg.get("words", [])
        if w.get("speaker") == best_speaker
    )
    # Phase A: Merge
    pairs_merged = merge_close_replies(pairs_detailed, gap_threshold=args.merge_gap)
    with open(PAIRS_MERGED_OUTPUT, "w", encoding="utf-8") as f:
        for p in pairs_merged:
            f.write(json.dumps({
                "input": p.get("input",""),
                "output": p.get("output",""),
                "output_speaker": p.get("output_speaker")
            }, ensure_ascii=False) + "\n")

    # Phase B: Overlap filtering
    overlap_summary = compute_overlap_and_write(
        pairs_merged,
        diarization_json_path="podcast.diarization.json",
        overlap_word_frac=args.overlap_frac
    )

    # Auto-flag suspicious pairs for manual review
    flagged_ids = auto_flag_low_quality_pairs(
        pairs_meta_path=PAIRS_META_OUTPUT,
        flagged_out="pairs_flagged.jsonl",
        overlap_pair_threshold=OVERLAP_THRESHOLD,
        low_conf_threshold=LOW_CONFIDENCE_THRESHOLD,
        long_input_small_output_ratio=INPUT_TO_OUTPUT_RATIO
    )
    logging.info(f"[FLAG] {len(flagged_ids)} pairs flagged for manual review (pairs_flagged.jsonl).")

    merged_pairs = pairs_merged  # keep reference for summary
    # Phase C: Post-filter
    # Load clean pairs produced above and apply post-filter (min words and optional min confidence)
    clean_pairs = []
    try:
        with open(PAIRS_CLEAN_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                clean_pairs.append(json.loads(line))
    except Exception:
        logging.warning("[POSTFILTER] pairs_clean.jsonl not found — skipping postfilter.")
        clean_pairs = []
    # Step 5c: Post-filter clean pairs
    # Run postfiltering into a temp prefix (function will write PAIRS_TMP_PREFIX + "_filtered.jsonl" and a summary)
    post_summary = postfilter_clean_pairs(
        clean_pairs,
        min_words=args.min_words,
        min_conf=args.min_conf,
        out_prefix=PAIRS_TMP_PREFIX
    )

    # Read the postfiltered items and write them as the canonical final dataset (PAIRS_OUTPUT).
    final_pairs = []
    tmp_filtered_path = f"{PAIRS_TMP_PREFIX}_filtered.jsonl"
    try:
        with open(tmp_filtered_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                final_pairs.append(json.loads(line))
    except FileNotFoundError:
        logging.error("[PIPELINE] Expected temporary filtered file not found: %s. Aborting final write.", tmp_filtered_path)
        final_pairs = []

    # Apply quality scoring to each final pair
    for p in final_pairs:
        s = score_pair(p, words_vad, best_speaker)
        p["quality"] = s

    # Partition by quality
    clean_pairs = [p for p in final_pairs if p["quality"]["score"] >= 0.7]
    borderline_pairs = [p for p in final_pairs if 0.4 <= p["quality"]["score"] < 0.7]
    flagged_pairs = [p for p in final_pairs if p["quality"]["score"] < 0.4]

    logging.info(f"[QUALITY] Clean={len(clean_pairs)}, Borderline={len(borderline_pairs)}, Flagged={len(flagged_pairs)}")

    # Write metrics CSV
    import csv
    with open("pair_quality_metrics.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["confidence","ratio","complete","score"])
        writer.writeheader()
        for p in final_pairs:
            writer.writerow(p["quality"])

    # helper to detect empty / whitespace-only inputs
    def _is_empty_input(pair):
        inp = pair.get("input")
        return inp is None or (isinstance(inp, str) and not inp.strip())

    # After postfilter_clean_pairs, before writing final_pairs
    final_pairs = [p for p in final_pairs if is_semantically_rich(p.get("output",""))]

    # Count empty inputs BEFORE we replace them (keep this original count for the summary)
    original_empty_input_count = sum(1 for p in final_pairs if _is_empty_input(p))
    original_empty_input_pct = (original_empty_input_count / len(final_pairs) * 100) if final_pairs else 0.0

    # ✅ Replace empty inputs with marker
    for p in final_pairs:
        if _is_empty_input(p):
            p["input"] = "[CONTEXT CONTINUES]"

    # Write canonical final output (overwrites any previous results.jsonl)
    with open(PAIRS_OUTPUT, "w", encoding="utf-8") as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # NEW: count flagged pairs for review
    flagged_count = 0
    if os.path.exists("pairs_flagged.jsonl"):
        with open("pairs_flagged.jsonl", "r", encoding="utf-8") as f:
            flagged_count = sum(1 for _ in f)

    # Also ensure postfilter summary is written under canonical name (the helper already writes something like pairs_tmp_postfilter_summary.json)
    # If the helper wrote a temp summary, move/rename it to the canonical POSTFILTER_SUMMARY
    tmp_summary_path = f"{PAIRS_TMP_PREFIX}_postfilter_summary.json"
    if os.path.exists(tmp_summary_path):
        try:
            shutil.move(tmp_summary_path, POSTFILTER_SUMMARY)
        except Exception:
            logging.warning("[PIPELINE] Could not rename temp postfilter summary %s -> %s", tmp_summary_path, POSTFILTER_SUMMARY)

    logging.info(f"[PIPELINE] Replaced {original_empty_input_count} empty inputs with [CONTEXT CONTINUES].")

    logging.info(f"[PIPELINE] Final output written to results.jsonl ({len(final_pairs)} kept).")
    if KEEP_DEBUG_OUTPUTS:
        logging.info("[PIPELINE] Debug outputs kept (merged, clean, overlap).")
    else:
        logging.info("[PIPELINE] Debug outputs removed (only results.jsonl kept).")

    # Clean up debug files if not requested
    if not KEEP_DEBUG_OUTPUTS:
        for fn in [PAIRS_MERGED_OUTPUT, PAIRS_META_OUTPUT, PAIRS_CLEAN_OUTPUT, f"{PAIRS_TMP_PREFIX}_filtered.jsonl"]:
            try:
                if os.path.exists(fn):
                    os.remove(fn)
                    logging.info("[CLEANUP] Removed %s", fn)
            except Exception as e:
                logging.warning("[CLEANUP] Could not remove %s: %s", fn, e)
    else:
        logging.info("[PIPELINE] Debug outputs kept (merged/clean/meta).")
    start_time = mark_step("Build training pairs", start_time)

    # ==========================
    # FINAL PIPELINE SUMMARY
    # ==========================
    logging.info("========================================")
    logging.info("📊 FINAL DATASET SUMMARY")
    logging.info("========================================")
    logging.info(f"🎙️  Target identified as {best_speaker} (score = {best_score:.4f})")
    logging.info(f"📝  Words attributed to Target : {target_word_count}")
    logging.info(f"💬  Raw pairs (pre-merge)      : {len(pairs)}")
    logging.info(f"🔗  After merging (gap={args.merge_gap}s)   : {len(merged_pairs)}")
    logging.info(f"🚫  Overlaps dropped           : {overlap_summary['overlapped_words']} words "
                 f"({overlap_summary['overlap_rate']:.1%})")
    logging.info(f"🧹  Clean pairs before filter  : {overlap_summary['pairs_clean_count']}")
    logging.info(f"⚖️  Filtering rules            : min_words={args.min_words}, min_conf={args.min_conf}")
    logging.info(f"✅  Final usable pairs         : {len(final_pairs)}")
    logging.info(f"📭  Pairs with empty input     : {original_empty_input_count} ({original_empty_input_pct:.1f}%) → replaced with [CONTEXT CONTINUES]")
    logging.info(f"❌  Dropped by length          : {post_summary['counts']['dropped_by_length']}")
    logging.info(f"❌  Dropped by confidence      : {post_summary['counts']['dropped_by_conf']}")
    logging.info(f"❌  Noise filter               : {dropped_non_target} words dropped by non-target word filter")
    logging.info(f"🚩  Flagged for review         : {flagged_count} Please review in pairs_flagged.jsonl")
    logging.info("========================================")

    print_timings_summary()
    
def main():
    try:
        run_pipeline()
    except Exception as e:
        logging.exception("Pipeline failed")

if __name__ == "__main__":
    main()
