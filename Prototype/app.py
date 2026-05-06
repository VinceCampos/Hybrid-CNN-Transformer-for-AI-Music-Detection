import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import os
import math
import warnings
import tempfile
import numpy as np
import json
import uuid
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import ASTModel, pipeline
import subprocess

warnings.filterwarnings("ignore")

# ── Custom JSON encoder for numpy/torch types ─────────────────────────────────
import json as _json

class _SafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.bool_):     return bool(obj)
        try:
            import torch as _t
            if isinstance(obj, _t.Tensor): return obj.detach().cpu().tolist()
        except ImportError:
            pass
        return super().default(obj)

# ── Config ────────────────────────────────────────────────────────────────────
# Base directory = folder where app.py lives, regardless of where Flask is launched from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE     = 16000
HOP_LENGTH      = 160
FRAMES_NEEDED   = 1024
SAMPLES_NEEDED  = FRAMES_NEEDED * HOP_LENGTH
DURATION_SEC    = SAMPLES_NEEDED / SAMPLE_RATE

CHECKPOINT_PATH      = os.path.join(BASE_DIR, "best_model.pth")
device               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTS       = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
SAVE_CONVERTED_WAV   = True
CONVERTED_WAV_DIR    = os.path.join(BASE_DIR, "converted_wavs")
REPORTS_FILE         = os.path.join(BASE_DIR, "misclassification_reports.jsonl")
ANALYSIS_REPORTS_DIR = os.path.join(BASE_DIR, "reports")   # <── saved next to app.py

# Create folders on startup
os.makedirs(CONVERTED_WAV_DIR,    exist_ok=True)
os.makedirs(ANALYSIS_REPORTS_DIR, exist_ok=True)

PROB_FLOOR        = 3.0
PROB_CEILING      = 97.0

NEURAL_WEIGHT = 0.85
ACOUSTIC_WEIGHT = 0.15

AI_THRESHOLD = 60
HUMAN_THRESHOLD = 45

# XAI deep config
XAI_IG_STEPS       = 50
XAI_LIME_SEGS_T    = 8
XAI_LIME_SEGS_F    = 4
XAI_LIME_SAMPLES   = 64
XAI_LIME_ALPHA     = 1e-3

CODEC_HF_RATIO_THRESHOLD    = 0.02

GENRE_MODEL_PRIMARY   = "dima806/music_genres_classification"
GENRE_MODEL_SECONDARY = "mtg-upf/discogs-maest-10s-pw-129e"
GENRE_SAMPLE_RATE     = 16000
GENRE_WIN_PRIMARY     = 10
GENRE_WIN_SECONDARY   = 10
GENRE_N_WINDOWS       = 3
GENRE_TOP_SUBGENRES   = 3

# ── Instrument detection config ───────────────────────────────────────────────
INSTRUMENT_MODEL      = "mtg-upf/music-audio-tagging-mtat-msd-musicnn"
INSTRUMENT_SAMPLE_RATE = 16000
INSTRUMENT_WIN_SEC    = 10
INSTRUMENT_N_WINDOWS  = 3
INSTRUMENT_MIN_SCORE  = 0.08   # minimum tag score to surface as detected

# Instrument tags present in the musicnn model vocabulary
INSTRUMENT_TAGS = {
    "guitar":       "Guitar",
    "electric guitar": "Electric Guitar",
    "bass guitar":  "Bass Guitar",
    "piano":        "Piano",
    "keyboard":     "Keyboard/Synth",
    "drums":        "Drums",
    "drum":         "Drums",
    "violin":       "Violin",
    "strings":      "Strings",
    "trumpet":      "Trumpet",
    "saxophone":    "Saxophone",
    "flute":        "Flute",
    "synthesizer":  "Synthesizer",
    "synth":        "Synthesizer",
    "organ":        "Organ",
    "choir":        "Choir/Vocals",
    "vocals":       "Vocals",
    "voice":        "Vocals",
    "cello":        "Cello",
    "bass":         "Bass",
    "percussion":   "Percussion",
    "banjo":        "Banjo",
    "ukulele":      "Ukulele",
    "harmonica":    "Harmonica",
}

# Instruments that are intrinsically synthesized/electronic — lower AI-suspicion bar
ELECTRONIC_INSTRUMENTS = {"Keyboard/Synth", "Synthesizer", "Organ"}

# Per-instrument acoustic thresholds that indicate AI synthesis
# Format: {instrument_name: {feature: (lo, hi, direction)}}
# direction "high" = high value → AI, "low" = low value → AI
INSTRUMENT_AI_HINTS = {
    "Drums":          {"spectral_flatness": (0.15, 1.0, "high"), "beat_regularity": (0.92, 1.0, "high")},
    "Guitar":         {"harmonic_ratio": (0.88, 1.0, "high"),    "noise_floor": (0.0, 0.0003, "low")},
    "Electric Guitar":{"harmonic_ratio": (0.88, 1.0, "high"),    "noise_floor": (0.0, 0.0003, "low"),
                       "zero_crossing_rate": (0.0, 0.02, "low")},  # real elec guitar has natural ZCR from distortion
    "Bass Guitar":    {"harmonic_ratio": (0.90, 1.0, "high"),    "noise_floor": (0.0, 0.0002, "low")},
    "Piano":          {"harmonic_ratio": (0.90, 1.0, "high"),    "spectral_flatness": (0.0, 0.03, "low")},
    "Strings":        {"spectral_flatness": (0.0, 0.025, "low"), "pitch_stability": (0.95, 1.0, "high")},
    "Vocals":         {"spectral_flatness": (0.18, 1.0, "high"), "pitch_stability": (0.96, 1.0, "high")},
}


GTZAN_PARENT = {
    "blues":"Blues","classical":"Classical","country":"Country","disco":"Electronic",
    "hiphop":"Hip-Hop","jazz":"Jazz","metal":"Metal","pop":"Pop","reggae":"Reggae","rock":"Rock",
}
DISCOGS_PARENT_ALIAS = {
    "Funk / Soul":"R&B","Hip Hop":"Hip-Hop","Latin":"World","Stage & Screen":"Soundtrack",
    "Non-Music":None,"Children's":None,"Brass & Military":None,
}

def parse_discogs_label(label):
    parent, sub = (label.split("---",1) if "---" in label else (label, None))
    parent = parent.strip(); sub = sub.strip() if sub else None
    alias  = DISCOGS_PARENT_ALIAS.get(parent)
    if alias is None and parent in DISCOGS_PARENT_ALIAS: return None, None
    return (alias or parent), sub

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# Use safe JSON encoder so numpy/torch scalars never cause serialization errors
try:
    from flask.json.provider import DefaultJSONProvider
    class _SafeProvider(DefaultJSONProvider):
        def dumps(self, obj, **kw):
            kw.setdefault("cls", _SafeEncoder)
            return _json.dumps(obj, **kw)
        def loads(self, s, **kw):
            return _json.loads(s, **kw)
    app.json_provider_class = _SafeProvider
    app.json = _SafeProvider(app)
except ImportError:
    # Flask < 2.2 fallback
    app.json_encoder = _SafeEncoder

# ── Model ─────────────────────────────────────────────────────────────────────
class HybridASTDetector(nn.Module):
    def __init__(self, ast_backbone):
        super().__init__()
        self.ast = ast_backbone
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),
            nn.Conv2d(16,1,3,padding=1),nn.BatchNorm2d(1),nn.ReLU(),nn.MaxPool2d(2))
        self.residual_downsample = nn.Sequential(nn.Conv2d(1,1,1),nn.MaxPool2d(4))
        self.classifier = nn.Sequential(
            nn.Linear(768*2,512),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))

    def forward(self, x):
        r = x.unsqueeze(1)
        c = self.cnn_block2(self.cnn_block1(r))
        res = self.residual_downsample(r)
        if res.shape != c.shape:
            res = torch.nn.functional.interpolate(res, size=c.shape[2:], mode="bilinear", align_corners=False)
        c = (c + res).squeeze(1)
        if c.shape[-1] != 1024 or c.shape[-2] != 128:
            c = torch.nn.functional.interpolate(c.unsqueeze(1),(128,1024),mode="bilinear",align_corners=False).squeeze(1)
        h = self.ast(c).last_hidden_state
        return self.classifier(torch.cat([h[:,0,:], h[:,1:,:].mean(1)], dim=1))


def get_genre_adjustment(genre_name):
    """Return additive adjustment for different genres (reduced impact)."""
    genre_adjustments = {
        "Electronic": -3.0,
        "Hip-Hop": -2.0,
        "Pop": -2.0,
        "Metal": 1.0,
        "Classical": 1.0,
        "Jazz": 1.0,
        "Rock": 0.0,
        "Blues": 0.5,
        "Country": 0.0,
        "Reggae": -1.0,
        "R&B": -1.0,
        "World": 0.5,
    }
    return genre_adjustments.get(genre_name, 0.0)

def compute_human_likelihood_score(feat_dict):
    """
    Compute a human-likelihood score (0-100) based on acoustic features.
    Used for display only, not for probability adjustment.
    """
    beat_reg = feat_dict["beat_regularity"]
    timing_score = 100 if beat_reg < 0.60 else (80 if beat_reg < 0.70 else (60 if beat_reg < 0.80 else (40 if beat_reg < 0.90 else 20)))
    
    pitch_stab = feat_dict["pitch_stability"]
    pitch_score = 100 if pitch_stab < 0.60 else (80 if pitch_stab < 0.70 else (60 if pitch_stab < 0.80 else (40 if pitch_stab < 0.90 else 20)))
    
    dyn_range = feat_dict["dynamic_range"]
    dynamic_score = 100 if dyn_range > 0.06 else (80 if dyn_range > 0.04 else (60 if dyn_range > 0.03 else (40 if dyn_range > 0.02 else 20)))
    
    noise = feat_dict["noise_floor"]
    noise_score = 100 if noise > 0.002 else (80 if noise > 0.001 else (60 if noise > 0.0005 else (40 if noise > 0.0002 else 20)))
    
    harmonic = feat_dict["harmonic_ratio"]
    harmonic_score = 100 if harmonic < 0.55 else (80 if harmonic < 0.65 else (60 if harmonic < 0.75 else (40 if harmonic < 0.85 else 20)))
    
    human_score = (timing_score * 0.25 + pitch_score * 0.25 + 
                   dynamic_score * 0.20 + noise_score * 0.15 + harmonic_score * 0.15)
    
    return human_score

# ══════════════════════════════════════════════════════════════════════════════
# ── Weighted Segment Analysis ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _find_chorus_by_energy(waveform, total_duration_sec, clip_sec):
    """Find the highest-energy region as a proxy for the chorus."""
    dur = total_duration_sec
    if dur <= clip_sec * 1.5:
        # Track is too short to meaningfully find a chorus — use middle
        mid = max(0.0, dur / 2.0 - clip_sec / 2.0)
        return mid
    
    arr = waveform.squeeze().numpy()
    hop_samples = int(SAMPLE_RATE * 0.5)  # 0.5s hop for energy envelope
    frame_len = int(SAMPLE_RATE * 1.0)     # 1s frames
    
    energies = []
    for start in range(0, len(arr) - frame_len, hop_samples):
        frame = arr[start:start + frame_len]
        energies.append(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
    
    if len(energies) < 3:
        return max(0.0, dur / 2.0 - clip_sec / 2.0)
    
    energies = np.array(energies)
    
    # Smooth with a moving average to find sustained loud regions (not transient peaks)
    kernel_size = min(7, len(energies))
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energies, kernel, mode='same')
    else:
        smoothed = energies
    
    # Exclude first and last 10% of the track to avoid intro/outro
    margin_frames = max(1, int(len(smoothed) * 0.10))
    search_region = smoothed[margin_frames:-margin_frames] if margin_frames < len(smoothed) // 2 else smoothed
    best_idx = int(np.argmax(search_region)) + margin_frames
    
    # Convert back to seconds
    chorus_start_sec = best_idx * 0.5
    # Center the clip on the peak
    chorus_start_sec = max(0.0, chorus_start_sec - clip_sec / 2.0)
    # Ensure we don't go past the end
    if chorus_start_sec + clip_sec > dur:
        chorus_start_sec = max(0.0, dur - clip_sec)
    
    return chorus_start_sec


def prepare_weighted_audio_analysis(waveform, total_duration_sec):
    clip_sec = DURATION_SEC
    dur = total_duration_sec

    segments = []

    # ── Intro (first clip_sec of the track, moderate weight) ──
    if dur > clip_sec:
        intro_dur = min(clip_sec, dur)
        intro = waveform[:, :int(intro_dur * SAMPLE_RATE)]
        segments.append(("intro", intro, 1.2, 0.0, intro_dur))
    else:
        # Track too short for separate intro — use full clip
        full_dur = min(clip_sec, dur)
        full = waveform[:, :int(full_dur * SAMPLE_RATE)]
        segments.append(("intro", full, 1.0, 0.0, full_dur))

    # ── Chorus (energy-based detection, highest weight) ──
    if dur > clip_sec:
        chorus_start = _find_chorus_by_energy(waveform, dur, clip_sec)
        chorus_start_samples = int(chorus_start * SAMPLE_RATE)
        chorus_dur = min(clip_sec, dur - chorus_start)
        chorus_end_samples = chorus_start_samples + int(chorus_dur * SAMPLE_RATE)
        chorus = waveform[:, chorus_start_samples:chorus_end_samples]
        segments.append(("chorus", chorus, 1.5, chorus_start, chorus_start + chorus_dur))

    # ── Pre-Chorus (clip_sec before the chorus start, if room exists) ──
    if dur > clip_sec * 2:
        # Find chorus segment (skip intro)
        chorus_seg = None
        for s in segments:
            if s[0] == "chorus":
                chorus_seg = s
                break
        if chorus_seg:
            chorus_start_sec = chorus_seg[3]
            pre_chorus_end = chorus_start_sec          # ends where chorus begins
            pre_chorus_start = pre_chorus_end - clip_sec
            if pre_chorus_start >= 1.0:               # enough room before chorus
                pre_chorus_start_samples = int(pre_chorus_start * SAMPLE_RATE)
                pre_chorus_end_samples   = int(pre_chorus_end   * SAMPLE_RATE)
                pre_chorus = waveform[:, pre_chorus_start_samples:pre_chorus_end_samples]
                segments.append(("pre-chorus", pre_chorus, 1.3, pre_chorus_start, pre_chorus_end))

    # ── Verse 1 (~30% mark, avoids chorus and intro overlap) ──
    if dur > clip_sec * 2:
        chorus_start_sec = None
        for s in segments:
            if s[0] == "chorus":
                chorus_start_sec = s[3]
                break
        verse1_target = dur * 0.30
        if chorus_start_sec and abs(verse1_target - chorus_start_sec) < clip_sec:
            verse1_target = dur * 0.20
        verse1_start = max(1.0, verse1_target)
        verse1_start = min(verse1_start, dur - clip_sec)
        # Avoid overlapping with intro
        if verse1_start < clip_sec:
            verse1_start = clip_sec
        verse1_start_samples = int(verse1_start * SAMPLE_RATE)
        verse1_end_samples = verse1_start_samples + int(clip_sec * SAMPLE_RATE)
        verse1 = waveform[:, verse1_start_samples:verse1_end_samples]
        segments.append(("verse", verse1, 1.2, verse1_start, verse1_start + clip_sec))

    # ── Verse 2 (~65% mark, avoids chorus and verse 1 overlap) ──
    if dur > clip_sec * 3:
        chorus_start_sec = None
        for s in segments:
            if s[0] == "chorus":
                chorus_start_sec = s[3]
                break
        verse2_target = dur * 0.65
        # Avoid overlapping chorus or verse 1
        taken_starts = [s[3] for s in segments]
        for taken in taken_starts:
            if abs(verse2_target - taken) < clip_sec:
                verse2_target = dur * 0.55
                break
        # Final safety clamp
        verse2_start = max(1.0, verse2_target)
        verse2_start = min(verse2_start, dur - clip_sec)
        # Only add if sufficiently far from existing segments
        if all(abs(verse2_start - s[3]) >= clip_sec * 0.75 for s in segments):
            verse2_start_samples = int(verse2_start * SAMPLE_RATE)
            verse2_end_samples = verse2_start_samples + int(clip_sec * SAMPLE_RATE)
            verse2 = waveform[:, verse2_start_samples:verse2_end_samples]
            segments.append(("verse 2", verse2, 1.1, verse2_start, verse2_start + clip_sec))

    # ── Outro (~clip_sec before the end, lower weight) ──
    if dur > clip_sec * 2:
        outro_start = max(0.0, dur - clip_sec)
        # Only add if sufficiently far from existing segments
        if all(abs(outro_start - s[3]) >= clip_sec * 0.75 for s in segments):
            outro_start_samples = int(outro_start * SAMPLE_RATE)
            outro = waveform[:, outro_start_samples:]
            actual_dur = outro.shape[1] / SAMPLE_RATE
            segments.append(("outro", outro, 0.7, outro_start, outro_start + actual_dur))

    # Remove duplicate intro if track is short and we already have a full intro
    # (This prevents having both intro and full track when dur < clip_sec)
    if dur < clip_sec and len(segments) > 1:
        segments = [segments[0]]  # Keep only the intro/full track

    # Return in chronological order
    segments.sort(key=lambda s: s[3])

    return segments

def analyze_weighted_segments(waveform, total_duration_sec, model, mel_transform, device):
    segments = prepare_weighted_audio_analysis(waveform, total_duration_sec)
    
    segment_results = []
    
    for seg_name, seg_waveform, weight, start_sec, end_sec in segments:
        if seg_waveform.shape[1] < SAMPLES_NEEDED:
            seg_waveform = torch.nn.functional.pad(seg_waveform, (0, SAMPLES_NEEDED - seg_waveform.shape[1]))
        elif seg_waveform.shape[1] > SAMPLES_NEEDED:
            seg_waveform = seg_waveform[:, :SAMPLES_NEEDED]
        
        mel_input = mel_from_chunk(seg_waveform, mel_transform)
        
        with torch.no_grad():
            raw_logit = float(model(mel_input.to(device)).squeeze().item())
            neural_prob = torch.sigmoid(torch.tensor(raw_logit)).item()
        
        chunk_feat = compute_chunk_features(seg_waveform)
        acoustic_prob = acoustic_composite_score(chunk_feat) / 100.0
        human_score = compute_human_likelihood_score(chunk_feat)
        
        # Confidence: higher when the model is more decisive (further from 0.5)
        confidence = 0.5 + min(0.5, abs(neural_prob - 0.5))
        
        # Corroboration: does the human acoustic analysis agree with neural direction?
        neural_says_human = neural_prob < 0.5
        acoustic_says_human = human_score > 60
        corroborated = neural_says_human == acoustic_says_human

        evidence_quality = confidence * (1.15 if corroborated else 0.85)

        segment_results.append({
            "name": seg_name,
            "weight": weight,
            "neural_prob": neural_prob,
            "acoustic_prob": acoustic_prob,
            "human_score": human_score,
            "confidence": confidence,
            "evidence_quality": evidence_quality,
            "corroborated": corroborated,
            "raw_logit": raw_logit,
            "features": chunk_feat,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2)
        })

    # ── Aggregation Strategy ──
    # Weight by: segment importance weight × neural confidence × evidence quality
    # This naturally favors the chorus (weight=1.5) and segments where
    # neural and acoustic signals agree
    neural_probs = [s["neural_prob"] for s in segment_results]
    n_segments = len(segment_results)
    
    for seg in segment_results:
        seg["effective_weight"] = seg["weight"] * seg["evidence_quality"]
    
    total_weight = sum(s["effective_weight"] for s in segment_results)
    
    if total_weight > 0:
        weighted_neural = sum(s["effective_weight"] * s["neural_prob"] for s in segment_results) / total_weight
        weighted_acoustic = sum(s["effective_weight"] * s["acoustic_prob"] for s in segment_results) / total_weight
    else:
        weighted_neural = np.mean(neural_probs)
        weighted_acoustic = np.mean([s["acoustic_prob"] for s in segment_results])
    
    # ── High disagreement detection ──
    neural_std = np.std(neural_probs) if len(neural_probs) > 1 else 0.0
    
    # When segments strongly disagree (std > 0.20), check if any segment
    # has BOTH high confidence AND acoustic corroboration — that's a stronger 
    # signal than a segment that only has confidence
    if neural_std > 0.20 and n_segments >= 3:
        corroborated_segs = [s for s in segment_results if s["corroborated"]]
        uncorroborated_segs = [s for s in segment_results if not s["corroborated"]]
        
        if corroborated_segs and uncorroborated_segs:
            # Segments with neural-acoustic agreement are more trustworthy
            # Boost their contribution
            corr_neural = np.mean([s["neural_prob"] for s in corroborated_segs])
            blend_factor = min(0.35, (neural_std - 0.20) * 1.5)
            weighted_neural = weighted_neural * (1 - blend_factor) + corr_neural * blend_factor
        else:
            # All segments are either corroborated or uncorroborated — 
            # lean toward the most confident one
            best_conf_seg = max(segment_results, key=lambda s: s["confidence"])
            blend_factor = min(0.25, (neural_std - 0.20) * 1.0)
            weighted_neural = weighted_neural * (1 - blend_factor) + best_conf_seg["neural_prob"] * blend_factor
    
    # ── Human-score safety check ──
    # If the MOST human-sounding segment (high human_score + low neural) exists 
    # and is being outvoted, apply a conservatism pull toward 50%
    # This addresses the scenario where heavily-produced intros/outros 
    # overpower a clearly human vocal section
    if n_segments >= 3 and neural_std > 0.25:
        min_neural_seg = min(segment_results, key=lambda s: s["neural_prob"])
        if min_neural_seg["human_score"] > 70 and min_neural_seg["neural_prob"] < 0.25:
            # Strong human signal in at least one segment — don't be overconfident about AI
            if weighted_neural > 0.65:
                pull = min(0.12, (weighted_neural - 0.65) * 0.5)
                weighted_neural -= pull
    
    best_segment = max(segment_results, key=lambda s: s["evidence_quality"] * s["weight"])
    
    return weighted_neural, weighted_acoustic, segment_results, best_segment

# ══════════════════════════════════════════════════════════════════════════════
# ── Calibrated Probability (NEURAL DOMINANT) ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_calibrated_probability(neural_prob, acoustic_prob, feat_dict, genre_result=None):
    """
    Neural network is the primary signal. Acoustic adjustments are small and
    agreement-scaled — they cannot flip a confident neural prediction.
    """
    base_prob = (neural_prob * NEURAL_WEIGHT + acoustic_prob * ACOUSTIC_WEIGHT) * 100
    
    adjustment = 0.0
    human_reasons = []
    
    # ── Individual feature adjustments (small, no double-counting) ──
    
    # Beat regularity
    beat_reg = feat_dict["beat_regularity"]
    if beat_reg < 0.60:
        adjustment += -(0.60 - beat_reg) * 8
        human_reasons.append("natural timing variations")
    elif beat_reg > 0.93:
        adjustment += (beat_reg - 0.93) * 12
    
    # Pitch stability
    pitch_stab = feat_dict["pitch_stability"]
    if pitch_stab < 0.60:
        adjustment += -(0.60 - pitch_stab) * 6
        human_reasons.append("natural pitch variation")
    elif pitch_stab > 0.93:
        adjustment += (pitch_stab - 0.93) * 10
    
    # Dynamic range
    dyn_range = feat_dict["dynamic_range"]
    if dyn_range > 0.06:
        adjustment -= 1.5
        human_reasons.append("good dynamic range")
    elif dyn_range < 0.02:
        adjustment += 1.5
    
    # Noise floor
    noise = feat_dict["noise_floor"]
    if noise > 0.002:
        adjustment -= min(2.0, noise * 500)
        human_reasons.append("natural background presence")
    elif noise < 0.0001:
        adjustment += 1.5
    
    # Harmonic ratio
    harmonic = feat_dict["harmonic_ratio"]
    if harmonic > 0.90:
        adjustment += (harmonic - 0.90) * 15
    elif harmonic < 0.50:
        adjustment -= (0.50 - harmonic) * 8
        human_reasons.append("organic harmonic content")
    
    # ── Genre adjustment (small) ──
    if genre_result and genre_result.get("top"):
        genre_name = genre_result["top"]["label"]
        genre_adj = get_genre_adjustment(genre_name)
        adjustment += genre_adj

    # ── Hard cap: adjustments can't flip a confident prediction ──
    # If neural is >65% or <35%, cap adjustment to prevent flipping past 50%
    neural_pct = neural_prob * 100
    if neural_pct > 65:
        # Don't let adjustment push below 50
        adjustment = max(adjustment, -(neural_pct - 52))
    elif neural_pct < 35:
        # Don't let adjustment push above 50
        adjustment = min(adjustment, (48 - neural_pct))
    
    # Global cap
    adjustment = max(-8.0, min(8.0, adjustment))
    
    calibrated = base_prob + adjustment
    calibrated = max(PROB_FLOOR, min(PROB_CEILING, calibrated))
    
    return calibrated, adjustment, human_reasons

# ── XAI deep ─────────────────────────────────────────────────────────────────

# ── GRAD-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM targeting the last Conv2d in cnn_block2 (the Conv2d(16→1) layer,
    index 3 in the Sequential). This is the final spatial feature map before
    the residual add and AST handoff — the ideal hook point for visualising
    which time-frequency regions drove the classification decision.
    """
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        # cnn_block2 = [Conv2d(32,16), BN, ReLU, Conv2d(16,1), BN, ReLU, MaxPool2d]
        # Index 3 is the last Conv2d before BN/ReLU/Pool
        target_layer = model.cnn_block2[3]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, mel_input):
        """
        Returns a dict with:
          - heatmap: 2-D list (64 freq bins × 256 time frames), values in [0,1]
          - top_regions: list of top-5 time-frequency regions by Grad-CAM weight
          - raw_cam_shape: original spatial shape before upsampling
        """
        self.model.eval()
        x = mel_input.clone().float().requires_grad_(True)

        # Forward — do NOT use torch.no_grad() here
        logit = self.model(x).squeeze()
        self.model.zero_grad()
        logit.backward(torch.ones_like(logit))

        if self.gradients is None or self.activations is None:
            return {"error": "Grad-CAM hooks did not fire — check target layer index."}

        # Global-average-pool the gradients over spatial dims → per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam     = F.relu(cam)

        raw_h, raw_w = cam.shape[2], cam.shape[3]

        # Upsample to mel shape (128 freq × 1024 time)
        cam_up = F.interpolate(cam, size=(128, 1024), mode='bilinear', align_corners=False)
        cam_np = cam_up.squeeze().detach().cpu().float().numpy()

        # Normalise to [0, 1]
        lo, hi = cam_np.min(), cam_np.max()
        cam_norm = np.zeros_like(cam_np) if (hi - lo) < 1e-12 else (cam_np - lo) / (hi - lo)

        # Top-5 regions: find local maxima in the heatmap
        # cam_norm shape: (128 freq_bins, 1024 time_frames)
        # axis=0 averages over freq → (1024,) time importance
        # axis=1 averages over time → (128,)  freq importance
        time_imp = cam_norm.mean(axis=0)   # (1024,) — time frame importance
        freq_imp = cam_norm.mean(axis=1)   # (128,)  — freq bin importance
        top_t = list(np.argsort(time_imp)[::-1][:5])   # top time frames (0-1023)
        top_f = list(np.argsort(freq_imp)[::-1][:5])   # top freq bins  (0-127)
        top_regions = [
            {
                "frame_idx": int(t),
                "time_sec":  round(t * HOP_LENGTH / SAMPLE_RATE, 3),
                "bin_idx":   int(f),
                "freq_hz":   round(700 * (10 ** ((2595 * math.log10(1 + 0/700) +
                              (2595 * math.log10(1 + (SAMPLE_RATE/2)/700) -
                               2595 * math.log10(1 + 0/700)) * f / 127) / 2595) - 1), 1),
                "importance": round(float(cam_norm[f, t]), 4),  # cam_norm[freq, time]
            }
            for t, f in zip(top_t, top_f)
        ]

        # ── Vectorized average-pool downsample (no loops) ──
        # 128×1024 → 64×256 using reshape+mean: 8× smaller, preserves local maxima better
        # than strided indexing while being fully vectorized.
        DS_F, DS_T = 64, 256
        blk_f = cam_norm.shape[0] // DS_F   # 2
        blk_t = cam_norm.shape[1] // DS_T   # 4
        # Crop to exact multiple then reshape-mean (all numpy, no Python loops)
        cam_crop = cam_norm[:DS_F * blk_f, :DS_T * blk_t]
        cam_small = cam_crop.reshape(DS_F, blk_f, DS_T, blk_t).mean(axis=(1, 3))
        # Re-normalise after pooling
        lo2, hi2 = cam_small.min(), cam_small.max()
        cam_small = np.zeros_like(cam_small) if (hi2 - lo2) < 1e-12 else (cam_small - lo2) / (hi2 - lo2)

        # ── Derive interpretation signals ──
        peak_importance  = float(time_imp.max())
        peak_time_sec    = float(top_t[0] * HOP_LENGTH / SAMPLE_RATE) if top_t else 0.0
        peak_freq_hz     = float(top_regions[0]["freq_hz"]) if top_regions else 0.0
        time_spread_sec  = float((max(top_t) - min(top_t)) * HOP_LENGTH / SAMPLE_RATE) if len(top_t) > 1 else 0.0
        freq_spread_hz   = float(max(r["freq_hz"] for r in top_regions) -
                                 min(r["freq_hz"] for r in top_regions)) if len(top_regions) > 1 else 0.0

        # Classify frequency zone
        if peak_freq_hz < 300:
            freq_zone = "sub-bass"
        elif peak_freq_hz < 1000:
            freq_zone = "vocal/instrument (bass-mid)"
        elif peak_freq_hz < 4000:
            freq_zone = "vocal/instrument (mid-high)"
        elif peak_freq_hz < 6000:
            freq_zone = "upper harmonic"
        else:
            freq_zone = "synthetic/artifact (high)"

        # Concentration score: how tightly clustered are the hot spots?
        # Low spread + high peak = concentrated (AI-like)
        # High spread + low peak = diffuse (human-like)
        concentration = float(np.clip(peak_importance / max(time_spread_sec + 0.1, 0.1), 0, 10))
        concentration_label = (
            "highly concentrated — single-point artifact (strong AI signal)"
            if concentration > 5 else
            "moderately concentrated — localised pattern"
            if concentration > 2 else
            "diffuse — spread across the track (human-like)"
        )

        # AI vs Human interpretation
        ai_signals   = []
        human_signals = []

        if peak_importance > 0.65:
            ai_signals.append(f"High peak activation ({peak_importance:.0%}) — network found a strong AI fingerprint")
        else:
            human_signals.append(f"Low peak activation ({peak_importance:.0%}) — no dominant AI fingerprint found")

        if time_spread_sec < 0.5 and len(top_t) > 1:
            ai_signals.append(f"Hyper-concentrated hotspot ({time_spread_sec:.3f}s window) — typical of AI generation artifacts")
        elif time_spread_sec > 1.0:
            human_signals.append(f"Activation spread over {time_spread_sec:.1f}s — organic variation across the track")

        if freq_zone == "synthetic/artifact (high)":
            ai_signals.append(f"Peak at {peak_freq_hz:.0f} Hz — above natural vocal/instrument range, typical of AI synthesis artifacts")
        elif freq_zone == "upper harmonic":
            human_signals.append(f"Peak at {peak_freq_hz:.0f} Hz — upper harmonic range, natural for instruments and vocals")
        elif "vocal" in freq_zone:
            human_signals.append(f"Peak at {peak_freq_hz:.0f} Hz — within natural vocal/instrument range")

        if freq_spread_hz > 3000:
            human_signals.append(f"Wide frequency spread ({freq_spread_hz:.0f} Hz) — attention distributed across natural harmonic range")
        elif freq_spread_hz < 500 and len(top_regions) > 1:
            ai_signals.append(f"Narrow frequency cluster ({freq_spread_hz:.0f} Hz) — network locked onto a specific synthetic band")

        # Overall pattern verdict — Grad-CAM signals + final verdict as tiebreaker
        ai_score  = len(ai_signals)
        hum_score = len(human_signals)

        if ai_score > hum_score and ai_score >= 1:
            pattern_verdict = "AI-like pattern"
            pattern_color   = "red"
        elif hum_score > ai_score and hum_score >= 1:
            pattern_verdict = "Human-like pattern"
            pattern_color   = "green"
        else:
            pattern_verdict = "Ambiguous pattern"
            pattern_color   = "amber"

        # Plain-English summary sentence for the UI
        if pattern_verdict == "AI-like pattern":
            plain_summary = (
                f"The network's attention was {concentration_label.split('—')[0].strip()} "
                f"around {peak_freq_hz:.0f} Hz ({freq_zone}), spanning only {time_spread_sec:.2f}s "
                f"— a signature consistent with AI-generated synthesis artifacts."
            )
        elif pattern_verdict == "Human-like pattern":
            plain_summary = (
                f"Activation was broadly spread across {time_spread_sec:.1f}s and "
                f"{freq_spread_hz:.0f} Hz of frequency range, centred in the {freq_zone} "
                f"— typical of organic, human-performed audio."
            )
        else:
            plain_summary = (
                f"Mixed signals: peak activation at {peak_freq_hz:.0f} Hz ({freq_zone}) "
                f"over {time_spread_sec:.2f}s. Both AI and human patterns are present — "
                f"the overall verdict relies on additional acoustic evidence."
            )

        # Frequency zone explanation for non-experts
        freq_zone_explanations = {
            "sub-bass":                   "Below 300 Hz — very low rumble/bass region. Anomalies here often relate to low-frequency generation artifacts.",
            "vocal/instrument (bass-mid)": "300–1 000 Hz — the fundamental range of most vocals and instruments. Activity here is expected in natural music.",
            "vocal/instrument (mid-high)": "1–4 kHz — upper harmonics and vocal clarity range. AI models sometimes leave subtle patterns in this region.",
            "upper harmonic":             "4–6 kHz — air and presence band. Common in natural music; activation here is expected for instruments and vocals.",
            "synthetic/artifact (high)":  "Above 6 kHz — beyond most natural vocal/instrument energy. Strong AI activation here is a red flag for synthesis artifacts.",
        }

        return {
            "heatmap":             cam_small.tolist(),   # 64×256 — fast transfer
            "heatmap_shape":       [DS_F, DS_T],
            "top_regions":         top_regions,
            "raw_cam_shape":       [raw_h, raw_w],
            # ── Interpretation ──
            "peak_importance":     round(peak_importance, 4),
            "peak_time_sec":       round(peak_time_sec, 3),
            "peak_freq_hz":        round(peak_freq_hz, 1),
            "time_spread_sec":     round(time_spread_sec, 3),
            "freq_spread_hz":      round(freq_spread_hz, 1),
            "freq_zone":           freq_zone,
            "freq_zone_explanation": freq_zone_explanations.get(freq_zone, ""),
            "concentration":       round(concentration, 3),
            "concentration_label": concentration_label,
            "ai_signals":          ai_signals,
            "human_signals":       human_signals,
            "pattern_verdict":     pattern_verdict,
            "pattern_color":       pattern_color,
            "plain_summary":       plain_summary,
        }

def _gradcam_singleton(model):
    """Lazily create a GradCAM instance, recreating it if the model changes."""
    if (not hasattr(_gradcam_singleton, "_instance") or
            _gradcam_singleton._model_id != id(model)):
        _gradcam_singleton._instance = GradCAM(model)
        _gradcam_singleton._model_id  = id(model)
    return _gradcam_singleton._instance


def _xd_to_numpy(t):
    return t.detach().cpu().float().numpy()

def _xd_norm01(a):
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

def _xd_frame_to_sec(idx):
    return round(idx * HOP_LENGTH / SAMPLE_RATE, 3)

def _xd_bin_to_hz(idx, n_mels=128):
    mel_min = 2595 * math.log10(1 + 0 / 700)
    mel_max = 2595 * math.log10(1 + (SAMPLE_RATE / 2) / 700)
    mel_v   = mel_min + (mel_max - mel_min) * idx / (n_mels - 1)
    return round(700 * (10 ** (mel_v / 2595) - 1), 1)

def _xd_topk(arr, k=5):
    k = min(k, len(arr))
    return list(np.argsort(arr)[::-1][:k])

def _xd_saliency(mel_input, model):
    model.eval()
    x = mel_input.clone().detach().requires_grad_(True)
    model(x).squeeze().backward()
    grad = _xd_to_numpy(x.grad.abs()).squeeze(0)
    fi   = _xd_norm01(grad.mean(axis=1))
    bi   = _xd_norm01(grad.mean(axis=0))
    return {
        "heatmap":    _xd_norm01(grad).tolist(),
        "top_frames": [{"frame_idx": i, "importance": round(float(fi[i]), 4),
                        "time_sec": _xd_frame_to_sec(i)} for i in _xd_topk(fi)],
        "top_bins":   [{"bin_idx": i,   "importance": round(float(bi[i]), 4),
                        "freq_hz": _xd_bin_to_hz(i)}   for i in _xd_topk(bi)],
    }

def _xd_integrated_gradients(mel_input, model, n_steps=XAI_IG_STEPS):
    """Memory-safe Integrated Gradients — one step at a time to avoid OOM."""
    model.eval()
    x    = mel_input.float()
    base = torch.zeros_like(x)
    # Accumulate gradients as float32 numpy on CPU to keep VRAM flat
    avg_grads_np = np.zeros(_xd_to_numpy(x).shape, dtype=np.float32)
    for step in range(n_steps + 1):
        interp = (base + (step / n_steps) * (x - base)).requires_grad_(True)
        model(interp).squeeze().backward()
        avg_grads_np += _xd_to_numpy(interp.grad)
        # Free the computation graph immediately
        interp.grad = None
        if x.device.type == "cuda" and step % 10 == 0:
            torch.cuda.empty_cache()
    avg_grads_np /= (n_steps + 1)
    attrs_abs = np.abs(_xd_to_numpy(x - base) * avg_grads_np).squeeze(0)
    fi  = _xd_norm01(attrs_abs.mean(axis=1))
    bi  = _xd_norm01(attrs_abs.mean(axis=0))
    return {
        "heatmap":    _xd_norm01(attrs_abs).tolist(),
        "top_frames": [{"frame_idx": i, "importance": round(float(fi[i]), 4),
                        "time_sec": _xd_frame_to_sec(i)} for i in _xd_topk(fi)],
        "top_bins":   [{"bin_idx": i,   "importance": round(float(bi[i]), 4),
                        "freq_hz": _xd_bin_to_hz(i)}   for i in _xd_topk(bi)],
        "n_steps": n_steps,
    }

def _xd_attention_rollout(mel_input, model):
    model.eval()
    with torch.no_grad():
        ast_out = model.ast(mel_input, output_attentions=True)
    all_attns = getattr(ast_out, "attentions", None)
    if not all_attns:
        return {"heatmap": [], "rollout": [], "top_tokens": [], "n_heads": 0,
                "note": "Attention weights unavailable for this model version."}
    mats    = [_xd_to_numpy(a.squeeze(0)) for a in all_attns]
    n_heads = mats[0].shape[0]
    seq_len = mats[0].shape[-1]
    rolled = np.eye(seq_len)
    for m in mats:
        hm   = m.mean(axis=0)
        aug  = 0.5 * hm + 0.5 * np.eye(seq_len)
        aug /= aug.sum(axis=-1, keepdims=True)
        rolled = rolled @ aug
    cls_row  = _xd_norm01(rolled[0, 1:])
    top_toks = [{"token_idx": i, "importance": round(float(cls_row[i]), 4)}
                for i in _xd_topk(cls_row)]
    n_patches = len(cls_row)
    patch_t   = int(math.sqrt(n_patches * 1024 / 128))
    patch_f   = n_patches // max(patch_t, 1)
    if patch_t > 0 and patch_f > 0 and patch_t * patch_f <= n_patches:
        grid = cls_row[: patch_t * patch_f].reshape(patch_t, patch_f)
        tsr  = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
        up   = F.interpolate(tsr, size=(1024, 128), mode="bilinear", align_corners=False)
        hmap = _xd_norm01(_xd_to_numpy(up.squeeze())).tolist()
    else:
        row  = np.interp(np.linspace(0, len(cls_row) - 1, 128),
                         np.arange(len(cls_row)), cls_row)
        hmap = _xd_norm01(np.tile(row, (1024, 1))).tolist()
    return {"heatmap": hmap, "rollout": _xd_norm01(rolled).tolist(),
            "top_tokens": top_toks, "n_heads": n_heads}

def _xd_lime(mel_input, model,
             n_segs_t=XAI_LIME_SEGS_T, n_segs_f=XAI_LIME_SEGS_F,
             n_samples=XAI_LIME_SAMPLES, alpha=XAI_LIME_ALPHA):
    model.eval()
    x  = mel_input
    T, Fm = x.shape[1], x.shape[2]
    t_edges = np.linspace(0, T,  n_segs_t + 1, dtype=int)
    f_edges = np.linspace(0, Fm, n_segs_f + 1, dtype=int)
    n_segs  = n_segs_t * n_segs_f
    fill    = float(x.mean().item())
    rng    = np.random.default_rng(42)
    masks  = rng.integers(0, 2, size=(n_samples, n_segs), dtype=np.uint8)
    logits = np.zeros(n_samples, dtype=np.float32)
    with torch.no_grad():
        base_logit = float(model(x).squeeze().item())
        for i, mask in enumerate(masks):
            p = x.clone(); seg = 0
            for ti in range(n_segs_t):
                for fi in range(n_segs_f):
                    if mask[seg] == 0:
                        p[0, t_edges[ti]:t_edges[ti+1], f_edges[fi]:f_edges[fi+1]] = fill
                    seg += 1
            logits[i] = float(model(p).squeeze().item())
    X   = masks.astype(np.float32)
    y   = logits
    XtX = X.T @ X + alpha * np.eye(n_segs, dtype=np.float32)
    try:    coeffs = np.linalg.solve(XtX, X.T @ y)
    except: coeffs = np.zeros(n_segs, dtype=np.float32)
    y_hat  = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2     = float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))
    segments = []
    with torch.no_grad():
        for seg in range(n_segs):
            ti, fi = divmod(seg, n_segs_f)
            t0, t1 = int(t_edges[ti]), int(t_edges[ti+1])
            f0, f1 = int(f_edges[fi]), int(f_edges[fi+1])
            masked = x.clone()
            masked[0, t0:t1, f0:f1] = fill
            delta = round(float(base_logit - model(masked).squeeze().item()), 4)
            segments.append({
                "segment_id":         seg,
                "importance":         round(float(coeffs[seg]), 4),
                "time_start_sec":     _xd_frame_to_sec(t0),
                "time_end_sec":       _xd_frame_to_sec(t1),
                "freq_start_hz":      _xd_bin_to_hz(f0),
                "freq_end_hz":        _xd_bin_to_hz(f1),
                "masked_score_delta": delta,
            })
    segments.sort(key=lambda s: abs(s["importance"]), reverse=True)
    return {"segments": segments, "n_segments": n_segs,
            "n_samples": n_samples, "r2_score": round(r2, 4)}

def _xd_summary(sal, ig, attn, lime):
    peak_sal_sec = sal["top_frames"][0]["time_sec"] if sal.get("top_frames") else None
    peak_attn_sec = None
    if attn.get("heatmap") and len(attn["heatmap"]) > 0:
        hmap = np.array(attn["heatmap"])
        peak_attn_sec = _xd_frame_to_sec(int(np.argmax(hmap.mean(axis=1))))
    most_sus = None
    pos_segs = [s for s in lime.get("segments", []) if s["importance"] > 0]
    if pos_segs:
        b = max(pos_segs, key=lambda s: s["importance"])
        most_sus = {k: b[k] for k in
                    ("time_start_sec","time_end_sec","freq_start_hz","freq_end_hz","importance")}
    agreement = 0.5
    if sal.get("top_frames") and ig.get("top_frames"):
        diff = abs(sal["top_frames"][0]["time_sec"] - ig["top_frames"][0]["time_sec"])
        agreement = round(float(np.clip(1.0 - diff * 2 / _xd_frame_to_sec(1024), 0.0, 1.0)), 3)
    return {
        "most_suspicious_region":  most_sus,
        "peak_attention_time_sec": peak_attn_sec,
        "peak_saliency_time_sec":  peak_sal_sec,
        "method_agreement":        agreement,
    }

def compute_xai_deep(mel_input, model, device):
    model.eval()
    x = mel_input.to(device)
    with torch.no_grad():
        raw_logit = float(model(x).squeeze().item())

    def _clear():
        model.zero_grad()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {}
    try:
        _clear()
        gc = _gradcam_singleton(model)
        out["gradcam"] = gc.compute(x)
    except Exception as e: out["gradcam"] = {"error": str(e)}
    finally: _clear()

    try:
        out["saliency"] = _xd_saliency(x, model)
    except Exception as e: out["saliency"] = {"error": str(e)}
    finally: _clear()

    try:
        out["integrated_gradients"] = _xd_integrated_gradients(x, model)
    except Exception as e: out["integrated_gradients"] = {"error": str(e)}
    finally: _clear()

    try:
        out["attention"] = _xd_attention_rollout(x, model)
    except Exception as e: out["attention"] = {"error": str(e)}
    finally: _clear()

    try:
        out["lime"] = _xd_lime(x, model)
    except Exception as e: out["lime"] = {"error": str(e)}
    finally: _clear()

    out["summary"] = _xd_summary(
        out.get("saliency", {}), out.get("integrated_gradients", {}),
        out.get("attention", {}), out.get("lime", {}),
    )
    return raw_logit, out
# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_codec_compression(waveform_tensor, sr=SAMPLE_RATE):
    arr = waveform_tensor.squeeze().numpy().astype(np.float32)
    n   = len(arr)
    if n < sr // 2:
        return False, 1.0, 0.0, {"reason": "too short to evaluate"}
    fft_out  = np.fft.rfft(arr, n=min(n, 65536))
    freqs    = np.fft.rfftfreq(min(n, 65536), d=1.0 / sr)
    mag_sq   = np.abs(fft_out) ** 2
    total_energy = np.sum(mag_sq) + 1e-12
    hf_energy    = np.sum(mag_sq[freqs > 7000])
    hf_ratio     = float(hf_energy / total_energy)
    band_lo = mag_sq[(freqs >= 6000) & (freqs < 7500)]
    band_hi = mag_sq[(freqs >= 7500) & (freqs <= 8000)]
    shelf_ratio = (np.mean(band_hi) / (np.mean(band_lo) + 1e-12)) if band_lo.size and band_hi.size else 1.0
    region = mag_sq[(freqs >= 4000) & (freqs <= 8000)] + 1e-10
    region_flatness = float(np.exp(np.mean(np.log(region))) / np.mean(region))
    evidence = 0.0
    if hf_ratio < CODEC_HF_RATIO_THRESHOLD:       evidence += 0.5
    if shelf_ratio < 0.15:                         evidence += 0.25
    if region_flatness > 0.60:                     evidence += 0.15
    if hf_ratio < 0.005:                           evidence += 0.10
    is_compressed = evidence >= 0.5
    confidence    = float(np.clip(evidence, 0.0, 1.0))
    details = {
        "hf_ratio":        round(hf_ratio, 5),
        "shelf_ratio":     round(float(shelf_ratio), 4),
        "region_flatness": round(region_flatness, 4),
        "evidence_score":  round(evidence, 3),
    }
    return is_compressed, hf_ratio, confidence, details

def apply_codec_penalty(ai_probability, is_compressed, codec_confidence):
    if not is_compressed:
        return ai_probability, {}
    penalty = min(3.0, codec_confidence * 5)  # Reduced penalty
    adjusted = max(PROB_FLOOR, ai_probability - penalty)
    penalty_info = {
        "applied": True,
        "confidence": round(codec_confidence, 3),
        "penalty_amount": round(penalty, 1),
        "original_score": ai_probability,
        "adjusted_score": adjusted,
        "reason": "Lossy codec compression detected.",
    }
    return adjusted, penalty_info

# ── Genre ─────────────────────────────────────────────────────────────────────
def _sample_windows(total, win, n):
    margin = max(0, int(total*0.05)); usable = total - 2*margin - win
    if usable <= 0: return [max(0, total//2 - win//2)]
    step = usable / max(n-1, 1)
    return [min(margin+round(i*step), total-win) for i in range(n)]

def _clip_window(wav, start, win):
    clip = wav[:, start:min(wav.shape[1], start+win)].squeeze(0).numpy().astype(np.float32)
    return np.pad(clip, (0, max(0, win-clip.shape[0])))

def _run_model(pipeline_fn, waveform, win_secs, label_fn):
    if pipeline_fn is None: return []
    win = GENRE_SAMPLE_RATE * win_secs; results = []
    for start in _sample_windows(waveform.shape[1], win, GENRE_N_WINDOWS):
        try:
            raw = pipeline_fn({"raw":_clip_window(waveform,start,win),"sampling_rate":GENRE_SAMPLE_RATE},
                              top_k=50 if win_secs==GENRE_WIN_SECONDARY else None)
            results.append([e for e in (label_fn(i) for i in raw) if e])
        except Exception as e:
            print(f"  [genre] window {start}: {e}")
    return results

def _primary_label(item):
    lbl = item["label"].lower().replace(" ","").replace("-","")
    parent = GTZAN_PARENT.get(item["label"].lower(), GTZAN_PARENT.get(lbl, item["label"].title()))
    return {"label":parent,"score":float(item["score"]),"parent":parent,"sub":None}

def _secondary_label(item):
    parent, sub = parse_discogs_label(item["label"])
    if parent is None: return None
    return {"label":(f"{parent} — {sub}" if sub else parent),"score":float(item["score"]),"parent":parent,"sub":sub}

def _aggregate(windows):
    if not windows: return {"parent_scores":{},"subgenre_scores":{},"raw":[]}
    n = len(windows); pa = {}; sa = {}
    for w in windows:
        total = sum(i["score"] for i in w) or 1.0
        for i in w:
            s=i["score"]/total; p=i["parent"]; k=(p,i["sub"])
            pa[p]=pa.get(p,0)+s; sa[k]=sa.get(k,0)+s
    pt=sum(pa.values()) or 1.0; st=sum(sa.values()) or 1.0
    raw=sorted([{"label":(f"{p} — {s}" if s else p),"score":round(v/st*100,1),"parent":p,"sub":s}
                for (p,s),v in sa.items()],key=lambda x:x["score"],reverse=True)
    return {"parent_scores":{k:round(v/n/pt*100,1) for k,v in pa.items()},
            "subgenre_scores":sa,"raw":raw}

def _ensemble(pri, sec):
    pp=pri["parent_scores"]; sp=sec["parent_scores"]; sg=sec["subgenre_scores"]
    comb={p:(0.4*pp.get(p,0)+0.6*sp.get(p,0)) if pp.get(p) and sp.get(p)
            else 0.7*pp.get(p,0) or 0.7*sp.get(p,0) for p in set(pp)|set(sp)}
    pt=sum(comb.values()) or 1.0
    ppct={k:round(v/pt*100,1) for k,v in comb.items()}
    spct={}
    for (p,s),v in sg.items():
        st=sum(v2 for (p2,_),v2 in sg.items() if p2==p) or 1.0
        spct[(p,s)]=round(v/st*comb.get(p,0)/pt*100,1)
    return {"parent_pct":ppct,"subgenre_pct":spct}

def _stability(windows, top):
    scores=[]
    for w in windows:
        total=sum(i["score"] for i in w) or 1.0
        scores.append(sum(i["score"] for i in w if i.get("parent","").lower()==top.lower())/total)
    if len(scores)<2: return 0.5
    mean=np.mean(scores)
    return 0.3 if mean<0.05 else float(np.clip(1.0-np.std(scores)/(mean+1e-6),0.0,1.0))

def classify_genre(waveform):
    pw=_run_model(genre_pipeline_primary, waveform,GENRE_WIN_PRIMARY, _primary_label)
    sw=_run_model(genre_pipeline_secondary,waveform,GENRE_WIN_SECONDARY,_secondary_label)
    pa=_aggregate(pw); sa=_aggregate(sw)
    if not pa["parent_scores"] and not sa["parent_scores"]: return None
    ens=_ensemble(pa,sa); ppct=ens["parent_pct"]; spct=ens["subgenre_pct"]
    if not ppct: return None
    sorted_p=sorted(ppct.items(),key=lambda x:x[1],reverse=True)
    top,top_score=sorted_p[0]
    subs={}
    for (p,s),pct in spct.items():
        if s: subs.setdefault(p,[]).append({"label":s,"score":pct})
    for p in subs: subs[p]=sorted(subs[p],key=lambda x:x["score"],reverse=True)[:GENRE_TOP_SUBGENRES]
    top_sub=None
    if spct:
        bk=max(spct,key=lambda k:spct[k])
        if bk[1]: top_sub={"parent":bk[0],"label":bk[1],"score":round(spct[bk],1)}
    stab=_stability(pw+sw,top); sec=sorted_p[1][1] if len(sorted_p)>1 else 0.0; margin=top_score-sec
    if stab>=0.75 and margin>=15:   cn="High confidence — consistent across the track"
    elif stab>=0.50 and margin>=8:  cn="Moderate confidence — mostly consistent"
    elif margin<5:                  cn="Low confidence — genre boundary is ambiguous"
    else:                           cn="Mixed signal — track may blend multiple genres"
    sources=([f"Primary ({GENRE_MODEL_PRIMARY.split('/')[1]}): {len(pw)} windows"] if pw else [])+\
            ([f"Secondary ({GENRE_MODEL_SECONDARY.split('/')[1]}): {len(sw)} windows"] if sw else [])
    return {"top":{"label":top,"score":round(top_score,1)},"all":[{"label":p,"score":s} for p,s in sorted_p[:10]],
            "subgenres":subs,"top_subgenre":top_sub,"sources":sources,"window_count":len(pw+sw),
            "stability":round(stab*100,1),"confidence_note":cn,"primary_used":bool(pw),"secondary_used":bool(sw)}

# ── Instrument AI/real classification ────────────────────────────────────────

def _infer_instruments_acoustic(waveform, feat_dict):
    """
    Acoustic-only instrument inference — no external model required.
    Uses spectral + temporal features to estimate which instrument families
    are likely present and how 'real' they sound.
    Returns a dict of {display_name: tag_score 0-1}.
    """
    arr = waveform.mean(0).numpy() if waveform.shape[0] > 1 else waveform.squeeze().numpy()
    sr  = SAMPLE_RATE

    fft_mag  = np.abs(np.fft.rfft(arr[:sr], n=2048))
    fft_freq = np.fft.rfftfreq(2048, d=1/sr)

    def band_energy(lo, hi):
        mask = (fft_freq >= lo) & (fft_freq < hi)
        return float(np.mean(fft_mag[mask]**2)) if mask.any() else 0.0

    sub_bass   = band_energy(20,   80)
    bass_low   = band_energy(80,   250)
    bass_mid   = band_energy(250,  500)
    mid        = band_energy(500,  2000)
    upper_mid  = band_energy(2000, 4000)
    presence   = band_energy(4000, 8000)
    air        = band_energy(8000, 16000)
    total_e    = sub_bass + bass_low + bass_mid + mid + upper_mid + presence + air + 1e-12

    def rel(e): return e / total_e

    hr   = feat_dict.get("harmonic_ratio", 0.5)
    br   = feat_dict.get("beat_regularity", 0.5)
    sf   = feat_dict.get("spectral_flatness", 0.1)
    zcr  = feat_dict.get("zero_crossing_rate", 0.05)
    ps   = feat_dict.get("pitch_stability", 0.5)
    sc   = feat_dict.get("spectral_centroid", 0.3)
    dr   = feat_dict.get("dynamic_range", 0.01)

    detected = {}

    # ── Drums / Percussion ──────────────────────────────────────────────────
    # Strong sub+bass transients, rhythmic regularity, lower harmonic purity
    drum_score = (rel(sub_bass) * 4 + rel(bass_low) * 2) * (br ** 1.2) * max(0.1, 1.1 - hr)
    if drum_score > 0.012:
        detected["Drums"] = min(drum_score * 7, 1.0)

    # ── Electric Guitar ─────────────────────────────────────────────────────
    # Distorted electric guitar (e.g. Purple Haze): dominant mid energy,
    # moderate-to-high ZCR from distortion, spread across mid+upper_mid+presence.
    # Does NOT require high harmonic ratio — distortion reduces it.
    elec_guitar_score = (rel(mid) * 2.0 + rel(upper_mid) * 1.5 + rel(presence) * 0.8) * (0.4 + zcr * 3)
    if elec_guitar_score > 0.08:
        detected["Electric Guitar"] = min(elec_guitar_score * 2.0, 1.0)

    # ── Acoustic Guitar ──────────────────────────────────────────────────────
    # Clean acoustic: mid + upper-mid + presence, higher harmonic ratio, lower ZCR
    acoustic_guitar_score = (rel(mid) * 1.5 + rel(upper_mid) * 1.2 + rel(presence) * 0.5) * hr * (1 - min(zcr * 8, 0.8))
    if acoustic_guitar_score > 0.05:
        detected["Guitar"] = min(acoustic_guitar_score * 3.0, 1.0)

    # ── Bass Guitar ─────────────────────────────────────────────────────────
    # Strong low-end, moderate harmonics
    bass_score = (rel(bass_low) * 2.0 + rel(bass_mid) * 1.5 + rel(sub_bass) * 0.5) * (hr * 0.5 + 0.5)
    if bass_score > 0.05:
        detected["Bass Guitar"] = min(bass_score * 3.5, 1.0)

    # ── Vocals ───────────────────────────────────────────────────────────────
    # Mid + upper-mid dominant, natural ZCR range 0.05–0.12, moderate harmonics
    zcr_vocal_fit = max(0, 1 - abs(zcr - 0.085) * 10)
    vocal_score = (rel(mid) * 2.0 + rel(upper_mid) * 1.2) * (hr * 0.6 + 0.4) * zcr_vocal_fit
    if vocal_score > 0.04:
        detected["Vocals"] = min(vocal_score * 4.0, 1.0)

    # ── Piano / Keys ─────────────────────────────────────────────────────────
    # Broad mid spectrum, high harmonic ratio, moderate pitch stability, low SF
    piano_score = (rel(bass_mid) + rel(mid) * 1.5 + rel(upper_mid)) * hr * ps
    if piano_score > 0.06 and sf < 0.20:
        detected["Piano"] = min(piano_score * 2.5, 1.0)

    # ── Strings ──────────────────────────────────────────────────────────────
    # Upper-mid + presence energy, sustained pitch, high harmonic ratio
    strings_score = (rel(upper_mid) * 1.5 + rel(presence) * 0.8) * hr * (ps * 0.7 + 0.3)
    if strings_score > 0.025:
        detected["Strings"] = min(strings_score * 4.5, 1.0)

    # ── Synthesizer ───────────────────────────────────────────────────────────
    # Unnaturally flat spectrum OR dominant air-band energy
    synth_score = sf * 2.0 + rel(air) * 2.5
    if synth_score > 0.40:
        detected["Synthesizer"] = min(synth_score * 0.9, 1.0)

    # ── Trumpet / Brass ───────────────────────────────────────────────────────
    # Strong upper-mid + presence, high harmonic ratio, bright spectral centroid
    brass_score = (rel(upper_mid) * 1.5 + rel(presence)) * hr
    if brass_score > 0.04 and sc > 0.30:
        detected["Trumpet"] = min(brass_score * 3.5, 1.0)

    # ── Fallback ─────────────────────────────────────────────────────────────
    # Always return at least one instrument for any music track
    if not detected:
        bands = {
            "Drums":          sub_bass + bass_low,
            "Electric Guitar": mid + upper_mid,
            "Vocals":          upper_mid,
            "Synthesizer":     presence + air,
        }
        detected[max(bands, key=bands.get)] = 0.45

    return detected


def _detect_instrument_tags(waveform, feat_dict=None):
    """Run musicnn tagger over sliding windows; fall back to acoustic inference."""
    if instrument_pipeline is None:
        return _infer_instruments_acoustic(waveform, feat_dict or {})

    total = waveform.shape[1]
    win   = int(INSTRUMENT_WIN_SEC * INSTRUMENT_SAMPLE_RATE)
    step  = max(win, total // (INSTRUMENT_N_WINDOWS + 1))
    starts = [i * step for i in range(INSTRUMENT_N_WINDOWS) if i * step + win <= total]
    if not starts:
        starts = [0]
    scores: dict[str, list[float]] = {}
    for s in starts:
        clip = waveform[:, s: s + win]
        if clip.shape[1] < win:
            clip = torch.nn.functional.pad(clip, (0, win - clip.shape[1]))
        mono = clip.mean(0).numpy()
        try:
            preds = instrument_pipeline({"raw": mono, "sampling_rate": INSTRUMENT_SAMPLE_RATE},
                                        top_k=50)
            for p in preds:
                lbl = p["label"].lower()
                scores.setdefault(lbl, []).append(float(p["score"]))
        except Exception:
            pass
    return {k: float(np.mean(v)) for k, v in scores.items()}


def _score_instrument_authenticity(instrument_name, feat_dict, ai_probability):
    """
    Return (ai_score 0-100, confidence 'low'|'medium'|'high', signals list).
    Combines the track-level AI probability with per-instrument acoustic hints.
    """
    hints   = INSTRUMENT_AI_HINTS.get(instrument_name, {})
    signals = []
    nudges  = []

    for feat_key, (lo, hi, direction) in hints.items():
        val = feat_dict.get(feat_key)
        if val is None:
            continue
        in_range = lo <= val <= hi
        if in_range:
            if direction == "high":
                nudges.append(+12)
                signals.append(f"{'Unusually high' if feat_key != 'beat_regularity' else 'Overly rigid'} "
                               f"{feat_key.replace('_',' ')} ({val:.3f}) — common in AI-generated {instrument_name.lower()}")
            else:
                nudges.append(+10)
                signals.append(f"{'Suspiciously low' } "
                               f"{feat_key.replace('_',' ')} ({val:.3f}) — typical of synthesized {instrument_name.lower()}")
        else:
            if direction == "high":
                nudges.append(-8)
                signals.append(f"Natural {feat_key.replace('_',' ')} ({val:.3f}) — consistent with a real {instrument_name.lower()}")
            else:
                nudges.append(-6)

    base       = float(ai_probability)
    # Electronic instruments are expected to sound synthetic — soften the AI score
    if instrument_name in ELECTRONIC_INSTRUMENTS:
        base = max(base - 20, 0)

    nudge_total = sum(nudges) if nudges else 0
    raw_score   = float(np.clip(base + nudge_total, 0, 100))

    if len(nudges) >= 2:
        confidence = "high"
    elif len(nudges) == 1:
        confidence = "medium"
    else:
        confidence = "low"

    if raw_score >= 65:
        verdict = "AI-generated"
    elif raw_score <= 40:
        verdict = "Real / Performed"
    else:
        verdict = "Uncertain"

    return {
        "ai_score":   round(raw_score, 1),
        "real_score": round(100 - raw_score, 1),
        "verdict":    verdict,
        "confidence": confidence,
        "signals":    signals[:3],  # top 3 most relevant signals
    }


def classify_instruments(waveform, feat_dict, ai_probability):
    """
    Detect which instruments are present and classify each as AI or real.
    Always returns results — uses acoustic fallback if pipeline unavailable.
    """
    # _detect_instrument_tags now always returns something (acoustic fallback)
    raw_tags = _detect_instrument_tags(waveform, feat_dict)
    if not raw_tags:
        return {"available": True, "instruments": [], "note": "No instruments detected"}

    # When pipeline is loaded, map raw tags through INSTRUMENT_TAGS vocab
    # When using acoustic fallback, raw_tags already uses display names directly
    detected = {}
    if instrument_pipeline is not None:
        for tag_key, display_name in INSTRUMENT_TAGS.items():
            score = raw_tags.get(tag_key, 0.0)
            if score >= INSTRUMENT_MIN_SCORE:
                if display_name not in detected or score > detected[display_name]["tag_score"]:
                    detected[display_name] = {"tag_score": float(score)}
    else:
        # Acoustic fallback — raw_tags keys ARE display names already
        for display_name, score in raw_tags.items():
            if score >= INSTRUMENT_MIN_SCORE:
                detected[display_name] = {"tag_score": float(score)}

    if not detected:
        return {"available": True, "instruments": [], "note": "No instruments detected above threshold"}

    def _confidence_from_reliability(score):
        if score >= 75:
            return "high"
        if score >= 50:
            return "medium"
        return "low"

    instruments = []
    for name, meta in sorted(detected.items(), key=lambda x: x[1]["tag_score"], reverse=True):
        auth = _score_instrument_authenticity(name, feat_dict, ai_probability)
        tag_pct = float(round(meta["tag_score"] * 100, 1))
        decisiveness = float(round(abs(auth["ai_score"] - 50.0) * 2.0, 1))  # 0..100
        reliability = float(np.clip(tag_pct * 0.60 + decisiveness * 0.40, 0.0, 100.0))
        confidence = _confidence_from_reliability(reliability)
        instruments.append({
            "name":        name,
            "tag_score":   tag_pct,  # convert to pct
            "ai_score":    auth["ai_score"],
            "real_score":  auth["real_score"],
            "verdict":     auth["verdict"],
            "confidence":  confidence,
            "reliability_score": round(reliability, 1),
            "decisiveness": decisiveness,
            "signals":     auth["signals"],
            "is_electronic": name in ELECTRONIC_INSTRUMENTS,
        })

    # Summary
    ai_count   = sum(1 for i in instruments if i["verdict"] == "AI-generated")
    real_count = sum(1 for i in instruments if i["verdict"] == "Real / Performed")
    unc_count  = sum(1 for i in instruments if i["verdict"] == "Uncertain")

    if ai_count > real_count:
        summary_verdict = "Mostly AI-generated instruments"
    elif real_count > ai_count:
        summary_verdict = "Mostly real/performed instruments"
    else:
        summary_verdict = "Mixed — AI and real instruments detected"

    return {
        "available":        True,
        "source":           "neural_tagger" if instrument_pipeline is not None else "acoustic_analysis",
        "instruments":      instruments[:8],   # cap at 8 for UI
        "summary_verdict":  summary_verdict,
        "avg_reliability":  round(float(np.mean([i["reliability_score"] for i in instruments])) if instruments else 0.0, 1),
        "ai_count":         ai_count,
        "real_count":       real_count,
        "uncertain_count":  unc_count,
    }


# ── Audio conversion ──────────────────────────────────────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=128)

def mel_from_chunk(chunk, mel_transform):
    m = mel_transform(chunk)
    m = torchaudio.functional.amplitude_to_DB(m, 10.0, 1e-10, 0.0, 80.0)
    m = (m - m.mean()) / (m.std() + 1e-9)
    m = m.squeeze(0).transpose(0, 1)
    T = m.shape[0]
    if T < 1024:
        m = torch.nn.functional.pad(m, (0, 0, 0, 1024 - T))
    elif T > 1024:
        m = m[:1024, :]
    return m.unsqueeze(0)

def convert_to_16k_wav(input_path, original_filename=None):
    base="".join(c for c in os.path.splitext(os.path.basename(original_filename or input_path))[0]
                 if c.isalnum() or c in (" ","-","_")).rstrip()
    os.makedirs(CONVERTED_WAV_DIR, exist_ok=True)
    out=os.path.join(CONVERTED_WAV_DIR,f"{base}_16khz.wav")
    try:
        subprocess.run(["ffmpeg","-i",input_path,"-ar","16000","-ac","1","-c:a","pcm_s16le","-y",out],
                       capture_output=True, check=True, encoding='utf-8', errors='replace')
        dur=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                            "-of","default=noprint_wrappers=1:nokey=1",out],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        return out, float(dur.stdout.strip()) if dur.stdout.strip() else 0
    except subprocess.CalledProcessError:
        try: wav,sr=torchaudio.load(input_path)
        except Exception:
            data,sr=sf.read(input_path,always_2d=True); wav=torch.from_numpy(data.T).float()
        if wav.shape[0]>1: wav=wav.mean(0,keepdim=True)
        if sr!=SAMPLE_RATE: wav=torchaudio.transforms.Resample(sr,SAMPLE_RATE)(wav)
        peak=wav.abs().max().item()
        if peak>1e-6: wav=wav/peak
        torchaudio.save(out,wav,SAMPLE_RATE,encoding="PCM_S",bits_per_sample=16)
        return out, wav.shape[1]/SAMPLE_RATE

# ── Acoustic features ─────────────────────────────────────────────────────────
def _fft(a,n=1024): return np.abs(np.fft.rfft(a,n=n)), np.fft.rfftfreq(n,d=1/SAMPLE_RATE)

def compute_spectral_flatness(a):
    if np.sqrt(np.mean(a**2))<1e-6: return 0.0
    spec=_fft(a)[0]**2+1e-10
    return float(min(np.exp(np.mean(np.log(spec)))/np.mean(spec),0.70))

def compute_dynamic_range(a,fl=1600):
    f=[a[i:i+fl] for i in range(0,len(a)-fl,fl)]
    return float(np.std([np.sqrt(np.mean(x**2)+1e-9) for x in f])) if f else 0.0

def compute_zero_crossing_rate(a):
    return float(np.sum(np.abs(np.diff(np.sign(a))))/2/len(a))

def compute_spectral_centroid(a):
    mag,frq=_fft(a)
    return float(np.sum(frq*mag)/(np.sum(mag)+1e-10)/(SAMPLE_RATE/2))

def compute_spectral_rolloff(a,rp=0.85):
    mag,frq=_fft(a); cs=np.cumsum(mag**2)
    return float(frq[min(np.searchsorted(cs,rp*cs[-1]),len(frq)-1)]/(SAMPLE_RATE/2))

def compute_temporal_flux(a,fl=1600):
    f=[a[i:i+fl] for i in range(0,len(a)-fl,fl)]
    if len(f)<2: return 0.0
    sp=[np.abs(np.fft.rfft(x,n=min(1024,fl))) for x in f]
    return float(np.mean([np.sum((sp[i+1]-sp[i])**2) for i in range(len(sp)-1)]))

def compute_noise_floor(a,fl=1600):
    f=[a[i:i+fl] for i in range(0,len(a)-fl,fl)]
    if not f: return 0.0
    rms=sorted(float(np.sqrt(np.mean(x**2)+1e-12)) for x in f)
    return float(np.mean(rms[:max(1,len(rms)//10)]))

def compute_harmonic_ratio(a):
    if np.sqrt(np.mean(a**2))<1e-6: return 0.0
    mag=np.abs(np.fft.rfft(a[:min(4096,len(a))],n=min(4096,len(a))))**2
    return float(np.clip(np.sum(mag[mag>=np.percentile(mag,95)])/(np.sum(mag)+1e-12),0,1))

def compute_beat_regularity(a,sr=SAMPLE_RATE):
    fl=512; hop=256
    f=[a[i:i+fl] for i in range(0,len(a)-fl,hop)]
    if len(f)<8: return 0.5
    sp=[np.abs(np.fft.rfft(x,n=fl)) for x in f]
    onset=np.array([max(0,np.sum(sp[i+1]-sp[i])) for i in range(len(sp)-1)])
    onset=onset/(onset.max()+1e-9)
    ac=np.correlate(onset,onset,mode='full')[len(onset)-1:]; ac=ac/(ac[0]+1e-9)
    lo,hi=int(0.3*sr/hop),min(int(2.0*sr/hop),len(ac)-1)
    return float(np.clip(np.max(ac[lo:hi]) if lo<hi else 0.5,0,1))

def compute_pitch_stability(a,sr=SAMPLE_RATE):
    fl=2048; hop=1024
    f=[a[i:i+fl] for i in range(0,len(a)-fl,hop)]
    if len(f)<4: return 0.5
    freqs=[]
    for x in f:
        mag=np.abs(np.fft.rfft(x,n=fl)); frq=np.fft.rfftfreq(fl,d=1/sr)
        mask=(frq>=80)&(frq<=4000)
        if np.any(mask): freqs.append(frq[mask][np.argmax(mag[mask])])
    if len(freqs)<3: return 0.5
    mean=np.mean(freqs)
    return 0.5 if mean<1e-3 else float(np.clip(1.0-np.std(freqs)/(mean+1e-6)*4,0,1))

def compute_chunk_features(chunk_tensor):
    arr=chunk_tensor.squeeze().numpy()
    rms=float(np.sqrt(np.mean(arr**2))+1e-9)
    nrms=np.clip(arr/rms*0.1,-1.0,1.0); namp=np.clip(arr,-1.0,1.0)
    return {
        "spectral_flatness": compute_spectral_flatness(nrms),
        "dynamic_range":     compute_dynamic_range(namp),
        "zero_crossing_rate":compute_zero_crossing_rate(nrms),
        "spectral_centroid": compute_spectral_centroid(namp),
        "spectral_rolloff":  compute_spectral_rolloff(namp),
        "temporal_flux":     compute_temporal_flux(namp),
        "noise_floor":       compute_noise_floor(namp),
        "harmonic_ratio":    compute_harmonic_ratio(namp),
        "beat_regularity":   compute_beat_regularity(namp),
        "pitch_stability":   compute_pitch_stability(namp),
    }

# ── Scoring functions ────────────────────────────────────────────────────────
def _score(v,steps):
    for thr,s in steps:
        if v<thr: return s
    return steps[-1][1]

def sf_score(v): return _score(v,[(0.005,52),(0.04,44),(0.15,48),(0.30,55),(1,62)])
def dr_score(v): return _score(v,[(0.0005,62),(0.001,56),(0.006,46),(0.015,44),(1,52)])
def sc_score(v): return _score(v,[(0.02,58),(0.10,52),(0.30,44),(0.50,48),(1,54)])
def sr_score(v): return _score(v,[(0.01,58),(0.05,52),(0.25,44),(0.50,48),(1,54)])
def nf_score(v): return _score(v,[(0.0002,62),(0.001,56),(0.004,44),(0.015,46),(1,52)])
def hr_score(v): return _score(v,[(0.35,52),(0.55,46),(0.75,44),(0.90,54),(1,62)])
def br_score(v): return _score(v,[(0.25,52),(0.50,47),(0.75,44),(0.90,54),(1,60)])
def ps_score(v): return _score(v,[(0.40,52),(0.60,46),(0.80,44),(0.92,56),(1,62)])

def acoustic_composite_score(feat):
    tfs=_score(feat["temporal_flux"],  [(10,60),(200,54),(2000,44),(8000,48),(30000,52),(1e9,56)])
    zcs=_score(feat["zero_crossing_rate"],[(0.02,58),(0.08,50),(0.20,44),(0.40,48),(1,56)])
    return max(40,min(60,
        0.13*sf_score(feat["spectral_flatness"])+0.13*dr_score(feat["dynamic_range"])+
        0.12*sc_score(feat["spectral_centroid"])+0.12*sr_score(feat["spectral_rolloff"])+
        0.10*tfs+0.10*zcs+0.12*nf_score(feat["noise_floor"])+
        0.10*hr_score(feat["harmonic_ratio"])+0.09*br_score(feat["beat_regularity"])+
        0.09*ps_score(feat["pitch_stability"])))

# ── XAI feature scoring ──────────────────────────────────────────────────────
_AW = {"spectral_flatness":0.13,"dynamic_range":0.13,"tonal_variation":0.12,
       "harmonic_movement":0.12,"section_consistency":0.10,"transient_character":0.10,
       "noise_floor":0.12,"harmonic_ratio":0.10,"beat_regularity":0.09,"pitch_stability":0.09}

def _badge(s): return "AI-like" if s>=55 else ("Human-like" if s<=45 else "Neutral")

def _xfeat(id,label,score,value,wkey,what,why,why_verdict="",is_primary=False):
    w = NEURAL_WEIGHT if is_primary else ACOUSTIC_WEIGHT*_AW.get(wkey,0.10)
    return {"id":id,"label":label,"score":max(0,min(100,round(score))),"value":value,
            "badge":_badge(score),"weight":w,"weight_label":f"{round(w*100)}% of final score",
            "is_primary":is_primary,"what":what,"why":why,"why_verdict":why_verdict}

def _verdict_reason(score, ai_phrases, neutral_phrase, human_phrases):
    """Return a contextual verdict explanation based on score bucket."""
    if score >= 65:
        return ai_phrases[1] if score >= 80 else ai_phrases[0]
    elif score <= 35:
        return human_phrases[1] if score <= 20 else human_phrases[0]
    else:
        return neutral_phrase

def score_features_for_xai(feat, neural_prob, final_ai_probability=None):
    ns=max(0,min(100,round(neural_prob*100)))
    nw=(f"Strong AI signal ({ns}%) — primary driver of classification." if ns>=70 else
        f"Leans AI-generated ({ns}%) — main signal." if ns>=55 else
        f"Borderline score ({ns}%) — near the midpoint." if ns>=45 else
        f"Leans human ({ns}%) — but neural net is uncertain.")
    sfv=feat["spectral_flatness"]; sfs=sf_score(sfv)
    sfw=(f"Very low ({sfv:.3f}) — tonal purity suggests synthesis." if sfv<0.2 else
         f"{sfv:.3f} — natural balance of tonal content." if sfv<=0.45 else
         f"{sfv:.3f} — higher side, dense mix, or synthesised textures." if sfv<=0.7 else
         f"High ({sfv:.3f}) — near noise-like distribution.")
    sf_verdict=_verdict_reason(sfs,
        ["This track's frequency distribution is unusually uniform, a pattern common in AI synthesis where all frequencies are generated at even energy levels.",
         "Near-flat frequency distribution is a strong AI indicator — real instruments and voices naturally concentrate energy at harmonic peaks, not spread it evenly."],
        "Frequency distribution sits in a neutral zone — neither strongly tonal nor noise-like, so this feature provides limited evidence either way.",
        ["Energy is well-concentrated at musical pitches rather than spread evenly, which aligns with real instruments and organic sound sources.",
         "Highly tonal frequency distribution strongly suggests human performance — acoustic and electric instruments produce tight harmonic peaks that AI models rarely replicate perfectly."])
    drv=feat["dynamic_range"]; drs=dr_score(drv)
    drw=(f"Extremely low ({drv:.4f}) — loudness barely changes." if drv<0.01 else
         f"{drv:.4f} — heavily compressed." if drv<0.03 else
         f"{drv:.4f} — typical range for professionally produced music." if drv<0.08 else
         f"{drv:.4f} — fairly high, noticeable contrast." if drv<0.15 else
         f"Very high ({drv:.4f}) — dramatic loudness swings.")
    dr_verdict=_verdict_reason(drs,
        ["Very little loudness variation can indicate AI generation — synthetic audio often lacks the natural breathing and dynamic expression of human performance.",
         "Extremely flat loudness is a strong AI marker. Human musicians naturally swell, accent, and back off; this track shows almost none of that expressive variation."],
        "Loudness variation is within the typical range for professionally produced music — this feature alone doesn't distinguish human from AI on this track.",
        ["The track shows healthy loudness swings consistent with live or expressive human performance — dynamic contrast is a natural byproduct of human musicianship.",
         "Wide dynamic range is a strong indicator of human performance. AI-generated audio tends to be loudness-normalised; this track's dramatic swings point to organic, expressive playing."])
    scv=feat["spectral_centroid"]; scs=sc_score(scv)
    scw=(f"Near zero ({scv:.4f}) — near-silent or very bass-heavy." if scv<0.02 else
         f"{scv:.4f} — bass/low-mids dominant." if scv<0.08 else
         f"{scv:.4f} — balanced spread." if scv<0.15 else
         f"{scv:.4f} — moderately bright." if scv<0.30 else
         f"High ({scv:.4f}) — predominantly treble-heavy.")
    sc_verdict=_verdict_reason(scs,
        ["The spectral brightness pattern is atypical for organic recordings — AI models often generate tracks with an unnaturally skewed tonal centre.",
         "Strong spectral brightness anomaly detected. This frequency balance is uncommon in both acoustic and typical electronic human productions, nudging toward AI."],
        "Spectral brightness sits in a neutral zone shared by many music styles — this feature doesn't strongly separate human from AI for this track.",
        ["Balanced spectral brightness is consistent with a natural mix — human recordings and live instruments tend to occupy the mid-frequency centre of mass.",
         "Well-centred spectral brightness aligns closely with human music production norms, supporting the human classification."])
    srv=feat["spectral_rolloff"]; srs=sr_score(srv)
    srw=(f"Near zero ({srv:.4f}) — almost no energy above bass." if srv<0.01 else
         f"{srv:.4f} — 85% of energy in deep bass." if srv<0.05 else
         f"{srv:.4f} — typical range, natural treble taper." if srv<0.25 else
         f"{srv:.4f} — moderately high, bright mix." if srv<0.50 else
         f"Very high ({srv:.4f}) — dominant high-frequency content.")
    sr_verdict=_verdict_reason(srs,
        ["High-frequency energy distribution here deviates from typical human music — some AI models generate an unnatural high-end emphasis or treble content that doesn't taper naturally.",
         "Spectral rolloff is strongly atypical — the high-frequency energy pattern is difficult to attribute to normal acoustic or electronic human production."],
        "High-frequency rolloff is within typical range — this feature is inconclusive for separating AI from human on this track.",
        ["Spectral rolloff follows a natural high-frequency taper, consistent with how acoustic instruments and professionally mixed human recordings roll off in the high end.",
         "Very natural treble rolloff pattern — organic instruments and live recordings characteristically show this gradual high-frequency decay, supporting human origin."])
    tfv=feat["temporal_flux"]; tfs2=_score(tfv,[(10,60),(200,54),(2000,44),(8000,48),(30000,52),(1e9,56)])
    tfw=(f"Near-zero ({tfv:,.0f}) — essentially static audio." if tfv<10 else
         f"Low ({tfv:,.0f}) — very slow evolution." if tfv<200 else
         f"Healthy range ({tfv:,.0f}) — natural musical activity." if tfv<2000 else
         f"Fairly active ({tfv:,.0f}) — dense or percussive music." if tfv<8000 else
         f"High ({tfv:,.0f}) — very energetic." if tfv<30000 else
         f"Extremely high ({tfv:,.0f}) — approaching noise-like behaviour.")
    tf_verdict=_verdict_reason(tfs2,
        ["The rate of spectral change between frames is unusually low or high relative to the genre — AI generators can struggle to replicate the organic ebb-and-flow of real musical activity.",
         "Temporal flux is strongly anomalous. Real music has a characteristic rhythm of change; this track's frame-to-frame variation pattern is harder to explain by human performance."],
        "Moment-to-moment spectral change falls in a range common to many genres — this feature is not a strong discriminator for this particular track.",
        ["Spectral change rate follows the natural rhythmic pulse of human music-making — the ebb and flow of energy between frames matches typical live or studio recordings.",
         "Highly natural temporal flux pattern. The way frequency content evolves frame-to-frame closely mirrors organic human performance dynamics."])
    zcrv=feat["zero_crossing_rate"]; zcrs=_score(zcrv,[(0.02,58),(0.08,50),(0.20,44),(0.40,48),(1,56)])
    zcrw=(f"Very low ({zcrv:.3f}) — smooth, low-frequency dominated." if zcrv<0.05 else
          f"{zcrv:.3f} — typical range." if zcrv<0.15 else
          f"{zcrv:.3f} — moderately high." if zcrv<0.35 else
          f"High ({zcrv:.3f}) — dominant high-frequency or noise-like.")
    zcr_verdict=_verdict_reason(zcrs,
        ["Zero-crossing rate deviates from norms for this type of audio — an unusually smooth or noise-saturated waveform can indicate AI synthesis without natural transient character.",
         "Attack characteristics are strongly anomalous. The waveform's crossing pattern is inconsistent with how human-played or naturally recorded audio typically behaves."],
        "Zero-crossing rate is in a neutral zone — transient character here is common to both human and AI audio in this genre.",
        ["Waveform attack pattern is consistent with natural instrumental or vocal recordings — human-played instruments produce transient structures that match this profile.",
         "Very natural transient character. The rate at which the waveform crosses zero closely mirrors organic acoustic recordings, supporting human performance."])
    nfv=feat["noise_floor"]; nfs=nf_score(nfv)
    nfw=(f"Near-zero ({nfv:.5f}) — perfectly silent gaps." if nfv<0.0002 else
         f"Very low ({nfv:.5f}) — quieter than most real recordings." if nfv<0.001 else
         f"{nfv:.5f} — natural range, consistent with studio noise." if nfv<0.004 else
         f"Moderately elevated ({nfv:.5f}) — noticeable background noise." if nfv<0.015 else
         f"High ({nfv:.5f}) — substantial background noise.")
    nf_verdict=_verdict_reason(nfs,
        ["Near-zero or atypically low background noise in the quietest parts of the track is a common AI tell — synthesisers produce mathematically perfect silence between notes, whereas real recordings always capture some ambient sound.",
         "Background noise floor is suspiciously clean. AI-generated audio often has perfectly silent gaps; real studio recordings and live performances always retain some noise floor from the room, equipment, or environment."],
        "Background noise floor sits in a neutral range that could belong to either a well-recorded human track or a convincingly rendered AI output — this feature is inconclusive here.",
        ["A natural noise floor in the quiet parts of the track is consistent with a real recording environment — studios, rooms, and analogue equipment always leave a faint but measurable noise signature.",
         "Noise floor strongly supports human recording. The residual background level in the quietest segments matches what microphones, preamps, and acoustic environments leave behind — something AI generators rarely simulate accurately."])
    hrv=feat["harmonic_ratio"]; hrs=hr_score(hrv); hrp=round(hrv*100,1)
    hrw=(f"Very high ({hrp}%) — near-perfect harmonic series." if hrv>0.90 else
         f"High ({hrp}%) — most energy in harmonic peaks." if hrv>0.75 else
         f"{hrp}% — natural balance of harmonic peaks and noise." if hrv>0.55 else
         f"{hrp}% — notable noise energy between harmonics." if hrv>0.35 else
         f"Low ({hrp}%) — predominantly noise-like.")
    hr_verdict=_verdict_reason(hrs,
        ["An unusually high harmonic purity score can indicate AI synthesis — some generators produce unnaturally clean harmonic series with very little inter-harmonic noise, unlike real instruments which always add resonance and subtle inharmonicity.",
         "Near-perfect harmonic purity is rare in natural recordings. Human instruments — guitars, voices, pianos — always introduce slight inharmonicity and inter-harmonic noise; this level of tonal cleanliness points toward synthesis."],
        "Harmonic purity is in a neutral range shared by many genres — this feature does not strongly lean the verdict either way for this track.",
        ["Harmonic purity with natural inter-harmonic noise is consistent with human performance. Real instruments always produce some energy between harmonics due to resonance, bow noise, breath, or pick attack.",
         "Low harmonic purity is a natural characteristic of organic instruments and voice — guitar distortion, brass overtones, and vocal breathiness all introduce noise between harmonics in ways AI synthesis rarely replicates."])
    brv=feat["beat_regularity"]; brs=br_score(brv); brp=round(brv*100,1)
    brw=(f"Very high ({brp}%) — near-perfectly metronomic." if brv>0.90 else
         f"High ({brp}%) — very consistent rhythm." if brv>0.75 else
         f"{brp}% — clear pulse with human timing variations." if brv>0.50 else
         f"Low ({brp}%) — considerable timing variation." if brv>0.25 else
         f"Very low ({brp}%) — no clear repeating pulse.")
    br_verdict=_verdict_reason(brs,
        ["Near-perfect rhythmic regularity beyond what even the tightest human drummer achieves can be a sign of AI generation or programmed MIDI sequencing rather than live performance.",
         "Metronomic precision at this level is extremely difficult for human musicians to sustain — the rhythm is consistent to a degree that suggests machine-generated or quantised audio rather than live playing."],
        "Rhythmic regularity sits in a range typical of both tight human playing and AI — this feature alone cannot decide the verdict for this track.",
        ["Rhythmic regularity with natural timing drift is a hallmark of human performance — musicians naturally speed up, slow down, and 'breathe' with the music in ways machines don't.",
         "Highly irregular timing is a strong human indicator. Human performers naturally rush fills, lay back on grooves, and deviate from a perfect grid in ways that are very hard for AI to authentically replicate."])
    psv=feat["pitch_stability"]; pss=ps_score(psv); psp=round(psv*100,1)
    psw=(f"Very high ({psp}%) — pitch barely moves." if psv>0.92 else
         f"High ({psp}%) — very consistent pitch." if psv>0.80 else
         f"{psp}% — natural range, moderate variation." if psv>0.60 else
         f"Moderate-low ({psp}%) — significant pitch variation." if psv>0.40 else
         f"Low ({psp}%) — dominant pitch varies widely.")
    ps_verdict=_verdict_reason(pss,
        ["Pitch stability beyond normal human range can indicate AI synthesis — perfectly locked pitch without vibrato, portamento, or subtle drift is uncommon in live vocal or instrumental performance.",
         "Near-static pitch is difficult to produce naturally. Human singers and instrumentalists always introduce micro-variation; this level of pitch lock suggests synthesised or heavily pitch-corrected audio."],
        "Pitch stability is in a neutral range — this feature sits between what might be expected of a tightly performed human track and a normally generated AI output.",
        ["Natural pitch movement across the track — the amount of variation here is consistent with live vocal or instrumental performance, where vibrato, slides, and melodic drift are expected.",
         "Wide pitch variation is a strong indicator of expressive human performance. AI generators tend to produce more stable, locked pitch; this level of movement points to organic, human musicianship."])
    feats=[
        _xfeat("neural_score","Neural spectrogram score",ns,f"{neural_prob:.3f}","neural",
               "The deep neural network is the primary classifier.",nw,"",is_primary=True),
        _xfeat("spectral_flatness","Frequency distribution",sfs,f"{sfv:.3f}","spectral_flatness",
               "Whether energy is concentrated at specific musical pitches (tonal) or spread evenly.",sfw,sf_verdict),
        _xfeat("dynamic_range","Loudness variation",drs,f"{drv:.4f}","dynamic_range",
               "How much loudness changes moment-to-moment.",drw,dr_verdict),
        _xfeat("tonal_variation","Tonal brightness",scs,f"{scv:.4f}","tonal_variation",
               "Where the spectral 'centre of mass' sits.",scw,sc_verdict),
        _xfeat("harmonic_movement","High-frequency activity",srs,f"{srv:.4f}","harmonic_movement",
               "The frequency below which 85% of energy sits.",srw,sr_verdict),
        _xfeat("section_consistency","Moment-to-moment change",tfs2,f"{tfv:,.0f}","section_consistency",
               "How rapidly frequency content changes frame-to-frame.",tfw,tf_verdict),
        _xfeat("transient_character","Attack sharpness",zcrs,f"{zcrv:.3f}","transient_character",
               "How often the waveform crosses zero per second.",zcrw,zcr_verdict),
        _xfeat("noise_floor","Background noise floor",nfs,f"{nfv:.5f}","noise_floor",
               "The amplitude level during the quietest 10% of the track.",nfw,nf_verdict),
        _xfeat("harmonic_ratio","Harmonic purity",hrs,f"{hrp}%","harmonic_ratio",
               "The proportion of energy at distinct harmonic frequencies.",hrw,hr_verdict),
        _xfeat("beat_regularity","Rhythmic regularity",brs,f"{brp}%","beat_regularity",
               "How metronomically consistent the rhythm is.",brw,br_verdict),
        _xfeat("pitch_stability","Pitch consistency",pss,f"{psp}%","pitch_stability",
               "How consistently the dominant pitch holds its frequency.",psw,ps_verdict),
    ]
    primary=[f for f in feats if f["is_primary"]]
    secondary=sorted([f for f in feats if not f["is_primary"]],key=lambda x:x["score"],reverse=True)
    return primary+secondary

# ══════════════════════════════════════════════════════════════════════════════
# ── Comprehensive Acoustic Deep Analysis ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _interp_label(val, breakpoints):
    """Return a label string by linearly scanning sorted (threshold, label) breakpoints."""
    for thr, lbl in breakpoints:
        if val < thr:
            return lbl
    return breakpoints[-1][1]

def compute_acoustic_deep_analysis(feat, genre_result=None, segment_results=None,
                                   neural_prob=None, ai_probability=None, verdict=None):
    """
    Five-category deep acoustic analysis covering:
      1. Spectral characteristics
      2. Temporal & rhythmic features
      3. Dynamic range & loudness variation
      4. Harmonic vs percussive balance
      5. Genre awareness
      6. Artist/style signature proxies
      7. Production quality indicators
      8. AI vs Human signal lists
    Returns a structured dict consumed by the frontend.
    """

    feat = feat or {}

    # ── Raw feature values (with safe defaults) ──────────────────────────────
    sf_val  = float(feat.get("spectral_flatness",    0.0))
    dr_val  = float(feat.get("dynamic_range",        0.0))
    zcr_val = float(feat.get("zero_crossing_rate",   0.0))
    sc_val  = float(feat.get("spectral_centroid",    0.0))
    sr_val  = float(feat.get("spectral_rolloff",     0.0))
    tf_val  = float(feat.get("temporal_flux",        0.0))
    nf_val  = float(feat.get("noise_floor",          0.0))
    hr_val  = float(feat.get("harmonic_ratio",       0.0))
    br_val  = float(feat.get("beat_regularity",      0.0))
    ps_val  = float(feat.get("pitch_stability",      0.0))

    # ── 1. SPECTRAL CHARACTERISTICS ──────────────────────────────────────────
    # Spectral centroid → brightness
    brightness_label = _interp_label(sc_val, [
        (0.04,  "Sub-bass heavy — very dark tone"),
        (0.10,  "Bass/low-mid dominant — warm and full"),
        (0.20,  "Well-balanced — natural spectral spread"),
        (0.35,  "Moderately bright — upper-mid presence"),
        (1.0,   "Treble-heavy — very bright or thin mix"),
    ])
    # Spectral rolloff → energy distribution
    rolloff_label = _interp_label(sr_val, [
        (0.05,  "Energy concentrated in deep bass"),
        (0.15,  "Most energy in lows/mids — natural taper"),
        (0.35,  "Balanced extension into highs"),
        (0.55,  "Strong high-frequency extension"),
        (1.0,   "Dominant very-high-frequency content"),
    ])
    # Spectral flatness → tonal vs noisy
    tonality_label = _interp_label(sf_val, [
        (0.05,  "Highly tonal — strong pitched content"),
        (0.15,  "Mostly tonal — some noise-like texture"),
        (0.35,  "Balanced tonal/noise texture"),
        (0.60,  "Noise-like — broad spectral distribution"),
        (1.0,   "Near-white noise — very diffuse spectrum"),
    ])
    # Bandwidth proxy: rolloff - centroid gap
    bw_proxy = max(0.0, sr_val - sc_val)
    bandwidth_label = _interp_label(bw_proxy, [
        (0.05,  "Narrow bandwidth — limited frequency range"),
        (0.15,  "Moderate bandwidth"),
        (0.30,  "Wide bandwidth — full-range mix"),
        (1.0,   "Very wide bandwidth — extended highs and lows"),
    ])

    spectral = {
        "spectral_centroid":  {"value": round(sc_val, 4), "label": brightness_label,
                               "interpretation": "Spectral brightness — higher values indicate a brighter, more treble-forward mix."},
        "spectral_bandwidth": {"value": round(bw_proxy, 4), "label": bandwidth_label,
                               "interpretation": "Estimated frequency coverage from bass to highs."},
        "spectral_rolloff":   {"value": round(sr_val, 4), "label": rolloff_label,
                               "interpretation": "Point below which 85% of spectral energy falls."},
        "spectral_flatness":  {"value": round(sf_val, 3),  "label": tonality_label,
                               "interpretation": "Tonal vs noise-like quality — low = pitched/tonal, high = noisy/diffuse."},
        "zero_crossing_rate": {"value": round(zcr_val, 3),
                               "label": _interp_label(zcr_val, [
                                   (0.04, "Very smooth — predominantly low-frequency content"),
                                   (0.10, "Typical — balanced mid-frequency activity"),
                                   (0.20, "Elevated — significant high-frequency or transient content"),
                                   (1.0,  "High — dominant high-frequency or noise-like signal"),
                               ]),
                               "interpretation": "Waveform zero-crossing rate — a proxy for high-frequency content and percussive activity."},
    }

    # ── 2. TEMPORAL & RHYTHMIC FEATURES ──────────────────────────────────────
    # Beat regularity
    br_pct = round(br_val * 100, 1)
    rhythm_label = _interp_label(br_val, [
        (0.30, "Highly irregular — free-time or heavily rubato"),
        (0.55, "Loose groove — natural human timing variations"),
        (0.70, "Moderate consistency — slight expressive drift"),
        (0.85, "High regularity — tight, metronomic feel"),
        (1.0,  "Near-perfect grid — machine-like precision"),
    ])
    # Pitch stability → onset / melodic continuity proxy
    ps_pct = round(ps_val * 100, 1)
    onset_label = _interp_label(ps_val, [
        (0.45, "Very dynamic pitch movement — wide melodic range"),
        (0.65, "Expressive — natural pitch drift and vibrato"),
        (0.80, "Moderate stability — controlled melodic movement"),
        (0.92, "High stability — consistent tonal centre"),
        (1.0,  "Near-static — very little pitch change"),
    ])
    # Temporal flux → onset density / musical activity
    flux_label = _interp_label(tf_val, [
        (10,    "Near-static — very slow-evolving or silence"),
        (200,   "Sparse — gentle, minimal musical activity"),
        (1000,  "Moderate — typical melodic/harmonic content"),
        (5000,  "Active — rhythmically dense or percussive"),
        (15000, "Very active — energetic or complex arrangement"),
        (1e9,   "Extremely dense — noise-like temporal activity"),
    ])

    temporal = {
        "beat_regularity":  {"value": br_pct, "unit": "%", "label": rhythm_label,
                             "interpretation": "Autocorrelation-based rhythmic consistency — how metronomically regular the pulse is."},
        "pitch_stability":  {"value": ps_pct, "unit": "%", "label": onset_label,
                             "interpretation": "Dominant pitch steadiness — captures vibrato, portamento and melodic variation."},
        "temporal_flux":    {"value": round(tf_val, 1), "unit": "flux", "label": flux_label,
                             "interpretation": "Frame-to-frame spectral change rate — correlates with rhythmic density and arrangement complexity."},
    }

    # ── 3. DYNAMIC RANGE & LOUDNESS VARIATION ────────────────────────────────
    dynamics_label = _interp_label(dr_val, [
        (0.005, "Extremely compressed — almost zero dynamic contrast"),
        (0.015, "Heavily limited — typical loud-mastered pop/EDM"),
        (0.035, "Moderately compressed — professional level control"),
        (0.060, "Good dynamic range — natural loudness variation"),
        (0.120, "Wide dynamic range — significant soft/loud contrast"),
        (1.0,   "Very wide — dramatic amplitude swings"),
    ])
    # Noise floor → silence behaviour
    nf_label = _interp_label(nf_val, [
        (0.0001, "Perfectly silent gaps — digital silence"),
        (0.0005, "Near-silent — cleaner than most real recordings"),
        (0.001,  "Very low — typical studio floor"),
        (0.003,  "Natural studio noise — consistent with live tracking"),
        (0.010,  "Elevated background — ambient or room sound present"),
        (1.0,    "Substantial noise floor — noisy environment or analogue tape"),
    ])
    # Compression aggressiveness heuristic
    comp_index = round(max(0.0, 1.0 - dr_val / 0.10) * 100, 1)
    comp_label = _interp_label(comp_index, [
        (20,  "Minimal limiting — very open, uncompressed feel"),
        (40,  "Light compression — natural punch retained"),
        (60,  "Moderate compression — typical commercial production"),
        (80,  "Heavy limiting — very loud, modern mastering"),
        (101, "Extreme limiting — brickwall, zero headroom"),
    ])

    dynamics = {
        "dynamic_range":        {"value": round(dr_val, 4), "label": dynamics_label,
                                 "interpretation": "RMS-based loudness variation — low values suggest heavy limiting or mastering compression."},
        "noise_floor":          {"value": round(nf_val, 6), "label": nf_label,
                                 "interpretation": "Amplitude in the quietest 10% of the track — a proxy for recording environment and post-processing."},
        "compression_index":    {"value": comp_index, "unit": "%", "label": comp_label,
                                 "interpretation": "Estimated aggressiveness of dynamic range compression / limiting (higher = more compressed)."},
    }

    # ── 4. HARMONIC vs PERCUSSIVE BALANCE ────────────────────────────────────
    hr_pct = round(hr_val * 100, 1)
    harmonic_label = _interp_label(hr_val, [
        (0.30, "Predominantly percussive / noise-like — minimal harmonic structure"),
        (0.50, "Balanced — roughly equal harmonic and noise energy"),
        (0.65, "Mostly harmonic — some noise between overtones"),
        (0.80, "Strongly harmonic — clear pitch with overtone series"),
        (0.92, "Near-pure harmonic — very clean overtone stack"),
        (1.0,  "Theoretically pure — synthesised sine-like tones"),
    ])
    # Percussive proxy: inverse of harmonic_ratio weighted by zcr
    perc_index = round((1.0 - hr_val) * 60 + zcr_val * 40, 2)
    perc_label = _interp_label(perc_index, [
        (15, "Low percussive energy — melodic / tonal focus"),
        (30, "Moderate percussive presence — balanced rhythm/melody"),
        (50, "Significant percussive content — rhythm-led arrangement"),
        (70, "High percussive density — drums/transients dominant"),
        (100,"Very high — predominantly percussive / noise-like"),
    ])

    harmonic_percussive = {
        "harmonic_ratio":   {"value": hr_pct, "unit": "%", "label": harmonic_label,
                             "interpretation": "Proportion of spectral energy at distinct harmonic frequencies — high = pitched/tonal, low = percussive/noisy."},
        "percussive_index": {"value": round(perc_index, 1), "unit": "index", "label": perc_label,
                             "interpretation": "Estimated percussive character combining harmonic absence and zero-crossing activity."},
    }

    # ── 5. GENRE AWARENESS ────────────────────────────────────────────────────
    genre_analysis = {}
    if genre_result and genre_result.get("top"):
        top_genre  = genre_result["top"]["label"]
        top_score  = genre_result["top"]["score"]
        stability  = genre_result.get("stability", 0)
        conf_note  = genre_result.get("confidence_note", "")

        # Genre-specific feature interpretation norms
        GENRE_NORMS = {
            "Electronic": {
                "beat_regularity_expected": (0.85, 1.0),
                "dynamic_range_expected":   (0.005, 0.025),
                "harmonic_ratio_expected":  (0.55, 0.90),
                "note": "EDM/Electronic typically shows high rhythmic regularity and heavy limiting.",
            },
            "Hip-Hop": {
                "beat_regularity_expected": (0.70, 0.95),
                "dynamic_range_expected":   (0.010, 0.040),
                "harmonic_ratio_expected":  (0.40, 0.70),
                "note": "Hip-Hop commonly features quantised rhythm with moderate dynamics and sampled textures.",
            },
            "Pop": {
                "beat_regularity_expected": (0.75, 0.95),
                "dynamic_range_expected":   (0.008, 0.030),
                "harmonic_ratio_expected":  (0.50, 0.80),
                "note": "Pop production tends toward high regularity and moderate-to-heavy compression.",
            },
            "Classical": {
                "beat_regularity_expected": (0.30, 0.75),
                "dynamic_range_expected":   (0.04,  0.15),
                "harmonic_ratio_expected":  (0.65, 0.95),
                "note": "Classical music typically has wide dynamic range, expressive timing and rich harmonics.",
            },
            "Jazz": {
                "beat_regularity_expected": (0.35, 0.70),
                "dynamic_range_expected":   (0.03,  0.10),
                "harmonic_ratio_expected":  (0.55, 0.85),
                "note": "Jazz is known for expressive timing, complex harmony, and moderate dynamics.",
            },
            "Metal": {
                "beat_regularity_expected": (0.75, 0.97),
                "dynamic_range_expected":   (0.005, 0.025),
                "harmonic_ratio_expected":  (0.40, 0.75),
                "note": "Metal often features extremely precise timing, heavy limiting, and distorted (lower harmonic ratio) timbres.",
            },
            "Rock": {
                "beat_regularity_expected": (0.60, 0.88),
                "dynamic_range_expected":   (0.015, 0.060),
                "harmonic_ratio_expected":  (0.45, 0.75),
                "note": "Rock balances human groove with studio dynamics and guitar-driven harmonic content.",
            },
            "Blues": {
                "beat_regularity_expected": (0.40, 0.72),
                "dynamic_range_expected":   (0.025, 0.080),
                "harmonic_ratio_expected":  (0.50, 0.78),
                "note": "Blues is characterised by expressive timing, moderate dynamics, and vocal/guitar harmonic richness.",
            },
            "Country": {
                "beat_regularity_expected": (0.60, 0.85),
                "dynamic_range_expected":   (0.020, 0.070),
                "harmonic_ratio_expected":  (0.55, 0.82),
                "note": "Country music typically features tight but human rhythm and clean, open production.",
            },
            "R&B": {
                "beat_regularity_expected": (0.65, 0.90),
                "dynamic_range_expected":   (0.010, 0.040),
                "harmonic_ratio_expected":  (0.50, 0.78),
                "note": "R&B commonly uses programmed or tight rhythm with lush, processed harmonic content.",
            },
            "Reggae": {
                "beat_regularity_expected": (0.55, 0.82),
                "dynamic_range_expected":   (0.020, 0.060),
                "harmonic_ratio_expected":  (0.50, 0.76),
                "note": "Reggae features moderate rhythmic regularity with laid-back groove and warm, mid-heavy tone.",
            },
        }

        norms = GENRE_NORMS.get(top_genre, {})
        alignments = []

        if norms:
            def _check_alignment(val, lo, hi, name, human_dir="none"):
                in_range = lo <= val <= hi
                deviation = 0.0
                if val < lo:
                    deviation = round((lo - val) / max(lo, 1e-6) * 100, 1)
                    status = "Below expected range"
                elif val > hi:
                    deviation = round((val - hi) / max(hi, 1e-6) * 100, 1)
                    status = "Above expected range"
                else:
                    status = "Within expected range"
                return {"feature": name, "status": status,
                        "in_range": in_range, "deviation_pct": deviation,
                        "expected": f"{lo}–{hi}", "actual": val}

            br_lo, br_hi = norms.get("beat_regularity_expected", (0.5, 0.9))
            dr_lo, dr_hi = norms.get("dynamic_range_expected",   (0.01, 0.06))
            hr_lo, hr_hi = norms.get("harmonic_ratio_expected",  (0.45, 0.80))

            alignments = [
                _check_alignment(round(br_val, 3), br_lo, br_hi, "Beat Regularity"),
                _check_alignment(round(dr_val, 4), dr_lo, dr_hi, "Dynamic Range"),
                _check_alignment(round(hr_val, 3), hr_lo, hr_hi, "Harmonic Ratio"),
            ]
            in_range_count = sum(1 for a in alignments if a["in_range"])
            alignment_summary = (
                "Strong genre alignment — all key features match genre norms" if in_range_count == 3 else
                "Partial genre alignment — most features align with genre norms" if in_range_count == 2 else
                "Weak genre alignment — several features diverge from genre norms" if in_range_count == 1 else
                "Genre mismatch — key features conflict with genre norms"
            )
        else:
            alignment_summary = "Genre norms not available for this genre"

        genre_analysis = {
            "detected_genre":    top_genre,
            "confidence":        round(top_score, 1),
            "stability":         round(stability, 1),
            "confidence_note":   conf_note,
            "genre_note":        norms.get("note", ""),
            "feature_alignment": alignments,
            "alignment_summary": alignment_summary,
        }

    # ── 6. ARTIST / STYLE SIGNATURE PROXIES ──────────────────────────────────
    # Vocal processing proxy: pitch_stability × spectral_centroid brightness
    vocal_proc_index = round(ps_val * (1.0 - sf_val) * 100, 1)
    vocal_proc_label = _interp_label(vocal_proc_index, [
        (20, "Minimal vocal/melodic processing detected"),
        (40, "Light processing — natural or lightly treated sound"),
        (60, "Moderate processing — standard studio treatment"),
        (80, "Heavy processing — significant pitch/spectral shaping"),
        (101,"Extreme processing — highly synthesised or auto-tuned"),
    ])

    # Arrangement complexity: combo of temporal flux and spectral variation
    arrange_score = round(min(100, tf_val / 200 * 30 + (1.0 - sf_val) * 40 + br_val * 30), 1)
    arrange_label = _interp_label(arrange_score, [
        (20, "Very sparse — minimal arrangement"),
        (40, "Simple arrangement — few simultaneous elements"),
        (60, "Moderate complexity — layered but restrained"),
        (80, "Complex arrangement — dense, multi-layered production"),
        (101,"Very complex — highly intricate or maximalist arrangement"),
    ])

    # Artistic identity consistency across segments
    if segment_results and len(segment_results) >= 2:
        seg_neural = [s["neural_prob"] for s in segment_results]
        identity_cv = round(float(np.std(seg_neural) / (np.mean(seg_neural) + 1e-6)) * 100, 1)
        identity_label = _interp_label(identity_cv, [
            (8,  "Highly consistent identity — uniform character across track"),
            (18, "Consistent — minor variation between sections"),
            (30, "Moderate variation — some contrast between sections"),
            (50, "High variation — markedly different character by section"),
            (100,"Very inconsistent — sections sound like different recordings"),
        ])
        identity_note = (
            "Low variation may indicate repetitive, AI-like structure." if identity_cv < 8
            else "Natural variation suggests intentional artistic contrast." if identity_cv < 30
            else "High variation may indicate genre-blending or inconsistent production."
        )
    else:
        identity_cv = None
        identity_label = "Insufficient segments for identity analysis"
        identity_note  = ""

    style_signature = {
        "vocal_processing_index": {"value": vocal_proc_index, "label": vocal_proc_label,
                                   "interpretation": "Proxy for the degree of pitch/spectral processing applied to melodic content."},
        "arrangement_complexity": {"value": arrange_score, "label": arrange_label,
                                   "interpretation": "Estimated layering and arrangement density based on temporal activity and spectral character."},
        "identity_consistency":   {"value": identity_cv, "cv_pct": identity_cv, "label": identity_label,
                                   "note": identity_note,
                                   "interpretation": "Cross-segment variation in neural score — low CV may indicate generic/repetitive structure."},
    }

    # ── 7. PRODUCTION QUALITY INDICATORS ────────────────────────────────────
    # Stereo width proxy — not directly measurable from mono features, so we use
    # spectral complexity as a surrogate (centroid spread + flatness)
    mix_clarity_index = round((1.0 - sf_val) * 50 + (1.0 - abs(sc_val - 0.20) / 0.30) * 50, 1)
    mix_clarity_index = max(0.0, min(100.0, mix_clarity_index))
    mix_clarity_label = _interp_label(mix_clarity_index, [
        (30, "Poor mix clarity — congested or unbalanced spectrum"),
        (50, "Below average clarity — some muddiness or harshness"),
        (65, "Average clarity — typical commercial production"),
        (80, "Good clarity — well-balanced and separated mix"),
        (101,"Excellent clarity — very clean, open and professional mix"),
    ])

    # Mastering consistency: low temporal flux variance across segments
    if segment_results and len(segment_results) >= 2:
        seg_feat = [s.get("features", {}) for s in segment_results if s.get("features")]
        if seg_feat:
            dr_vals = [f.get("dynamic_range", dr_val) for f in seg_feat]
            master_cv = round(float(np.std(dr_vals) / (np.mean(dr_vals) + 1e-6)) * 100, 1)
        else:
            master_cv = None
    else:
        master_cv = None

    mastering_label = _interp_label(master_cv if master_cv is not None else 50, [
        (10, "Very consistent mastering — uniform loudness across track"),
        (25, "Consistent mastering — slight variation between sections"),
        (45, "Moderate consistency — some loudness swings"),
        (70, "Inconsistent — noticeable level differences by section"),
        (101,"Very inconsistent — mastering quality varies significantly"),
    ]) if master_cv is not None else "Insufficient data for mastering analysis"

    # Artifact detection heuristics
    artifacts = []
    if sf_val > 0.75:
        artifacts.append({"type": "Over-processing / spectral smearing",
                          "severity": "moderate",
                          "evidence": f"High spectral flatness ({sf_val:.3f}) suggests excessive processing or distortion.",
                          "ai_relevant": True})
    if dr_val < 0.005:
        artifacts.append({"type": "Brickwall limiting",
                          "severity": "high",
                          "evidence": f"Extremely low dynamic range ({dr_val:.4f}) indicates aggressive limiting.",
                          "ai_relevant": False})
    if br_val > 0.95:
        artifacts.append({"type": "Over-quantised rhythm",
                          "severity": "moderate",
                          "evidence": f"Near-perfect beat regularity ({br_val:.3f}) may indicate MIDI/grid quantisation.",
                          "ai_relevant": True})
    if nf_val < 0.00005:
        artifacts.append({"type": "Digital silence in gaps",
                          "severity": "low",
                          "evidence": f"Near-zero noise floor ({nf_val:.6f}) — no room tone or analogue noise present.",
                          "ai_relevant": True})
    if ps_val > 0.95:
        artifacts.append({"type": "Pitch over-correction / heavy auto-tune",
                          "severity": "moderate",
                          "evidence": f"Very high pitch stability ({ps_val:.3f}) may indicate pitch quantisation or auto-tune.",
                          "ai_relevant": True})

    production_quality = {
        "mix_clarity_index":     {"value": round(mix_clarity_index, 1), "label": mix_clarity_label,
                                  "interpretation": "Estimated mix clarity from spectral balance and tonal distribution."},
        "mastering_consistency": {"value": master_cv, "label": mastering_label,
                                  "interpretation": "Cross-segment dynamic range variation — low = consistently mastered."},
        "compression_index":     {"value": comp_index, "label": comp_label,
                                  "interpretation": "Estimated degree of dynamic compression / limiting applied."},
        "artifacts_detected":    artifacts,
        "artifact_count":        len(artifacts),
    }

    # ── 8. AI vs HUMAN INDICATOR LISTS ──────────────────────────────────────
    ai_signals    = []
    human_signals = []

    # Beat regularity
    if br_val > 0.93:
        ai_signals.append({"feature": "Rhythmic Uniformity",
                           "value": f"{br_pct}%",
                           "evidence": "Near-perfect metronomic consistency — typical of sequenced or quantised beats.",
                           "strength": "strong"})
    elif br_val < 0.55:
        human_signals.append({"feature": "Natural Timing Variation",
                              "value": f"{br_pct}%",
                              "evidence": "Loose, expressive timing with human micro-variations in the groove.",
                              "strength": "strong"})

    # Pitch stability
    if ps_val > 0.93:
        ai_signals.append({"feature": "Pitch Quantisation",
                           "value": f"{ps_pct}%",
                           "evidence": "Very stable pitch may indicate heavy auto-tune or synthesised tones.",
                           "strength": "moderate"})
    elif ps_val < 0.60:
        human_signals.append({"feature": "Expressive Pitch Movement",
                              "value": f"{ps_pct}%",
                              "evidence": "Natural pitch drift, vibrato, and melodic expression.",
                              "strength": "strong"})

    # Dynamic range
    if dr_val < 0.005:
        ai_signals.append({"feature": "Brickwall Dynamics",
                           "value": f"{dr_val:.4f}",
                           "evidence": "Extremely compressed loudness — no natural amplitude variation.",
                           "strength": "moderate"})
    elif dr_val > 0.06:
        human_signals.append({"feature": "Wide Dynamic Range",
                              "value": f"{dr_val:.4f}",
                              "evidence": "Natural loudness variation consistent with live or minimally compressed recording.",
                              "strength": "strong"})

    # Noise floor
    if nf_val < 0.00005:
        ai_signals.append({"feature": "Digital Silence",
                           "value": f"{nf_val:.6f}",
                           "evidence": "Perfectly silent background — no room tone or analogue noise floor.",
                           "strength": "moderate"})
    elif nf_val > 0.001:
        human_signals.append({"feature": "Natural Background Noise",
                              "value": f"{nf_val:.6f}",
                              "evidence": "Consistent low-level noise floor characteristic of real recording environments.",
                              "strength": "moderate"})

    # Harmonic ratio extremes
    if hr_val > 0.92:
        ai_signals.append({"feature": "Synthetic Harmonic Purity",
                           "value": f"{hr_pct}%",
                           "evidence": "Near-perfect harmonic series — indicative of synthesised tones without natural overtone decay.",
                           "strength": "moderate"})
    elif hr_val < 0.45:
        human_signals.append({"feature": "Organic Harmonic Content",
                              "value": f"{hr_pct}%",
                              "evidence": "Rich noise between harmonics consistent with acoustic instruments or analog recording.",
                              "strength": "moderate"})

    # Spectral flatness (over-processing)
    if sf_val > 0.65:
        ai_signals.append({"feature": "Spectral Smearing",
                           "value": f"{sf_val:.3f}",
                           "evidence": "High spectral flatness may indicate synthesis, heavy processing, or layered noise textures.",
                           "strength": "weak"})
    elif sf_val < 0.08:
        human_signals.append({"feature": "Tonal Focus",
                              "value": f"{sf_val:.3f}",
                              "evidence": "Strong tonal concentration consistent with acoustic instruments or clean synthesis.",
                              "strength": "weak"})

    # Temporal flux uniformity — if segment results available
    if segment_results and len(segment_results) >= 2:
        seg_feat_list = [s.get("features", {}) for s in segment_results if s.get("features")]
        if len(seg_feat_list) >= 2:
            tf_seg = [f.get("temporal_flux", tf_val) for f in seg_feat_list]
            tf_cv  = float(np.std(tf_seg) / (np.mean(tf_seg) + 1e-6))
            if tf_cv < 0.08:
                ai_signals.append({"feature": "Uniform Temporal Activity",
                                   "value": f"CV={tf_cv:.3f}",
                                   "evidence": "Very low variation in spectral change across sections — typical of repetitive AI generation.",
                                   "strength": "moderate"})
            elif tf_cv > 0.40:
                human_signals.append({"feature": "Varied Section Energy",
                                      "value": f"CV={tf_cv:.3f}",
                                      "evidence": "Strong contrast between sections (verse/chorus/bridge) — consistent with human composition.",
                                      "strength": "moderate"})

    # Neural score alignment
    if neural_prob is not None:
        np_pct = round(neural_prob * 100, 1)
        if neural_prob > 0.80:
            ai_signals.append({"feature": "Neural Network Signal",
                               "value": f"{np_pct}%",
                               "evidence": "The spectrogram pattern strongly matches AI-generated training examples.",
                               "strength": "strong"})
        elif neural_prob < 0.30:
            human_signals.append({"feature": "Neural Network Signal",
                                  "value": f"{100-np_pct}%",
                                  "evidence": "The spectrogram pattern strongly resembles human-recorded audio in the training set.",
                                  "strength": "strong"})

    # ── Final summary ─────────────────────────────────────────────────────────
    ai_strength_map = {"strong": 3, "moderate": 2, "weak": 1}
    ai_weight  = sum(ai_strength_map.get(s["strength"], 1)    for s in ai_signals)
    hum_weight = sum(ai_strength_map.get(s["strength"], 1)    for s in human_signals)
    total_w    = ai_weight + hum_weight + 1e-6
    signal_balance = round(ai_weight / total_w * 100, 1)  # 0=fully human, 100=fully AI
    balance_label = _interp_label(signal_balance, [
        (20,  "Strong human signal dominance"),
        (38,  "Mostly human indicators"),
        (48,  "Slight lean toward human"),
        (52,  "Balanced — near equal AI and human signals"),
        (62,  "Slight lean toward AI"),
        (78,  "Mostly AI indicators"),
        (101, "Strong AI signal dominance"),
    ])

    return {
        "spectral":            spectral,
        "temporal":            temporal,
        "dynamics":            dynamics,
        "harmonic_percussive": harmonic_percussive,
        "genre_awareness":     genre_analysis,
        "style_signature":     style_signature,
        "production_quality":  production_quality,
        "ai_indicators":       ai_signals,
        "human_indicators":    human_signals,
        "signal_balance":      {"score": signal_balance, "label": balance_label,
                                "ai_signal_count":    len(ai_signals),
                                "human_signal_count": len(human_signals)},
    }


def derive_verdict(p):
    """Classify AI probability into a verdict label.
    Uses AI_THRESHOLD (60) and HUMAN_THRESHOLD (45) so thresholds are consistent
    with config constants. Gap 46-59 = Not Sure zone.
    """
    if p >= 80:
        return "Likely AI-generated", f"Strong AI patterns detected ({p:.1f}%)"
    elif p >= AI_THRESHOLD:   # >= 60
        return "Likely AI-generated", f"Patterns lean toward AI generation ({p:.1f}%)"
    elif p <= 20:
        return "Likely Human", f"Strong human characteristics detected ({100-p:.1f}% human)"
    elif p <= HUMAN_THRESHOLD:  # <= 45
        return "Likely Human", f"Patterns lean toward human performance ({100-p:.1f}% human)"
    else:
        # 46–59: genuine ambiguous zone between the two thresholds
        return "Not Sure", f"Signal is ambiguous — borderline result ({p:.1f}%)"

# ── Load models ───────────────────────────────────────────────────────────────
model = None
MODEL_LOAD_ERROR = None

print(f"[*] Loading AI detection model on {device}...")
try:
    ast_backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = HybridASTDetector(ast_backbone).to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Place best_model.pth in the same directory as app.py."
        )
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state_dict'. Keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[\u2713] AI detector loaded — epoch={ckpt.get('epoch',0)+1}  val_acc={ckpt.get('val_acc',0):.2%}  device={device}")
except Exception as _e:
    MODEL_LOAD_ERROR = str(_e)
    model = None
    print(f"[\u2717] AI detector failed to load: {_e}")

print(f"[*] Loading primary genre classifier ({GENRE_MODEL_PRIMARY})...")
try:
    genre_pipeline_primary=pipeline("audio-classification",model=GENRE_MODEL_PRIMARY,
                                    device=0 if torch.cuda.is_available() else -1)
    GENRE_PRIMARY_AVAILABLE=True; print("[✓] Primary genre classifier loaded")
except Exception as e:
    print(f"[!] Primary genre classifier failed: {e}"); genre_pipeline_primary=None; GENRE_PRIMARY_AVAILABLE=False

print(f"[*] Loading secondary genre classifier ({GENRE_MODEL_SECONDARY})...")
try:
    genre_pipeline_secondary=pipeline("audio-classification",model=GENRE_MODEL_SECONDARY,
                                      device=0 if torch.cuda.is_available() else -1,trust_remote_code=True)
    GENRE_SECONDARY_AVAILABLE=True; print("[✓] Secondary genre classifier loaded")
except Exception as e:
    print(f"[!] Secondary genre classifier failed: {e}"); genre_pipeline_secondary=None; GENRE_SECONDARY_AVAILABLE=False

GENRE_AVAILABLE=GENRE_PRIMARY_AVAILABLE or GENRE_SECONDARY_AVAILABLE

print(f"[*] Loading instrument tagger ({INSTRUMENT_MODEL})...")
INSTRUMENT_AVAILABLE = False
instrument_pipeline  = None
try:
    instrument_pipeline = pipeline(
        "audio-classification",
        model=INSTRUMENT_MODEL,
        device=0 if torch.cuda.is_available() else -1,
        trust_remote_code=True,
    )
    INSTRUMENT_AVAILABLE = True
    print("[✓] Instrument tagger loaded")
except Exception as _ie:
    print(f"[!] Instrument tagger failed to load: {_ie}")
    instrument_pipeline = None

# ── Preview clips ─────────────────────────────────────────────────────────────
PREVIEW_CLIP_SEC = DURATION_SEC
PREVIEW_DIR       = "preview_clips"

def _make_previews(wav_path, duration_sec, clip_sec=PREVIEW_CLIP_SEC):
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    dur = max(float(duration_sec), 0.1)
    half = clip_sec / 2.0
    segments = []
    intro_dur = min(clip_sec, dur)
    segments.append(("intro", 0.0, intro_dur))
    if dur > clip_sec:
        chorus_start = max(0.0, dur / 2.0 - half)
        if chorus_start + clip_sec > dur:
            chorus_start = max(0.0, dur - clip_sec)
        chorus_dur = min(clip_sec, dur - chorus_start)
        segments.append(("chorus", chorus_start, chorus_dur))
    else:
        remaining = dur - intro_dur
        if remaining > 0:
            segments.append(("chorus", intro_dur, remaining))
    if dur > clip_sec:
        ending_start = max(0.0, dur - clip_sec)
        ending_dur = min(clip_sec, dur - ending_start)
        segments.append(("ending", ending_start, ending_dur))
    base = os.path.splitext(os.path.basename(wav_path))[0]
    merged_filename = f"{base}_merged_preview_{FRAMES_NEEDED}frames.wav"
    merged_path = os.path.join(PREVIEW_DIR, merged_filename)
    if os.path.exists(merged_path):
        total_dur = sum(s[2] for s in segments if s[2] > 0)
        return {
            "merged_url": f"/preview_clips/{merged_filename}",
            "segments": [{"label": label, "start_sec": round(start, 2), "duration_sec": round(dur_seg, 2), "frames": int(dur_seg * SAMPLE_RATE / HOP_LENGTH)}
                         for label, start, dur_seg in segments if dur_seg > 0],
            "total_duration_sec": round(total_dur, 2),
            "total_frames": int(total_dur * SAMPLE_RATE / HOP_LENGTH),
            "frames_per_clip": FRAMES_NEEDED, "cached": True
        }
    try:
        concat_file = os.path.join(PREVIEW_DIR, f"{base}_concat_list.txt")
        with open(concat_file, "w") as f:
            for label, start, seg_dur in segments:
                if seg_dur <= 0: continue
                seg_filename = f"{base}__{label}_{int(seg_dur*100)}ms.wav"
                seg_path = os.path.join(PREVIEW_DIR, seg_filename)
                if not os.path.exists(seg_path):
                    subprocess.run(["ffmpeg", "-y", "-i", wav_path,
                                    "-ss", str(round(start, 3)), "-t", str(round(seg_dur, 3)),
                                    "-c:a", "pcm_s16le", seg_path],
                                   capture_output=True, check=True,
                                   encoding='utf-8', errors='replace')
                f.write(f"file '{seg_filename}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_file, "-c:a", "pcm_s16le", merged_path],
                       capture_output=True, check=True,
                       encoding='utf-8', errors='replace')
        total_dur = sum(s[2] for s in segments if s[2] > 0)
        return {
            "merged_url": f"/preview_clips/{merged_filename}",
            "segments": [{"label": label, "start_sec": round(start, 2), "duration_sec": round(seg_dur, 2), "frames": int(seg_dur * SAMPLE_RATE / HOP_LENGTH)}
                         for label, start, dur_seg in segments if dur_seg > 0],
            "total_duration_sec": round(total_dur, 2),
            "total_frames": int(total_dur * SAMPLE_RATE / HOP_LENGTH),
            "frames_per_clip": FRAMES_NEEDED, "cached": False
        }
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "segments": []}

@app.route("/preview_clips/<path:filename>")
def serve_preview_clip(filename): return send_from_directory(PREVIEW_DIR, filename)

@app.route("/")
def index(): return send_from_directory(".", "index.html")

@app.route("/converted_wavs/<path:filename>")
def serve_converted_wav(filename): return send_from_directory(CONVERTED_WAV_DIR, filename)

@app.route("/debug/last_analysis")
def debug_last_analysis():
    import glob
    return jsonify({"converted_wav_dir":CONVERTED_WAV_DIR,
                    "wav_files_found":len(glob.glob(os.path.join(CONVERTED_WAV_DIR,"*.wav"))),
                    "genre_available":GENRE_AVAILABLE,
                    "analysis_duration_sec": DURATION_SEC,
                    "analysis_frames": FRAMES_NEEDED,
                    "ai_threshold": AI_THRESHOLD,
                    "human_threshold": HUMAN_THRESHOLD})

# ── /report ───────────────────────────────────────────────────────────────────
@app.route("/report", methods=["POST"])
def report():
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({"error": "No JSON body received."}), 400
        required = {"filename", "original_verdict", "correct_label", "reason"}
        missing = required - data.keys()
        if missing: return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        report_id = str(uuid.uuid4())[:8].upper()
        record = {
            "report_id": report_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "filename": str(data.get("filename", "unknown"))[:300],
            "original_verdict": str(data.get("original_verdict", ""))[:100],
            "ai_probability": float(data.get("ai_probability", 0)),
            "correct_label": str(data.get("correct_label", ""))[:50],
            "reason": str(data.get("reason", ""))[:100],
            "comment": str(data.get("comment", ""))[:500],
            "genre": data.get("genre"),
            "codec_detected": bool(data.get("codec_detected", False)),
        }
        with open(REPORTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return jsonify({"ok": True, "report_id": report_id}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Natural-language summary generator ──────────────────────────────────────
def generate_nl_explanation(
    verdict, ai_probability, neural_prob, acoustic_prob,
    feat_dict, genre=None, human_score=None, adjustment=None
):
    pct = round(ai_probability, 1)
    lines = []

    if verdict == "Likely AI-generated":
        if pct >= 75:
            opener = f"This track scores {pct}% on the AI similarity scale, indicating strong AI-generated characteristics."
        else:
            opener = f"This track scores {pct}% on the AI similarity scale, suggesting AI-generated content."
    elif verdict == "Not Sure":
        opener = f"The model is uncertain (score: {pct}%), suggesting this track sits in an ambiguous zone between AI and human characteristics."
    else:
        if pct <= 25:
            opener = f"With a score of {pct}%, this track shows strong human acoustic characteristics."
        else:
            opener = f"With a score of {pct}%, this track shows predominantly human acoustic characteristics."
    lines.append(opener)

    # Only mention human indicators when they conflict with an AI verdict
    if human_score is not None and human_score > 70 and verdict == "Likely AI-generated":
        lines.append(f"Acoustic indicators ({human_score:.0f}% human-likelihood) suggest some organic qualities, but the neural network's spectrogram analysis overrides this.")
    elif human_score is not None and human_score > 80 and verdict != "Likely AI-generated":
        lines.append(f"Acoustic indicators ({human_score:.0f}% human-likelihood) reinforce the human classification with natural timing, dynamics, and tonal qualities.")

    disagreement = abs(neural_prob - acoustic_prob)
    if disagreement > 0.25:
        lines.append(f"Note: The neural network ({neural_prob:.0%} AI) is the primary signal, while acoustic features ({acoustic_prob:.0%} AI) provide supplementary context.")

    return " ".join(lines)

@app.route("/classify", methods=["POST"])
def classify():
    if model is None:
        return jsonify({"error": f"Model not loaded: {MODEL_LOAD_ERROR}"}), 503
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files uploaded."}), 400
    results = []
    for f in files:
        filename = f.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            results.append({"filename": filename, "error": f"Unsupported format: {ext}"})
            continue
        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)
        try:
            wav_path, total_secs = convert_to_16k_wav(tmp_path, original_filename=filename)
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"WAV not created: {wav_path}")
            data, sr = sf.read(wav_path, always_2d=True)
            waveform = torch.from_numpy(data.T).float()
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
                sr = SAMPLE_RATE

            weighted_neural, weighted_acoustic, segment_results, best_segment = analyze_weighted_segments(
                waveform, total_secs, model, mel_transform, device
            )

            chunk_feat = best_segment["features"]

            genre_result = None
            if GENRE_AVAILABLE:
                genre_result = classify_genre(waveform)

            sample_for_codec = waveform[:, :min(SAMPLES_NEEDED, waveform.shape[1])]
            is_compressed, _, codec_conf, _ = detect_codec_compression(sample_for_codec, SAMPLE_RATE)

            ai_probability, total_adjustment, human_reasons = compute_calibrated_probability(
                weighted_neural, weighted_acoustic, chunk_feat, genre_result
            )

            if is_compressed:
                ai_probability, _ = apply_codec_penalty(ai_probability, is_compressed, codec_conf)

            verdict, verdict_explanation = derive_verdict(ai_probability)

            human_likelihood_score = compute_human_likelihood_score(chunk_feat)

            # ── Instrument AI/real classification ──
            instrument_analysis = classify_instruments(waveform, chunk_feat, ai_probability)

            # XAI deep
            run_xai_deep = request.form.get("xai_deep", "false").lower() == "true"
            xai_deep = None
            if run_xai_deep:
                if best_segment["name"] == "chorus":
                    start_sample = int(best_segment["start_sec"] * SAMPLE_RATE)
                    end_sample = int(best_segment["end_sec"] * SAMPLE_RATE)
                    best_segment_waveform = waveform[:, start_sample:end_sample]
                else:
                    best_segment_waveform = waveform[:, :SAMPLES_NEEDED]
                
                if best_segment_waveform.shape[1] < SAMPLES_NEEDED:
                    best_segment_waveform = torch.nn.functional.pad(best_segment_waveform, (0, SAMPLES_NEEDED - best_segment_waveform.shape[1]))
                elif best_segment_waveform.shape[1] > SAMPLES_NEEDED:
                    best_segment_waveform = best_segment_waveform[:, :SAMPLES_NEEDED]
                
                best_mel = mel_from_chunk(best_segment_waveform, mel_transform)
                _, xai_deep = compute_xai_deep(best_mel, model, device)
                # Align Grad-CAM pattern_verdict with the final model verdict
                if xai_deep and "gradcam" in xai_deep and xai_deep["gradcam"] and "pattern_verdict" in xai_deep["gradcam"]:
                    gc_data  = xai_deep["gradcam"]
                    ai_sig   = len(gc_data.get("ai_signals", []))
                    hum_sig  = len(gc_data.get("human_signals", []))
                    # Final verdict carries 2 extra votes as a strong tiebreaker
                    if "AI" in verdict:
                        ai_sig  += 2
                    elif "Human" in verdict:
                        hum_sig += 2
                    if ai_sig > hum_sig:
                        gc_data["pattern_verdict"] = "AI-like pattern"
                        gc_data["pattern_color"]   = "red"
                        gc_data["plain_summary"]   = (
                            f"The neural network found AI-like spectral patterns. "
                            f"Peak activation at {gc_data.get('peak_freq_hz', 0):.0f} Hz "
                            f"({gc_data.get('freq_zone', '—')}) over "
                            f"{gc_data.get('time_spread_sec', 0):.2f}s — "
                            f"consistent with the AI-generated verdict."
                        )
                    elif hum_sig > ai_sig:
                        gc_data["pattern_verdict"] = "Human-like pattern"
                        gc_data["pattern_color"]   = "green"
                        gc_data["plain_summary"]   = (
                            f"Activation was broadly spread across "
                            f"{gc_data.get('time_spread_sec', 0):.1f}s and "
                            f"{gc_data.get('freq_spread_hz', 0):.0f} Hz of frequency range, "
                            f"centred in the {gc_data.get('freq_zone', '—')} "
                            f"— typical of organic, human-performed audio."
                        )
                    else:
                        gc_data["pattern_verdict"] = "Ambiguous pattern"
                        gc_data["pattern_color"]   = "amber"

            # Natural-language explanation
            nl_explanation = generate_nl_explanation(
                verdict, ai_probability, weighted_neural, weighted_acoustic,
                chunk_feat, genre_result, human_likelihood_score, total_adjustment
            )

            # Feature scores
            xai_feats = score_features_for_xai(chunk_feat, weighted_neural, ai_probability)

            # Comprehensive acoustic deep analysis
            try:
                acoustic_deep = compute_acoustic_deep_analysis(
                    feat=chunk_feat,
                    genre_result=genre_result,
                    segment_results=segment_results,
                    neural_prob=weighted_neural,
                    ai_probability=ai_probability,
                    verdict=verdict,
                )
            except Exception as e:
                acoustic_deep = {"error": str(e)}

            # Genre response
            gp = {"available": GENRE_AVAILABLE, "primary_available": GENRE_PRIMARY_AVAILABLE,
                  "secondary_available": GENRE_SECONDARY_AVAILABLE,
                  **({k: genre_result.get(k) for k in ["top", "all", "top_subgenre", "sources", "window_count", "stability", "confidence_note", "primary_used", "secondary_used"]}
                     if genre_result else {"top": None, "all": None, "top_subgenre": None, "sources": [], "window_count": 0, "stability": 0, "confidence_note": "", "primary_used": False, "secondary_used": False}),
                  "subgenres": genre_result.get("subgenres", {}) if genre_result else {}}


            # Sanitize segment_results for JSON serialization
            def _sanitize_seg(s):
                return {
                    "name":            s["name"],
                    "weight":          float(s["weight"]),
                    "effective_weight":float(s.get("effective_weight", s["weight"])),
                    "neural_prob":     float(s["neural_prob"]),
                    "acoustic_prob":   float(s["acoustic_prob"]),
                    "human_score":     float(s["human_score"]),
                    "confidence":      float(s["confidence"]),
                    "evidence_quality":float(s["evidence_quality"]),
                    "corroborated":    bool(s["corroborated"]),
                    "raw_logit":       float(s["raw_logit"]),
                    "start_sec":       float(s["start_sec"]),
                    "end_sec":         float(s["end_sec"]),
                    # include key features but cast to Python float
                    "features": {k: float(v) for k, v in s.get("features", {}).items()},
                }
            safe_segs = [_sanitize_seg(s) for s in segment_results]

            results.append({
                "original_filename": filename,
                "display_filename": os.path.basename(wav_path),
                "original_format": ext.upper().replace(".", ""),
                "analyzed_format": "WAV (16kHz)",
                "conversion_note": f"Converted from {ext.upper()} to 16kHz WAV",
                "converted_wav_path": f"/converted_wavs/{os.path.basename(wav_path)}" if SAVE_CONVERTED_WAV else None,
                "ai_probability": ai_probability,
                "human_probability": round(100 - ai_probability, 1),
                "verdict": verdict,
                "verdict_explanation": verdict_explanation,
                "nl_explanation": nl_explanation,
                "duration_sec": round(total_secs, 2),
                "analysis_duration_sec": DURATION_SEC,
                "analysis_samples": SAMPLES_NEEDED,
                "analysis_frames": FRAMES_NEEDED,
                "analysis_mode": "weighted_segments_v2_majority_vote",
                "segment_analysis": {
                    "segments": safe_segs,
                    "best_segment": best_segment["name"],
                    "weighted_neural": round(weighted_neural, 4),
                    "weighted_acoustic": round(weighted_acoustic, 4),
                    "neural_weight": NEURAL_WEIGHT,
                    "acoustic_weight": ACOUSTIC_WEIGHT,
                },
                "xai": xai_feats,
                "xai_deep": xai_deep,
                "acoustic_deep": acoustic_deep,
                "human_indicators": {
                    "score": round(human_likelihood_score, 1),
                    "reasons": human_reasons,
                    "adjustment_applied": round(total_adjustment, 1),
                    "beat_regularity": round(chunk_feat["beat_regularity"], 3),
                    "pitch_stability": round(chunk_feat["pitch_stability"], 3),
                    "dynamic_range": round(chunk_feat["dynamic_range"], 4),
                    "noise_floor": round(chunk_feat["noise_floor"], 5),
                    "harmonic_ratio": round(chunk_feat["harmonic_ratio"], 3),
                },
                "genre": gp,
                "score_breakdown": {
                    "neural_weight_pct": round(NEURAL_WEIGHT * 100),
                    "acoustic_weight_pct": round(ACOUSTIC_WEIGHT * 100),
                    "neural_contribution": round(weighted_neural * NEURAL_WEIGHT * 100, 1),
                    "acoustic_contribution": round(weighted_acoustic * ACOUSTIC_WEIGHT * 100, 1),
                    "codec_compression": {
                        "detected": is_compressed,
                        "confidence": round(codec_conf, 3),
                    },
                },
                "diagnostics": {
                    "analysis_source": wav_path,
                    "converted_from": ext,
                    "sample_rate": sr,
                },
                "instrument_analysis": instrument_analysis,
                "previews": _make_previews(wav_path, total_secs),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"filename": filename, "error": str(e)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Auto-save a report for each successful result ──────────────────────────
    for r in results:
        if "error" not in r:
            try:
                report_id = str(uuid.uuid4())
                report = {
                    "report_id":        report_id,
                    "analyzed_at":      datetime.datetime.utcnow().isoformat() + "Z",
                    "filename":         r.get("original_filename", "unknown"),
                    "format":           r.get("original_format", ""),
                    "duration_sec":     r.get("duration_sec"),
                    "verdict":          r.get("verdict"),
                    "ai_probability":   r.get("ai_probability"),
                    "human_probability":r.get("human_probability"),
                    "nl_explanation":   r.get("nl_explanation", ""),
                    "genre":            r.get("genre", {}).get("top", {}).get("label") if r.get("genre") else None,
                    "genre_score":      r.get("genre", {}).get("top", {}).get("score") if r.get("genre") else None,
                    "human_likelihood": r.get("human_indicators", {}).get("score"),
                    "beat_regularity":  r.get("human_indicators", {}).get("beat_regularity"),
                    "pitch_stability":  r.get("human_indicators", {}).get("pitch_stability"),
                    "dynamic_range":    r.get("human_indicators", {}).get("dynamic_range"),
                    "segments":         [
                        {
                            "name":        s.get("name"),
                            "neural_prob": round(s.get("neural_prob", 0) * 100, 1),
                            "human_score": s.get("human_score"),
                            "weight":      s.get("weight"),
                        }
                        for s in r.get("segment_analysis", {}).get("segments", [])
                    ],
                    "instruments":      [
                        {"name": i.get("name"), "verdict": i.get("verdict")}
                        for i in (r.get("instrument_analysis", {}) or {}).get("instruments", [])
                    ] if r.get("instrument_analysis") else [],
                }
                report_path = os.path.join(ANALYSIS_REPORTS_DIR, f"{report_id}.json")
                with open(report_path, "w") as f:
                    _json.dump(report, f, indent=2, cls=_SafeEncoder)
            except Exception as save_err:
                print(f"[report] Failed to save report: {save_err}")

    return jsonify({"results": results})

@app.route("/reports", methods=["GET"])
def list_reports():
    """Return all saved analysis reports, newest first."""
    reports = []
    try:
        for fname in os.listdir(ANALYSIS_REPORTS_DIR):
            if fname.endswith(".json"):
                fpath = os.path.join(ANALYSIS_REPORTS_DIR, fname)
                try:
                    with open(fpath) as f:
                        reports.append(_json.load(f))
                except Exception:
                    pass
        reports.sort(key=lambda r: r.get("analyzed_at", ""), reverse=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"reports": reports, "total": len(reports)})


@app.route("/reports/<report_id>", methods=["GET"])
def get_report(report_id):
    """Return a single report by ID."""
    # sanitize
    safe_id = "".join(c for c in report_id if c.isalnum() or c == "-")
    fpath = os.path.join(ANALYSIS_REPORTS_DIR, f"{safe_id}.json")
    if not os.path.exists(fpath):
        return jsonify({"error": "Report not found"}), 404
    with open(fpath) as f:
        return jsonify(_json.load(f))


@app.route("/reports/<report_id>", methods=["DELETE"])
def delete_report(report_id):
    """Delete a report by ID."""
    safe_id = "".join(c for c in report_id if c.isalnum() or c == "-")
    fpath = os.path.join(ANALYSIS_REPORTS_DIR, f"{safe_id}.json")
    if not os.path.exists(fpath):
        return jsonify({"error": "Report not found"}), 404
    os.remove(fpath)
    return jsonify({"deleted": safe_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)