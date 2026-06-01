"""
build_library.py — Run ONCE to build the music database. Re-run to update.

Dependencies:
    pip install librosa ollama faster-whisper numpy
    pip install torch transformers  # for CLAP embeddings

    CLAP model will auto-download on first run (~900MB).
    Whisper model: uses faster-whisper "medium" (same as tara_server.py).

Overnight run estimate for ~300 songs:
    - librosa feature extraction:  ~1-2 hrs
    - Whisper transcription:       ~2-4 hrs
    - CLAP embedding:              ~4-8 hrs (CPU)
    - LLM classification:          ~2-3 hrs
    Total: 8-16 hrs depending on song length and hardware.
"""

import os
import json
import hashlib
import warnings
import numpy as np
import librosa
import ollama
from datetime import datetime
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

SUPPORTED = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
WHISPER_MODEL_SIZE = "large-v3"       # same model already in tara_server.py
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
LLM_MODEL = "gemma2:9b"
LOAD_DURATION_SECS = 120            # analyze first 2 mins (vs 60s before)
WHISPER_MIN_WORDS = 10              # fewer words = treat as instrumental/unclear


# ── Lazy singletons — loaded once, reused across all songs ───────────────────

_whisper = None
_clap_model = None
_clap_processor = None


def get_whisper():
    global _whisper
    if _whisper is None:
        print("  📥 Loading Whisper model...")
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper


def get_clap():
    global _clap_model, _clap_processor
    if _clap_model is None:
        print("  📥 Loading CLAP model (first run: ~900MB download)...")
        from transformers import ClapModel, ClapProcessor
        _clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        _clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID)
        _clap_model.eval()
        print("  ✅ CLAP loaded.")
    return _clap_model, _clap_processor


# ── Layer 1: Richer librosa features ─────────────────────────────────────────

def extract_features(filepath: str) -> dict:
    """
    Extended librosa analysis covering:
      - Rhythm: tempo, onset strength
      - Energy: RMS, dynamic range
      - Timbre: MFCCs (mean + variance), spectral contrast, centroid, rolloff
      - Harmony: chroma key, harmonic-to-percussive ratio
      - Texture: ZCR (vocal hint), spectral flatness
    """
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=LOAD_DURATION_SECS, mono=True)

        # Rhythm
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = float(np.mean(onset_env))
        onset_std = float(np.std(onset_env))       # high std = punchy/rhythmic

        # Energy & dynamics
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))               # high std = dynamic range

        # Harmonic / percussive separation
        y_harm, y_perc = librosa.effects.hpss(y)
        harmonic_ratio = float(np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-6))

        # Timbre — MFCCs (13 coefficients, mean + variance)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_var = np.var(mfcc, axis=1).tolist()

        # Spectral shape
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))  # 0=tonal, 1=noisy

        # Spectral contrast — difference between peaks and valleys per band
        # High contrast = clear instruments; low contrast = dense/washy
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1).tolist()

        # ZCR — proxy for vocal presence and noisiness
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        has_vocals_hint = zcr > 0.05

        # Key from chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_idx = int(np.argmax(np.mean(chroma, axis=1)))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Tempo stability — low variance = steady beat
        if len(beats) > 2:
            beat_intervals = np.diff(beats)
            tempo_stability = float(1.0 / (np.std(beat_intervals) + 1e-6))
            tempo_stability = min(tempo_stability, 100.0)   # cap for readability
        else:
            tempo_stability = 0.0

        return {
            # Core
            "tempo_bpm": round(tempo, 1),
            "tempo_stability": round(tempo_stability, 2),   # high = steady groove
            "duration_secs": round(librosa.get_duration(y=y, sr=sr), 1),
            "key": keys[key_idx],

            # Energy & dynamics
            "energy": round(rms_mean * 1000, 3),
            "dynamic_range": round(rms_std * 1000, 3),      # high = dramatic swells

            # Rhythm texture
            "onset_strength": round(onset_mean, 3),         # high = punchy hits
            "onset_variance": round(onset_std, 3),          # high = irregular rhythm

            # Harmonic texture
            "harmonic_ratio": round(harmonic_ratio, 3),     # >1 = melodic; <1 = percussive
            "spectral_flatness": round(spec_flatness, 4),   # ~0 = tonal; ~1 = noise/distortion

            # Brightness & presence
            "spectral_centroid_hz": round(spec_centroid, 1),
            "spectral_rolloff_hz": round(spec_rolloff, 1),
            "spectral_contrast": [round(v, 2) for v in spec_contrast_mean],

            # Timbre fingerprint (for MFCC-based similarity)
            "mfcc_mean": [round(v, 3) for v in mfcc_mean],
            "mfcc_var": [round(v, 3) for v in mfcc_var],

            # Vocal hint
            "has_vocals_hint": has_vocals_hint,
            "zcr": round(zcr, 5),

            "error": None
        }

    except Exception as e:
        return {"error": str(e)}


# ── Layer 2: Whisper transcription ────────────────────────────────────────────

def transcribe_lyrics(filepath: str) -> dict:
    """
    Transcribe vocals using faster-whisper.
    Returns text + confidence flag. Treats short/empty results as instrumental.
    """
    try:
        wm = get_whisper()
        segments, info = wm.transcribe(
            filepath,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,   # reduces hallucination loops
            no_speech_threshold=0.6,            # skip silent/non-vocal segments
            log_prob_threshold=-1.0
        )

        words = []
        total_no_speech_prob = 0.0
        seg_count = 0

        for seg in segments:
            # Skip segments Whisper itself is unsure about
            if seg.no_speech_prob > 0.7:
                continue
            words.append(seg.text.strip())
            total_no_speech_prob += seg.no_speech_prob
            seg_count += 1

        full_text = " ".join(words).strip()
        word_count = len(full_text.split())

        if word_count < WHISPER_MIN_WORDS:
            return {
                "lyrics": None,
                "lyric_confidence": "low",
                "likely_instrumental": True,
                "detected_language": info.language,
                "error": None
            }

        avg_no_speech = total_no_speech_prob / seg_count if seg_count > 0 else 1.0
        confidence = "high" if avg_no_speech < 0.3 else "medium"

        return {
            "lyrics": full_text,
            "lyric_confidence": confidence,
            "likely_instrumental": False,
            "detected_language": info.language,
            "error": None
        }

    except Exception as e:
        return {
            "lyrics": None,
            "lyric_confidence": "error",
            "likely_instrumental": False,
            "detected_language": None,
            "error": str(e)
        }


# ── Layer 3: CLAP audio embedding ────────────────────────────────────────────

def extract_clap_embedding(filepath: str) -> dict:
    """
    Encode audio into CLAP's shared audio-text embedding space.
    The resulting 512-dim vector can be compared against text queries
    using cosine similarity — enabling semantic search without keywords.

    We average embeddings from 3 clips (start, middle, end) to capture
    the full character of the song rather than just the intro.
    """
    try:
        import torch
        model, processor = get_clap()

        # Load full song for multi-clip sampling
        y, sr = librosa.load(filepath, sr=48000, mono=True)   # CLAP expects 48kHz
        total_secs = len(y) / sr

        # Sample 3 clips of 10s each: start (~10s in), middle, near-end
        clip_len = 10 * sr
        offsets = []
        if total_secs > 30:
            offsets = [
                int(10 * sr),
                int((total_secs / 2) * sr),
                int(max(0, (total_secs - 20)) * sr)
            ]
        else:
            offsets = [0]

        embeddings = []
        for offset in offsets:
            clip = y[offset: offset + clip_len]
            if len(clip) < sr:          # skip clips shorter than 1s
                continue

            inputs = processor(
                audio=clip,
                sampling_rate=48000,
                return_tensors="pt"
            )
            with torch.no_grad():
                emb = model.get_audio_features(**inputs)
                if hasattr(emb, "pooler_output"):
                    emb = emb.pooler_output
                elif hasattr(emb, "last_hidden_state"):
                    emb = emb.last_hidden_state[:, 0, :]  # CLS token
                    # L2-normalize so cosine sim = dot product
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.squeeze().numpy().tolist())

        if not embeddings:
            return {"embedding": None, "error": "No valid clips extracted"}

        # Average across clips → single representative vector
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        return {"embedding": avg_embedding, "error": None}

    except Exception as e:
        return {"embedding": None, "error": str(e)}


# ── Layer 4: LLM classification ──────────────────────────────────────────────

def classify_with_llm(filename: str, features: dict, transcription: dict) -> dict:
    """
    Full classification using rich acoustic features + lyrics (if available).
    Produces affect dimensions, prose theme description, and narrative structure.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    lyrics = transcription.get("lyrics")
    is_instrumental = transcription.get("likely_instrumental", False)

    # Build a human-readable feature summary for the LLM
    hr_ratio = features.get("harmonic_ratio", 1.0)
    instrumentation_hint = (
        "likely melodic/harmonic (strings, synths, piano)" if hr_ratio > 2.0
        else "likely percussive (drums, beats, heavy rhythm)" if hr_ratio < 0.5
        else "balanced mix of melodic and rhythmic elements"
    )

    contrast = features.get("spectral_contrast", [])
    brightness = "bright/airy" if features.get("spectral_centroid_hz", 0) > 3000 else "dark/warm"
    texture = "noisy/distorted" if features.get("spectral_flatness", 0) > 0.1 else "clean/tonal"
    dynamic = "highly dynamic (dramatic swells)" if features.get("dynamic_range", 0) > 2.0 else "consistent energy level"
    rhythm = "steady/groovy" if features.get("tempo_stability", 0) > 5 else "loose/organic rhythm"

    lyrics_section = ""
    if lyrics:
        # Truncate to avoid token overload — first 400 words is plenty
        words = lyrics.split()[:400]
        lyrics_section = f"""
Transcribed lyrics (confidence: {transcription.get('lyric_confidence')}):
\"\"\"{' '.join(words)}\"\"\"
"""
    else:
        lyrics_section = f"\nNo lyrics detected — {'confirmed instrumental' if is_instrumental else 'unclear/ambient/heavy production'}."

    prompt = f"""You are a professional music librarian and music psychologist.
Analyze this song using its audio measurements and {'lyrics' if lyrics else 'audio character alone'}.

Filename: {name}

Audio measurements:
  - Tempo: {features.get('tempo_bpm')} BPM ({rhythm})
  - Energy: {features.get('energy')} ({dynamic})
  - Key: {features.get('key')}
  - Instrumentation character: {instrumentation_hint}
  - Tonal character: {brightness}, {texture}
  - Onset strength: {features.get('onset_strength')} (high = punchy beats)
  - Has vocals: {features.get('has_vocals_hint')}
  - Duration: {features.get('duration_secs')} seconds
{lyrics_section}

Respond ONLY with a valid JSON object. No markdown, no explanation:
{{
  "title": "Clean human-readable song title guessed from filename",
  "artist": "Artist name if identifiable, else null",
  "type": "one of: instrumental, vocal, ambient, classical, soundtrack, electronic",

  "affect": {{
    "arousal": <1-10, physical activation energy>,
    "valence": <1-10, positivity vs darkness>,
    "tension": <1-10, harmonic and rhythmic restlessness>,
    "drive": <1-10, forward momentum and propulsion>
  }},

  "theme_description": "2-3 sentences describing the emotional world, imagery, and atmosphere of this song. Be specific and evocative, not generic.",

  "narrative": {{
    "setting": "brief physical/temporal setting (e.g. late night city, open road, intimate room)",
    "emotional_arc": "how the song moves emotionally (e.g. tension to release, steady melancholy, building euphoria)",
    "perspective": "one of: introspective, expansive, communal, confrontational, dreamlike",
    "imagery": "3-4 evocative words or short phrases that capture the song's visual/sensory world"
  }},

  "instrumentation": {{
    "primary": ["list of 1-3 dominant instruments or sound sources"],
    "character": "one of: sparse, layered, dense, stripped, orchestral, electronic, acoustic, hybrid"
  }},

  "genres": ["list of 1-3 genre labels"],
  "good_for": ["list of 2-4 use cases: focus, sleep, workout, party, dinner, driving, meditation, mourning, celebration, background"],

  "confidence": "one of: high (had lyrics), medium (instrumental but clear character), low (ambiguous)"
}}"""

    try:
        response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are a music classifier. Always respond with valid JSON only. No markdown, no extra text, no thinking tags.'
            },
            {'role': 'user', 'content': prompt}
        ])
        raw = response['message']['content'].strip()

        # Strip thinking tags if model outputs them (qwen3 sometimes does)
        import re
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except Exception as e:
        return {
            "title": name,
            "artist": None,
            "type": "unknown",
            "affect": {"arousal": 5, "valence": 5, "tension": 5, "drive": 5},
            "theme_description": "Classification failed.",
            "narrative": {},
            "instrumentation": {},
            "genres": [],
            "good_for": [],
            "confidence": "low",
            "llm_error": str(e)
        }


# ── File hashing ──────────────────────────────────────────────────────────────

def file_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read(65536))
    return h.hexdigest()


# ── Progress tracking ─────────────────────────────────────────────────────────

def print_progress(i, total, filepath, stage):
    name = os.path.basename(filepath)[:50]
    bar_len = 30
    filled = int(bar_len * i / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(100 * i / total)
    print(f"\r  [{bar}] {pct}% ({i}/{total}) {stage}: {name}", end="", flush=True)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_library(music_dir: str, db_path: str, force: bool = False):
    # Load existing DB
    if os.path.exists(db_path) and not force:
        with open(db_path) as f:
            db = json.load(f)
        print(f"📚 Loaded existing library: {len(db['songs'])} songs")
    else:
        db = {"songs": {}, "version": 2, "built_at": datetime.now().isoformat()}
        print("📚 Starting fresh library (version 2)")

    # ── Remove deleted files ───────────────────────────────────────────────
    deleted = [path for path in db["songs"] if not os.path.exists(path)]
    if deleted:
        print(f"\n🗑️  Removing {len(deleted)} deleted song(s) from library:")
        for path in deleted:
            title = db["songs"][path].get("classification", {}).get("title") or os.path.basename(path)
            print(f"   ✗ {title}")
            del db["songs"][path]
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=2)
        print()

    # Scan music folder — top level ONLY, no subdirectories
    all_files = []
    for fname in os.listdir(music_dir):
        filepath = os.path.join(music_dir, fname)
        if os.path.isfile(filepath) and os.path.splitext(fname)[1].lower() in SUPPORTED:
            all_files.append(filepath)

    print(f"🎵 Found {len(all_files)} music files\n")

    new_count = 0
    errors = []

    for i, filepath in enumerate(all_files, 1):
        fhash = file_hash(filepath)
        name = os.path.basename(filepath)

        existing = db["songs"].get(filepath, {})
        already_done = (
            existing.get("hash") == fhash
            and existing.get("features")
            and existing.get("transcription")
            and existing.get("clap_embedding")
            and existing.get("classification")
            and not force
        )

        if already_done:
            print(f"  [{i}/{len(all_files)}] ✅ Skip: {name}")
            continue

        print(f"\n  [{i}/{len(all_files)}] ── {name}")

        # ── Stage 1: Acoustic features ────────────────────────────────────────
        print(f"    🎼 Extracting audio features...", end=" ", flush=True)
        features = extract_features(filepath)
        if features.get("error"):
            print(f"FAILED: {features['error']}")
            errors.append((filepath, "features", features["error"]))
            continue
        print(f"✓  {features['tempo_bpm']} BPM | energy={features['energy']} | key={features['key']} | h/p={features['harmonic_ratio']}")

        # ── Stage 2: Whisper transcription ────────────────────────────────────
        print(f"    🎤 Transcribing lyrics...", end=" ", flush=True)
        transcription = transcribe_lyrics(filepath)
        if transcription.get("error"):
            print(f"WARNING: {transcription['error']}")
        elif transcription["likely_instrumental"]:
            print(f"✓  Instrumental/no clear vocals detected")
        else:
            word_count = len((transcription.get("lyrics") or "").split())
            conf = transcription.get("lyric_confidence", "?")
            print(f"✓  {word_count} words transcribed (confidence: {conf})")

        # ── Stage 3: CLAP embedding ───────────────────────────────────────────
        clap_result = {"embedding": None, "error": None}
        print(f"    🔮 Computing CLAP embedding...", end=" ", flush=True)
        clap_result = extract_clap_embedding(filepath)
        if clap_result.get("error"):
            print(f"WARNING: {clap_result['error']}")
        elif clap_result.get("embedding"):
            print(f"✓  512-dim vector ready")
        else:
            print(f"✗  No embedding")

        # ── Stage 4: LLM classification ───────────────────────────────────────
        print(f"    🤖 LLM classification...", end=" ", flush=True)
        classification = classify_with_llm(filepath, features, transcription)
        title = classification.get("title", name)
        affect = classification.get("affect", {})
        theme = classification.get("theme_description", "")[:80]
        print(f"✓  {title}")
        print(f"       affect: arousal={affect.get('arousal')} valence={affect.get('valence')} tension={affect.get('tension')} drive={affect.get('drive')}")
        print(f"       theme: {theme}...")

        # ── Save entry ────────────────────────────────────────────────────────
        db["songs"][filepath] = {
            "hash": fhash,
            "filepath": filepath,
            "features": features,
            "transcription": {
                "lyrics": transcription.get("lyrics"),
                "lyric_confidence": transcription.get("lyric_confidence"),
                "likely_instrumental": transcription.get("likely_instrumental"),
                "detected_language": transcription.get("detected_language"),
            },
            "clap_embedding": clap_result.get("embedding"),
            "classification": classification,
            "play_history": existing.get("play_history", [])
        }
        new_count += 1

        # Save after every song
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'─'*60}")
    print(f"✅  Done! {new_count} new/updated songs.")
    print(f"   Total in library:    {len(db['songs'])}")
    print(f"   With CLAP vectors:   {sum(1 for s in db['songs'].values() if s.get('clap_embedding'))}")
    print(f"   With lyrics:         {sum(1 for s in db['songs'].values() if s.get('transcription', {}).get('lyrics'))}")
    print(f"   Instrumentals:       {sum(1 for s in db['songs'].values() if s.get('transcription', {}).get('likely_instrumental'))}")

    if errors:
        print(f"\n⚠️  {len(errors)} errors:")
        for fp, stage, err in errors:
            print(f"   {stage}: {os.path.basename(fp)} — {err}")

    print(f"\n   Library saved to: {db_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    music_dir = input("📁 Music folder path: ").strip()
    if not os.path.isdir(music_dir):
        print(f"❌ Not a directory: {music_dir}")
        exit(1)

    # Confirm what was found before committing
    found = [
        f for f in os.listdir(music_dir)
        if os.path.isfile(os.path.join(music_dir, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED
    ]
    print(f"\n🎵 Found {len(found)} music files in: {music_dir}")
    print(f"   (subdirectories are ignored — only files directly in this folder are analyzed)")
    confirm = input("\nProceed? (Y/n): ").strip().lower()
    if confirm == "n":
        print("Aborted.")
        exit(0)

    force_input = input("🔄 Re-analyze all songs from scratch? (y/N): ").strip().lower()
    force = force_input == "y"

    db_path = os.path.join(music_dir, "music_library.json")
    build_library(music_dir, db_path, force=force)