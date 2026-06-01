"""
tara_system.py — Raspberry Pi side TARA client.

Changes in this version:
  - No permanent data stored on Pi except daily logs.
      Wake word clips, interaction audio → streamed to Mac via HTTP.
  - Chunked song streaming: Mac sends raw PCM over HTTP; Pi buffers
    ~2 s then starts cvlc, appending chunks as they arrive. No silence gap.
  - Fixed TTS sequencing: single play_tts_bytes() function used for
    spoken intro AND post-song prompt. Adds 500 ms BT pre-silence,
    waits 300 ms after cvlc exits so BT codec drains before next audio.
  - BT watchdog: background thread re-sets BT sink as default every 30 s
    if it becomes available, handling mid-session reconnects.
  - Startup BT timeout raised to 45 s.
  - espeak offline fallback removed (espeak not available on this Pi).
  - All wake clip dirs, interaction log dirs removed from Pi filesystem.
"""

import os
import re
import signal
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import io
import time
import subprocess
import threading
import queue
import uuid
import json
import wave
import struct
import tempfile
from collections import deque
from openwakeword.model import Model
import logging
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
# Daily logs stay on the Pi; everything else goes to the Mac.

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tara_logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"tara_{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
log = logging.getLogger("TARA")

# ── Configuration ─────────────────────────────────────────────────────────────

MAC_BASE_URL          = "http://192.168.29.48:8000"
MAC_SERVER_URL        = f"{MAC_BASE_URL}/process_audio"
FAST_TEXT_URL         = f"{MAC_BASE_URL}/fast_text_command"
LOG_INTERACTION_URL   = f"{MAC_BASE_URL}/log_interaction"
REPORT_PLAYBACK_URL   = f"{MAC_BASE_URL}/report_playback"
REPORT_CORRECTION_URL = f"{MAC_BASE_URL}/report_correction"
MAC_HEALTH_URL        = f"{MAC_BASE_URL}/health"
TIMER_ALERT_URL       = f"{MAC_BASE_URL}/timer_alert"
NEXT_SONG_PROMPT_URL  = f"{MAC_BASE_URL}/next_song_prompt"
LABEL_WAKE_CLIP_URL   = f"{MAC_BASE_URL}/label_wake_clip"
STREAM_SONG_URL       = f"{MAC_BASE_URL}/stream_song"
LOG_WAKE_CLIP_URL     = f"{MAC_BASE_URL}/log_wake_clip"

WAKE_WORD_MODEL = os.path.abspath("hey_jarvis_v0.1.tflite")
MEL_MODEL       = os.path.abspath("melspectrogram.tflite")
EMBEDDING_MODEL = os.path.abspath("embedding_model.tflite")
WAKE_WORD_KEY   = "hey_jarvis_v0.1"

VOSK_MODEL_PATH = os.path.expanduser("~/vosk-model")

THINKING_WAV    = os.path.abspath("thinking.wav")
ALERT_WAV       = os.path.abspath("alert.wav")

BT_SPEAKER_MAC  = "60:AB:D2:0F:34:AA"
BT_SINK_NAME    = f"bluez_sink.{BT_SPEAKER_MAC.replace(':', '_')}.a2dp_sink"

# ── Audio config ──────────────────────────────────────────────────────────────

MIC_SAMPLE_RATE   = 48000
WAKE_WORD_RATE    = 16000
DOWNSAMPLE        = MIC_SAMPLE_RATE // WAKE_WORD_RATE
CHUNK_SIZE        = 1024
MIC_CHUNK         = CHUNK_SIZE * DOWNSAMPLE

MAX_DURATION         = 5
SILENCE_DURATION     = 0.8
SILENCE_DURATION_MID = 0.4
SILENCE_THRESHOLD    = 0.002

# ── Wake word detection tuning ────────────────────────────────────────────────

WAKE_THRESHOLD    = 0.35
SCORE_WINDOW      = 5
SCORE_FIRE_THRESH = 1.1
COOLDOWN_CHUNKS   = 15

RING_BUFFER_SECS   = 2
RING_BUFFER_CHUNKS = int(RING_BUFFER_SECS * WAKE_WORD_RATE / CHUNK_SIZE)

# ── TTS / playback constants ──────────────────────────────────────────────────

# Pre-silence prepended to every TTS clip to fill the BT codec buffer
# before the audible part starts.  500 ms is enough for most BT stacks.
BT_PRE_SILENCE_SECS  = 0.50

# Extra wait AFTER cvlc exits before the next audio action (song start,
# listen window open, etc.).  Lets the BT codec drain its output buffer
# so the tail of the TTS is not clipped.
BT_POST_DRAIN_SECS   = 0.30

DIM_VOLUME   = "20%"
FULL_VOLUME  = "100%"
IDLE_VOLUME  = "50%"
SERVER_TIMEOUT = 120

# ── Chunked streaming config ──────────────────────────────────────────────────

# How many bytes of raw PCM to buffer before telling cvlc to start.
# At 44100 Hz, 16-bit stereo: 44100*2*2 = 176400 bytes/s
# 2 seconds ≈ 352800 bytes.  We use 300000 (~1.7 s) as the trigger.
STREAM_PREBUFFER_BYTES = 300_000

# Size of each HTTP read chunk from the Mac streaming endpoint
STREAM_HTTP_CHUNK     = 65536   # 64 KB

# ── Bluetooth sink setup ──────────────────────────────────────────────────────

def _bt_sink_present() -> bool:
    """Return True if the BT sink is visible to PulseAudio right now."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sinks", "short"],
            capture_output=True, text=True, timeout=3
        )
        return BT_SINK_NAME in result.stdout
    except Exception:
        return False

def _set_bt_sink_default():
    """Silently set BT sink as PulseAudio default. Call only when sink is present."""
    try:
        subprocess.run(
            ["pactl", "set-default-sink", BT_SINK_NAME],
            timeout=3, capture_output=True
        )
        log.debug("BT sink set as default.")
    except Exception as e:
        log.debug(f"_set_bt_sink_default: {e}")

def wait_for_bt_sink(timeout: int = 45) -> bool:
    """Block until BT sink appears in PulseAudio or timeout expires."""
    log.info(f"Waiting for BT sink (up to {timeout}s)...")
    for _ in range(timeout):
        if _bt_sink_present():
            _set_bt_sink_default()
            log.info(f"BT sink ready: {BT_SINK_NAME}")
            print(f"🔊 Bluetooth speaker connected.")
            return True
        time.sleep(1)
    log.warning(f"BT sink not found after {timeout}s — using system default.")
    print("⚠️  Bluetooth speaker not found — using default audio output.")
    return False

def connect_bt_speaker(mac: str):
    try:
        subprocess.run(["bluetoothctl", "connect", mac],
                       timeout=10, capture_output=True)
        log.info(f"bluetoothctl connect {mac} issued")
    except Exception as e:
        log.warning(f"BT connect failed: {e}")

def _bt_watchdog():
    """
    Background thread: every 30 s, if the BT sink is available but not
    the current default, re-set it.  Handles mid-session reconnections
    without any manual intervention.
    """
    while True:
        time.sleep(30)
        try:
            if _bt_sink_present():
                # Check if it's already the default
                result = subprocess.run(
                    ["pactl", "get-default-sink"],
                    capture_output=True, text=True, timeout=3
                )
                current = result.stdout.strip()
                if BT_SINK_NAME not in current:
                    _set_bt_sink_default()
                    log.info("BT watchdog: re-set BT sink as default.")
        except Exception as e:
            log.debug(f"BT watchdog error: {e}")

print(f"🔊 Connecting Bluetooth speaker ({BT_SPEAKER_MAC})...")
connect_bt_speaker(BT_SPEAKER_MAC)
bt_ready = wait_for_bt_sink(timeout=45)

threading.Thread(target=_bt_watchdog, daemon=True).start()

# ── Shared playback state ─────────────────────────────────────────────────────

class PlaybackState:
    def __init__(self):
        self._lock      = threading.Lock()
        self.proc       = None
        self.is_playing = False
        self.is_paused  = False
        self.title      = ""
        self.filepath   = ""   # logical identifier only (Mac-side path)
        self.start_time: float = 0.0

    def start(self, proc, title: str, filepath: str):
        with self._lock:
            self.proc       = proc
            self.is_playing = True
            self.is_paused  = False
            self.title      = title
            self.filepath   = filepath
            self.start_time = time.time()

    def clear(self):
        with self._lock:
            self.proc       = None
            self.is_playing = False
            self.is_paused  = False
            self.title      = ""
            self.filepath   = ""
            self.start_time = 0.0

    def elapsed_seconds(self) -> float:
        with self._lock:
            if self.start_time == 0.0:
                return 0.0
            return time.time() - self.start_time

    def get_proc(self):
        with self._lock: return self.proc

    def playing(self) -> bool:
        with self._lock: return self.is_playing

    def paused(self) -> bool:
        with self._lock: return self.is_paused

    def set_paused(self, value: bool):
        with self._lock: self.is_paused = value

    def get_title(self) -> str:
        with self._lock: return self.title

    def get_filepath(self) -> str:
        with self._lock: return self.filepath

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "title":      self.title,
                "filepath":   self.filepath,
                "is_playing": self.is_playing,
                "is_paused":  self.is_paused,
                "elapsed":    round(time.time() - self.start_time, 1) if self.start_time else 0.0,
            }

playback = PlaybackState()
processing_lock = threading.Lock()

_post_song_listen      = False
_post_song_listen_lock = threading.Lock()
POST_SONG_LISTEN_SECS  = 12

def mac_is_reachable() -> bool:
    try:
        r = requests.get(MAC_HEALTH_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# ── Wake word model ───────────────────────────────────────────────────────────

print("🔄 Initializing T.A.R.A. Wake Word Engine...")
oww_model = Model(
    wakeword_models=[WAKE_WORD_MODEL],
    melspec_model_path=MEL_MODEL,
    embedding_model_path=EMBEDDING_MODEL
)

# ── Vosk fast-path ────────────────────────────────────────────────────────────

VOSK_AVAILABLE  = False
vosk_recognizer = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    if os.path.exists(VOSK_MODEL_PATH):
        print(f"🔄 Loading Vosk model from {VOSK_MODEL_PATH}...")
        _vosk_model     = VoskModel(VOSK_MODEL_PATH)
        vosk_recognizer = KaldiRecognizer(_vosk_model, WAKE_WORD_RATE)
        vosk_recognizer.SetWords(True)
        VOSK_AVAILABLE  = True
        print("✅ Vosk loaded.")
    else:
        print(f"⚠️  Vosk model not found at {VOSK_MODEL_PATH}.")
except ImportError:
    print("⚠️  vosk not installed. Fast path disabled.")
except Exception as e:
    print(f"⚠️  Vosk failed to load: {e}. Fast path disabled.")

vosk_lock = threading.Lock()

# ── Fuzzy matching ────────────────────────────────────────────────────────────

RAPIDFUZZ_AVAILABLE = False
try:
    from rapidfuzz import fuzz as _fuzz
    RAPIDFUZZ_AVAILABLE = True
    print("✅ rapidfuzz loaded.")
except ImportError:
    print("⚠️  rapidfuzz not installed. Falling back to exact keyword matching.")

def _fuzzy_match_any(text: str, keywords: set, threshold: int = 78) -> bool:
    t = text.lower()
    for kw in keywords:
        if kw in t:
            return True
        if not RAPIDFUZZ_AVAILABLE:
            continue
        kw_words = kw.split()
        n        = len(kw_words)
        t_words  = t.split()
        for i in range(max(1, len(t_words) - n + 1)):
            window = " ".join(t_words[i: i + n])
            if _fuzz.ratio(kw, window) >= threshold:
                return True
    return False

# ── Volume helpers ────────────────────────────────────────────────────────────

def set_idle_volume():
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", IDLE_VOLUME],
            timeout=2
        )
        log.info(f"System volume reset to idle baseline: {IDLE_VOLUME}")
    except Exception as e:
        log.warning(f"Failed to reset system volume: {e}")

def get_sink_input_index(pid: int):
    try:
        result = subprocess.run(
            ["pactl", "list", "sink-inputs"],
            capture_output=True, text=True, timeout=3
        )
        current_index = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Sink Input #"):
                current_index = line.split("#")[1].strip()
            if f'application.process.id = "{pid}"' in line:
                return current_index
    except Exception as e:
        log.warning(f"pactl list failed: {e}")
    return None

def set_sink_volume(index: str, volume: str):
    try:
        subprocess.run(
            ["pactl", "set-sink-input-volume", index, volume], timeout=2
        )
    except Exception as e:
        log.warning(f"pactl volume set failed: {e}")

def dim_song():
    proc = playback.get_proc()
    if not proc:
        return None
    idx = get_sink_input_index(proc.pid)
    if idx:
        set_sink_volume(idx, DIM_VOLUME)
    return idx

def restore_song(idx):
    if idx:
        set_sink_volume(idx, FULL_VOLUME)

# ── Core TTS playback function ────────────────────────────────────────────────
# Single function used for ALL spoken output (intro, post-song prompt,
# general TTS responses).  Handles BT pre-silence and post-drain wait.

def _run_cvlc_blocking(wav_path: str, duration_secs: float):
    """Run cvlc synchronously; kill it if it hangs beyond grace period."""
    grace = duration_secs + 6.0
    proc  = subprocess.Popen(["cvlc", "--play-and-exit", wav_path])
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        log.warning(f"cvlc hung after {grace:.0f}s — terminating")
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

def play_tts_bytes(wav_bytes: bytes) -> float:
    """
    Play TTS audio (WAV bytes) through the BT speaker.

    Steps:
      1. Prepend BT_PRE_SILENCE_SECS of silence so the codec warms up
         before the audible speech starts.
      2. Write to a temp file and play via cvlc.
      3. Wait BT_POST_DRAIN_SECS after cvlc exits so the codec drains
         and the tail of the speech is not clipped by the next audio event.

    Returns the duration of the audio in seconds (excluding silence padding),
    useful for callers that want to know how long the speech was.
    """
    try:
        buf = io.BytesIO(wav_bytes)
        audio, sr = sf.read(buf, dtype='int16')
    except Exception as e:
        log.error(f"play_tts_bytes: could not decode WAV: {e}")
        return 0.0

    # Prepend silence
    silence_samples = int(BT_PRE_SILENCE_SECS * sr)
    if audio.ndim == 1:
        silence = np.zeros(silence_samples, dtype='int16')
    else:
        silence = np.zeros((silence_samples, audio.shape[1]), dtype='int16')
    padded    = np.concatenate([silence, audio], axis=0)
    total_dur = len(padded) / sr
    channels  = 1 if padded.ndim == 1 else padded.shape[1]

    tmp = os.path.join(tempfile.gettempdir(), f"tara_tts_{uuid.uuid4().hex[:8]}.wav")
    try:
        with sf.SoundFile(tmp, mode='w', samplerate=sr,
                          channels=channels, subtype='PCM_16') as f:
            f.write(padded)
        set_idle_volume()
        _run_cvlc_blocking(tmp, total_dur)
    except Exception as e:
        log.error(f"play_tts_bytes playback failed: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    # Give the BT codec time to drain before the next sound starts
    time.sleep(BT_POST_DRAIN_SECS)
    speech_dur = len(audio) / sr
    return speech_dur

def play_tts_over_song(wav_bytes: bytes):
    """
    Play TTS while a song is playing: pause song → TTS → resume song.
    Uses play_tts_bytes() so BT handling is consistent.
    """
    was_playing = playback.playing()
    was_paused  = playback.paused()
    if was_playing and not was_paused:
        pause_playback()
        time.sleep(0.3)
    play_tts_bytes(wav_bytes)
    if was_playing and not was_paused:
        resume_playback()

# ── Chime ─────────────────────────────────────────────────────────────────────

def play_chime():
    sr          = 48000
    pre_silence = int(sr * BT_PRE_SILENCE_SECS)
    pulse_ms    = 0.14
    gap_ms      = 0.06
    peak        = 0.90

    pulse_len = int(sr * pulse_ms)
    gap_len   = int(sr * gap_ms)

    def chirp_pulse(f_start: float, f_end: float) -> np.ndarray:
        inst_freq = np.linspace(f_start, f_end, pulse_len)
        phase     = 2 * np.pi * np.cumsum(inst_freq) / sr
        wave      = np.sin(phase).astype(np.float32)
        atk  = int(sr * 0.010)
        rel  = int(sr * 0.020)
        env  = np.ones(pulse_len, dtype=np.float32)
        env[:atk]             = np.linspace(0.0, 1.0, atk)
        env[pulse_len - rel:] = np.linspace(1.0, 0.0, rel)
        return (wave * env * peak * 32767).astype(np.int16)

    gap    = np.zeros(gap_len, dtype=np.int16)
    pre    = np.zeros(pre_silence, dtype=np.int16)
    pulses = [
        chirp_pulse(800,  1200),
        chirp_pulse(1000, 1500),
        chirp_pulse(1200, 1800),
    ]
    chime = np.concatenate([pre] + [p for pulse in pulses for p in (pulse, gap)])

    tmp = os.path.join(tempfile.gettempdir(), f"tara_chime_{uuid.uuid4().hex[:8]}.wav")
    try:
        with sf.SoundFile(tmp, mode="w", samplerate=sr,
                          channels=1, subtype="PCM_16") as f:
            f.write(chime)
        set_idle_volume()
        _run_cvlc_blocking(tmp, len(chime) / sr)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ── Chunked song streaming and playback ───────────────────────────────────────

def play_song(mac_filepath: str, title: str, spoken_intro: str = ""):
    """
    Stream and play a song from the Mac.

    Flow:
      1. If spoken_intro provided: fetch TTS from Mac, play it via
         play_tts_bytes() (blocking, with BT drain wait).
      2. Open a streaming HTTP connection to /stream_song on the Mac.
      3. Write incoming chunks to a temp WAV file on Pi.
      4. Once STREAM_PREBUFFER_BYTES have arrived, start cvlc on that file.
         cvlc reads ahead as more data is appended — seamless playback.
      5. When the stream ends (Mac sends all data), the file is complete
         and cvlc finishes naturally.
      6. Report playback to Mac; fetch and play post-song prompt.
    """
    def _run():
        global _post_song_listen

        # ── 1. Spoken intro ───────────────────────────────────────────────
        if spoken_intro:
            log.info(f"Fetching spoken intro TTS: '{spoken_intro[:60]}'")
            try:
                resp = requests.post(
                    f"{MAC_BASE_URL}/tts",
                    json={"text": spoken_intro},
                    timeout=20,
                )
                if resp.ok and "audio" in resp.headers.get("Content-Type", ""):
                    log.info(f"TTS response: {len(resp.content)} bytes — playing intro")
                    play_tts_bytes(resp.content)   # blocks + BT drain wait
                    log.info("Spoken intro played OK")
                else:
                    log.warning(f"TTS bad response: {resp.status_code} — {resp.text[:100]}")
            except Exception as e:
                log.warning(f"Spoken intro failed (non-fatal): {e}")

        # ── 2. Open streaming connection ──────────────────────────────────
        log.info(f"Starting chunked stream for: {title} ({mac_filepath})")
        tmp_wav = os.path.join(
            tempfile.gettempdir(), f"tara_song_{uuid.uuid4().hex[:8]}.wav"
        )
        cvlc_proc = None
        bytes_received = 0
        started        = False

        try:
            with requests.get(
                STREAM_SONG_URL,
                params={"filepath": mac_filepath},
                stream=True,
                timeout=30,
            ) as stream_resp:

                if not stream_resp.ok:
                    log.error(f"stream_song failed: {stream_resp.status_code}")
                    return

                # ── 3. Write chunks to temp file ──────────────────────────
                with open(tmp_wav, "wb") as wav_out:
                    for chunk in stream_resp.iter_content(
                        chunk_size=STREAM_HTTP_CHUNK
                    ):
                        if not chunk:
                            continue
                        wav_out.write(chunk)
                        wav_out.flush()
                        bytes_received += len(chunk)

                        # ── 4. Start cvlc once pre-buffer is full ─────────
                        if not started and bytes_received >= STREAM_PREBUFFER_BYTES:
                            log.info(
                                f"Pre-buffer reached ({bytes_received} bytes) "
                                f"— starting cvlc"
                            )
                            cvlc_proc = subprocess.Popen(
                                ["cvlc", "--play-and-exit", tmp_wav]
                            )
                            playback.start(cvlc_proc, title, mac_filepath)
                            started = True

            # ── 5. Stream finished — wait for cvlc to finish ──────────────
            if cvlc_proc is None:
                # File was smaller than pre-buffer; start cvlc now
                log.info("File smaller than pre-buffer — starting cvlc after full download")
                cvlc_proc = subprocess.Popen(
                    ["cvlc", "--play-and-exit", tmp_wav]
                )
                playback.start(cvlc_proc, title, mac_filepath)

            cvlc_proc.wait()

        except Exception as e:
            log.error(f"Streaming playback error: {e}")
            if cvlc_proc and cvlc_proc.poll() is None:
                cvlc_proc.terminate()
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
                log.debug(f"Temp song file removed: {tmp_wav}")

        # ── 6. Post-playback reporting ────────────────────────────────────
        elapsed     = playback.elapsed_seconds()
        natural_end = (playback.get_proc() is cvlc_proc)

        if natural_end:
            playback.clear()
            set_idle_volume()
            log.info(f"Playback ended naturally: {title} ({elapsed:.0f}s)")
            print(f"\n🎵 Song ended: {title}")
            _report_playback(
                filepath=mac_filepath, title=title,
                elapsed=elapsed, completed=True
            )
            _trigger_post_song_prompt()

    threading.Thread(target=_run, daemon=True).start()


def stop_playback():
    snap = playback.snapshot()
    proc = playback.get_proc()
    if proc and proc.poll() is None:
        proc.terminate()
    elapsed = snap["elapsed"]
    playback.clear()
    set_idle_volume()
    log.info(f"Playback stopped after {elapsed:.0f}s")
    if snap["filepath"] and elapsed > 2:
        _report_playback(
            filepath=snap["filepath"],
            title=snap["title"],
            elapsed=elapsed,
            completed=False,
        )

def pause_playback():
    proc = playback.get_proc()
    if proc and proc.poll() is None and not playback.paused():
        proc.send_signal(signal.SIGSTOP)
        playback.set_paused(True)
        set_idle_volume()
        log.info("Playback paused")

def resume_playback():
    proc = playback.get_proc()
    if proc and proc.poll() is None and playback.paused():
        proc.send_signal(signal.SIGCONT)
        playback.set_paused(False)
        log.info("Playback resumed")

# ── Post-song prompt ──────────────────────────────────────────────────────────

def _trigger_post_song_prompt():
    """
    Fetch voiced follow-up prompt from Mac, play it via play_tts_bytes()
    (so BT handling is consistent), then open the listen window.
    """
    global _post_song_listen

    try:
        health = requests.get(MAC_HEALTH_URL, timeout=2)
        if health.status_code != 200:
            raise Exception(f"health returned {health.status_code}")
    except Exception as e:
        log.warning(f"Post-song prompt skipped — Mac unreachable: {e}")
        print("👂 Standing by... Say 'Hey Jarvis'")
        return

    log.info("Fetching post-song prompt from Mac...")
    try:
        resp = requests.get(NEXT_SONG_PROMPT_URL, timeout=15)
        log.info(
            f"Prompt response: status={resp.status_code} "
            f"size={len(resp.content)} bytes"
        )
        if not resp.ok or "audio" not in resp.headers.get("Content-Type", ""):
            log.warning(f"next_song_prompt bad response: {resp.status_code}")
            print("👂 Standing by... Say 'Hey Jarvis'")
            return

        prompt_text = resp.headers.get("X-Prompt-Text", "")
        if prompt_text:
            print(f"\n💬 TARA: {prompt_text}")
        log.info(f"Post-song prompt text: '{prompt_text}'")

        # play_tts_bytes blocks until audio is fully played + BT drains
        play_tts_bytes(resp.content)
        log.info("Post-song prompt played OK")

        # Open listen window AFTER prompt has fully played
        with _post_song_listen_lock:
            _post_song_listen = True
        log.info("Post-song listen window open")
        print("👂 Listening for your reply...")

        def _expire():
            time.sleep(POST_SONG_LISTEN_SECS)
            with _post_song_listen_lock:
                global _post_song_listen
                if _post_song_listen:
                    _post_song_listen = False
                    log.info("Post-song listen window expired")
                    print("👂 Standing by... Say 'Hey Jarvis'")
        threading.Thread(target=_expire, daemon=True).start()

    except Exception as e:
        log.warning(f"Post-song prompt failed: {e}")
        print("👂 Standing by... Say 'Hey Jarvis'")

# ── Reporting helpers ─────────────────────────────────────────────────────────

def _report_playback(filepath: str, title: str, elapsed: float, completed: bool):
    def _send():
        try:
            requests.post(
                REPORT_PLAYBACK_URL,
                json={
                    "filepath":  filepath,
                    "title":     title,
                    "elapsed":   round(elapsed, 1),
                    "completed": completed,
                    "timestamp": datetime.now().isoformat(),
                },
                timeout=5,
            )
            log.info(f"Reported playback: {title} completed={completed} elapsed={elapsed:.0f}s")
        except Exception as e:
            log.warning(f"report_playback failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def _report_correction(heard: str, intended: str):
    def _send():
        try:
            requests.post(
                REPORT_CORRECTION_URL,
                json={
                    "heard":     heard,
                    "intended":  intended,
                    "timestamp": datetime.now().isoformat(),
                },
                timeout=5,
            )
        except Exception as e:
            log.warning(f"report_correction failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def _log_interaction_to_mac(
    session_id: str,
    trigger_wav_bytes: bytes,
    command_wav_bytes: bytes,
    vosk_transcript: str,
    fast_path_result: str,
    playback_state: dict,
):
    """Send interaction audio + metadata to Mac for storage. No local files kept."""
    def _send():
        try:
            metadata = {
                "session_id":       session_id,
                "timestamp":        datetime.now().isoformat(),
                "vosk_transcript":  vosk_transcript,
                "fast_path_result": fast_path_result,
                "playback_state":   playback_state,
            }
            files = {
                "metadata": (
                    "metadata.json",
                    json.dumps(metadata).encode(),
                    "application/json",
                ),
            }
            if command_wav_bytes:
                files["command_wav"] = (
                    f"cmd_{session_id}.wav",
                    command_wav_bytes,
                    "audio/wav",
                )
            if trigger_wav_bytes:
                files["trigger_wav"] = (
                    f"wake_{session_id}.wav",
                    trigger_wav_bytes,
                    "audio/wav",
                )
            requests.post(LOG_INTERACTION_URL, files=files, timeout=15)
            log.info(f"Interaction logged to Mac: session={session_id}")
        except Exception as e:
            log.warning(f"log_interaction failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def _send_wake_clip_to_mac(wav_bytes: bytes, meta: dict):
    """Send ring-buffer clip + metadata to Mac for labelling and storage."""
    def _send():
        try:
            files = {
                "file": ("wake_clip.wav", wav_bytes, "audio/wav"),
                "metadata": (
                    "metadata.json",
                    json.dumps(meta).encode(),
                    "application/json",
                ),
            }
            requests.post(LOG_WAKE_CLIP_URL, files=files, timeout=15)
            log.info("Wake clip sent to Mac for labelling.")
        except Exception as e:
            log.debug(f"Wake clip send failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

# ── Mic queue and ring buffer ─────────────────────────────────────────────────

mic_queue:       queue.Queue = queue.Queue(maxsize=200)
wake_ring_buffer: deque      = deque(maxlen=RING_BUFFER_CHUNKS)
ring_buffer_lock              = threading.Lock()

def flush_queue(seconds: float = 1.0):
    n = int(seconds * MIC_SAMPLE_RATE / MIC_CHUNK) + 1
    for _ in range(n):
        try:
            mic_queue.get_nowait()
        except queue.Empty:
            break

def reset_wake_model():
    for key in oww_model.prediction_buffer:
        oww_model.prediction_buffer[key].clear()

# ── Thinking sound ────────────────────────────────────────────────────────────

class ThinkingPlayer:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if not os.path.exists(THINKING_WAV):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        try:
            info = sf.info(THINKING_WAV)
        except Exception as e:
            log.warning(f"Cannot read thinking.wav: {e}")
            return
        while not self._stop_event.is_set():
            proc = subprocess.Popen(["cvlc", "--play-and-exit", THINKING_WAV])
            deadline = time.time() + info.duration + 2.0
            while time.time() < deadline:
                if self._stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
                time.sleep(0.05)
            if proc.poll() is None:
                proc.terminate()

thinking_player = ThinkingPlayer()

# ── Vosk transcription ────────────────────────────────────────────────────────

def vosk_transcribe(audio_48k: np.ndarray) -> str:
    if not VOSK_AVAILABLE:
        return ""
    try:
        audio_16k   = audio_48k[::DOWNSAMPLE].flatten()
        audio_int16 = (audio_16k * 32768).clip(-32768, 32767).astype(np.int16)
        raw_bytes   = audio_int16.tobytes()
        with vosk_lock:
            vosk_recognizer.AcceptWaveform(raw_bytes)
            result = json.loads(vosk_recognizer.Result())
            vosk_recognizer.Reset()
        return result.get("text", "").strip()
    except Exception as e:
        log.warning(f"Vosk transcription failed: {e}")
        return ""

# ── Fast-path keyword sets ────────────────────────────────────────────────────

_STOP_KW   = {
    "stop", "turn off", "kill it", "stop it", "stop music",
    "stop the music", "stop playing", "cut it", "cut",
}
_PAUSE_KW  = {
    "pause", "hold on", "wait", "pause it", "pause please",
    "pause the music", "pause the song", "hold it", "freeze",
    "stop for now", "wait a moment", "one second",
}
_RESUME_KW = {
    "resume", "continue", "unpause", "keep going", "carry on",
    "play again", "start again", "go ahead", "resume music",
}
_SKIP_KW   = {
    "skip", "next song", "next track", "play something else",
    "another song", "different song", "change the song",
    "not this one", "next one",
}

def handle_fast_path(transcription: str) -> tuple[bool, str]:
    t = transcription.lower()
    if _fuzzy_match_any(t, _STOP_KW):
        log.info(f"Fast path: STOP — '{transcription}'")
        stop_playback()
        return True, "stop"
    if _fuzzy_match_any(t, _PAUSE_KW):
        log.info(f"Fast path: PAUSE — '{transcription}'")
        pause_playback()
        return True, "pause"
    if _fuzzy_match_any(t, _RESUME_KW):
        log.info(f"Fast path: RESUME — '{transcription}'")
        resume_playback()
        return True, "resume"
    if _fuzzy_match_any(t, _SKIP_KW):
        log.info("Fast path: SKIP — forwarding to slow path")
        return False, "skip_forward"
    return False, "no_match"

# ── Wake ring buffer capture ──────────────────────────────────────────────────

def capture_wake_ring_buffer(
    window_sum: float = 0.0, peak_score: float = 0.0
) -> bytes:
    """
    Capture the current ring buffer as WAV bytes and send them to Mac.
    Returns the raw WAV bytes (for attaching to the interaction log too).
    No files are written to Pi disk.
    """
    with ring_buffer_lock:
        chunks = list(wake_ring_buffer)
    if not chunks:
        return b""
    audio_int16 = np.concatenate(chunks, axis=0)

    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", samplerate=WAKE_WORD_RATE,
        channels=1, subtype="PCM_16", format="WAV"
    ) as wf:
        wf.write(audio_int16)
    wav_bytes = buf.getvalue()

    meta = {
        "timestamp":    datetime.now().isoformat(),
        "window_sum":   round(window_sum, 4),
        "peak_score":   round(peak_score, 4),
        "music_active": bool(playback.playing()),
    }
    _send_wake_clip_to_mac(wav_bytes, meta)
    return wav_bytes

def capture_command_audio(audio_data: np.ndarray) -> bytes:
    """
    Encode command audio as WAV bytes (in memory only; no Pi disk write).
    """
    audio_int16 = (audio_data.flatten() * 32768).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", samplerate=MIC_SAMPLE_RATE,
        channels=1, subtype="PCM_16", format="WAV"
    ) as wf:
        wf.write(audio_int16)
    return buf.getvalue()

# ── VAD recording ─────────────────────────────────────────────────────────────

def record_until_silence(mid_song: bool = False) -> np.ndarray:
    silence_dur    = SILENCE_DURATION_MID if mid_song else SILENCE_DURATION
    max_frames     = int(MAX_DURATION * MIC_SAMPLE_RATE)
    silence_needed = int(silence_dur * MIC_SAMPLE_RATE / MIC_CHUNK)
    recorded       = []
    total_frames   = 0
    silent_chunks  = 0
    speech_started = False
    max_rms_seen   = 0.0

    while total_frames < max_frames:
        try:
            chunk = mic_queue.get(timeout=2.0)
        except queue.Empty:
            log.warning("Mic queue timed out during recording")
            break
        recorded.append(chunk.copy())
        total_frames += MIC_CHUNK
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        max_rms_seen = max(max_rms_seen, rms)
        if rms > SILENCE_THRESHOLD:
            speech_started = True
            silent_chunks  = 0
        elif speech_started:
            silent_chunks += 1
            if silent_chunks >= silence_needed:
                break

    log.info(
        f"Recording done — {total_frames/MIC_SAMPLE_RATE:.1f}s, "
        f"peak RMS={max_rms_seen:.4f}, speech_started={speech_started}"
    )
    if not speech_started:
        log.warning(
            f"VAD never triggered! Peak RMS {max_rms_seen:.4f} vs "
            f"threshold {SILENCE_THRESHOLD}."
        )
    return np.concatenate(recorded, axis=0)

# ── Last Whisper transcript (for correction detection) ────────────────────────

_last_whisper_transcript      = ""
_last_whisper_transcript_lock = threading.Lock()

def _store_last_transcript(text: str):
    global _last_whisper_transcript
    with _last_whisper_transcript_lock:
        _last_whisper_transcript = text

def _get_last_transcript() -> str:
    with _last_whisper_transcript_lock:
        return _last_whisper_transcript

# ── Main record-and-process pipeline ─────────────────────────────────────────

def record_and_process(window_sum: float = 0.0, peak_score: float = 0.0):
    if not processing_lock.acquire(blocking=False):
        log.info("Already processing — ignoring wake word")
        return

    session_id        = uuid.uuid4().hex[:12]
    trigger_wav_bytes = b""
    command_wav_bytes = b""
    vosk_text         = ""
    fast_path_label   = "none"
    response_audio    = None
    mid_song_audio    = False
    lock_released     = False

    try:
        is_mid_song   = playback.playing()
        current_title = playback.get_title()
        pb_snapshot   = playback.snapshot()

        print(f"\n✨ T.A.R.A. Triggered! "
              f"{'[mid-song: ' + current_title + ']' if is_mid_song else ''}")
        log.info(f"Wake word detected — session={session_id} mid_song={is_mid_song}")

        # Capture ring buffer (sends to Mac in background, no Pi file)
        trigger_wav_bytes = capture_wake_ring_buffer(
            window_sum=window_sum, peak_score=peak_score
        )
        play_chime()
        flush_queue(seconds=0.3)

        audio_data = record_until_silence(mid_song=is_mid_song)
        print(f"   Recorded {len(audio_data)/MIC_SAMPLE_RATE:.1f}s of audio")

        # Encode command audio in memory
        command_wav_bytes = capture_command_audio(audio_data)

        if VOSK_AVAILABLE:
            vosk_text = vosk_transcribe(audio_data)
            if vosk_text:
                print(f"🎙️  Vosk: '{vosk_text}'")
                log.info(f"Vosk: '{vosk_text}'")

                last_t = _get_last_transcript()
                if last_t and vosk_text and last_t.lower() != vosk_text.lower():
                    last_words = last_t.lower().split()
                    vosk_words = vosk_text.lower().split()
                    if last_words and vosk_words and last_words[0] == vosk_words[0]:
                        _report_correction(heard=last_t, intended=vosk_text)

                handled, fast_path_label = handle_fast_path(vosk_text)
                if handled:
                    _log_interaction_to_mac(
                        session_id=session_id,
                        trigger_wav_bytes=trigger_wav_bytes,
                        command_wav_bytes=command_wav_bytes,
                        vosk_transcript=vosk_text,
                        fast_path_result=fast_path_label,
                        playback_state=pb_snapshot,
                    )
                    flush_queue(seconds=1.0)
                    reset_wake_model()
                    processing_lock.release()
                    lock_released = True
                    print("\n👂 Standing by... Say 'Hey Jarvis'")
                    return
                print(f"  ↩️  Vosk: no fast-path match ('{vosk_text}'), using slow path")
            else:
                print("  ↩️  Vosk returned empty — using slow path")

        if not mac_is_reachable():
            log.warning("Mac unreachable — no response")
            print("❌ Mac not reachable")
            _log_interaction_to_mac(
                session_id=session_id,
                trigger_wav_bytes=trigger_wav_bytes,
                command_wav_bytes=command_wav_bytes,
                vosk_transcript=vosk_text,
                fast_path_result="mac_offline",
                playback_state=pb_snapshot,
            )
            flush_queue(seconds=1.0)
            reset_wake_model()
            processing_lock.release()
            lock_released = True
            print("\n👂 Standing by... Say 'Hey Jarvis'")
            return

        if not playback.playing():
            thinking_player.start()

        # Encode audio for upload
        buf = io.BytesIO(command_wav_bytes)
        buf.seek(0)

        headers = {"X-Mid-Song": "true"} if is_mid_song else {}

        t_send = time.time()
        print(f"📡 Sending to Brain (up to {SERVER_TIMEOUT}s)...")

        try:
            response = requests.post(
                MAC_SERVER_URL,
                files={'file': ('audio.wav', buf)},
                headers=headers,
                timeout=SERVER_TIMEOUT,
            )

            thinking_player.stop()
            elapsed      = time.time() - t_send
            content_type = response.headers.get('Content-Type', '')
            log.info(f"Server responded in {elapsed:.1f}s — {content_type}")

            whisper_text = response.headers.get("X-Whisper-Transcript", "")
            if whisper_text:
                _store_last_transcript(whisper_text)

            _log_interaction_to_mac(
                session_id=session_id,
                trigger_wav_bytes=trigger_wav_bytes,
                command_wav_bytes=command_wav_bytes,
                vosk_transcript=vosk_text,
                fast_path_result="slow_path_" + content_type.split("/")[-1],
                playback_state=pb_snapshot,
            )

            if 'application/json' in content_type:
                data   = response.json()
                action = data.get("action", "")
                status = data.get("status", "")

                if status == "play_local":
                    filepath     = data.get("filepath", "")
                    title        = data.get("title", "Unknown")
                    artist       = data.get("artist", "")
                    reason       = data.get("reason", "")
                    spoken_intro = data.get("spoken_intro", "")

                    if action == "skip":
                        print(f"⏭️  Skipping to: {title}")
                        log.info(f"Skip -> {title}")
                        stop_playback()
                        time.sleep(0.3)
                    else:
                        display = f"{title}{' by ' + artist if artist else ''}"
                        print(f"\n🎵 Now playing: {display}")
                        print(f"   Reason: {reason}")
                        log.info(f"Playing: {title}")

                    flush_queue(seconds=1.0)
                    reset_wake_model()
                    processing_lock.release()
                    lock_released = True
                    print("\n👂 Standing by... Say 'Hey Jarvis'")

                    # filepath is the Mac-side path; passed to stream endpoint
                    play_song(filepath, title, spoken_intro=spoken_intro)
                    return

                elif action == "stop":
                    flush_queue(seconds=1.0)
                    reset_wake_model()
                    processing_lock.release()
                    lock_released = True
                    print("\n👂 Standing by... Say 'Hey Jarvis'")
                    stop_playback()
                    return

                elif action == "pause":
                    flush_queue(seconds=1.0)
                    reset_wake_model()
                    processing_lock.release()
                    lock_released = True
                    print("\n👂 Standing by... Say 'Hey Jarvis'")
                    pause_playback()
                    return

                elif action == "resume":
                    flush_queue(seconds=1.0)
                    reset_wake_model()
                    processing_lock.release()
                    lock_released = True
                    print("\n👂 Standing by... Say 'Hey Jarvis'")
                    resume_playback()
                    return

                elif status == "no_speech":
                    print("🤷 TARA didn't hear anything.")
                    log.info("No speech detected")

                elif status == "error":
                    print(f"❌ Server error: {data.get('message')}")
                    log.error(f"Server error: {data.get('message')}")

                else:
                    print(f"ℹ️  Server: {data}")

            elif 'audio' in content_type:
                is_mid_response = (
                    response.headers.get('X-Mid-Song-Response', 'false').lower() == 'true'
                )
                response_audio = response.content
                mid_song_audio = is_mid_response and playback.playing()

            else:
                print(f"ℹ️  Unknown response type: {content_type}")

        except requests.exceptions.Timeout:
            thinking_player.stop()
            print(f"❌ Mac took longer than {SERVER_TIMEOUT}s.")
            log.error("Request timeout")
        except Exception as e:
            thinking_player.stop()
            print(f"❌ Request error: {e}")
            log.error(f"Request error: {e}")

    finally:
        thinking_player.stop()
        if not lock_released:
            flush_queue(seconds=1.0)
            reset_wake_model()
            processing_lock.release()
            print("\n👂 Standing by... Say 'Hey Jarvis'")

    if response_audio:
        try:
            if mid_song_audio:
                log.info("Playing mid-song TTS with pause/resume")
                play_tts_over_song(response_audio)
            else:
                log.info("Playing TTS response")
                play_tts_bytes(response_audio)
        except Exception as e:
            log.error(f"Audio playback error: {e}")

# ── Mic reader thread ─────────────────────────────────────────────────────────

def _mic_reader():
    with sd.InputStream(
        device=None,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype='float32',
    ) as stream:
        log.info("Mic reader thread started")
        while True:
            mic_audio, _ = stream.read(MIC_CHUNK)
            audio_16k    = mic_audio[::DOWNSAMPLE].flatten()
            audio_int16  = (audio_16k * 32768).clip(-32768, 32767).astype(np.int16)
            with ring_buffer_lock:
                wake_ring_buffer.append(audio_int16)
            try:
                mic_queue.put_nowait(mic_audio)
            except queue.Full:
                try:
                    mic_queue.get_nowait()
                except queue.Empty:
                    pass
                mic_queue.put_nowait(mic_audio)

threading.Thread(target=_mic_reader, daemon=True).start()

# ── Timer alert poller ────────────────────────────────────────────────────────

def _poll_mac_for_alerts():
    while True:
        try:
            resp = requests.get(TIMER_ALERT_URL, timeout=2)
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "audio" in content_type:
                    alert_text = resp.headers.get("X-Alert-Text", "Timer finished.")
                    log.info(f"Mac timer alert received: {alert_text}")
                    print(f"\n⏰ Timer Fired: {alert_text}")
                    if playback.playing():
                        play_tts_over_song(resp.content)
                    else:
                        play_tts_bytes(resp.content)
        except Exception:
            pass
        time.sleep(2.0)

threading.Thread(target=_poll_mac_for_alerts, daemon=True).start()

# ── Main wake word detection loop ─────────────────────────────────────────────

print("✅ TARA Brain is Online.")
print("👂 Standing by... Say 'Hey Jarvis'")

_score_window   = deque(maxlen=SCORE_WINDOW)
_cooldown_count = 0

while True:
    mic_audio   = mic_queue.get()
    audio_16k   = mic_audio[::DOWNSAMPLE]
    audio_int16 = (audio_16k.flatten() * 32768).astype(np.int16)

    # ── Post-song listen window ───────────────────────────────────────────
    with _post_song_listen_lock:
        listen_open = _post_song_listen

    if listen_open:
        rms = float(np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))) / 32768
        if rms > 0.008:
            with _post_song_listen_lock:
                _post_song_listen = False
            log.info("Post-song reply detected — launching record_and_process")
            _score_window.clear()
            _cooldown_count = COOLDOWN_CHUNKS
            threading.Thread(
                target=record_and_process,
                kwargs={"window_sum": 0.0, "peak_score": 0.0},
                daemon=True,
            ).start()
        continue

    # ── Normal wake-word accumulator ──────────────────────────────────────
    if _cooldown_count > 0:
        _cooldown_count -= 1
        continue

    prediction = oww_model.predict(audio_int16)
    score      = prediction[WAKE_WORD_KEY]
    print(f"  Jarvis: {score:.4f}", end='\r', flush=True)

    _score_window.append(score if score >= WAKE_THRESHOLD else 0.0)
    window_sum = sum(_score_window)

    if score >= WAKE_THRESHOLD:
        log.debug(f"Wake score: {score:.4f}  window_sum={window_sum:.4f}")

    if window_sum >= SCORE_FIRE_THRESH:
        peak = max(_score_window)
        _score_window.clear()
        _cooldown_count = COOLDOWN_CHUNKS
        log.info(f"Wake word fired — window_sum={window_sum:.3f} peak={peak:.3f}")
        threading.Thread(
            target=record_and_process,
            kwargs={"window_sum": window_sum, "peak_score": peak},
            daemon=True,
        ).start()
