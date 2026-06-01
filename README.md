# T.A.R.A. (Thermodynamic Audio Retrieval Assistant)

T.A.R.A. is a distributed, offline-first voice assistant and autonomous music selection engine. Instead of relying on traditional collaborative filtering or cloud APIs, T.A.R.A. treats music selection as a thermodynamic process, modeling track progression on microstructural grain growth in a high-dimensional embedding space. 

Designed for edge efficiency and computational depth, the system is split across two nodes: a lightweight Raspberry Pi client for zero-latency wake-word and acoustic processing, and a high-performance Mac server that handles deep ML inference, semantic parsing, and the physics-based selection engine.

---

## 🧠 The Thermodynamic Selection Engine (Grain Physics)

The defining feature of T.A.R.A. is its autonomous playback queue, which abandons standard heuristic shuffles in favor of a microstructural grain-growth model (inspired by Vedanti et al. 2020). 

All computation operates via pure matrix algebra on a pre-built 512-dimensional CLAP unit hypersphere. Unplayed songs represent a disordered "matrix," while played songs form crystallized domains. The goal of the algorithm is to naturally evolve a "vibe" (grain) by balancing local expansion with global convergence.

### The Mathematical Framework
For every unplayed candidate song vector $v$, the engine computes a selection score based on a Boltzmann-like distribution:

$$Score(v) = \delta S^g(v) \cdot \exp\left(-\frac{\Delta S_m(v)}{T}\right) \cdot W_{profile}(v)$$

* **Grain Centroid ($\mu$) & Radius ($r$):** The current vibe is defined by a normalized centroid vector $\mu$ and a boundary radius $r$ (cosine similarity floor).
* **Local Grain Entropy ($\delta S^g$):** To prevent the system from getting stuck on identical songs, the engine prioritizes "frontier" songs. We calculate $\delta S^g(v) = \max(0, d(v) - \bar{d})$, where $d(v)$ is the candidate's distance to the centroid, and $\bar{d}$ is the mean interior distance. A positive value means the candidate expands the grain.
* **Global Microstructural Entropy ($\Delta S_m$):** Adding a song that is wildly different from the played set increases the global spread of the session. The exponential term sharply penalizes high-$\Delta S_m$ candidates, forcing the session to coarsen into a coherent domain.
* **Time-of-Day Annealing ($T$):** The temperature $T$ governs the strictness of the grain boundary. Modulated by the time of day, a high $T$ (mornings) dampens the $\Delta S_m$ penalty, allowing wide exploration. A low $T$ (late nights) enforces strict boundary adherence for a locked-in vibe.

### Dynamic Recrystallization (The "Skip" Mechanic)
When a user issues a `skip` command, the system models it as a thermal spike and dynamic recrystallization. The current grain shatters, releasing boundary energy ($S_m$ resets to $1.0$). The engine then nucleates a new grain seed in the unplayed matrix—specifically searching for a vector that maximizes distance from the shattered centroid while minimizing energy cost relative to the unplayed centroid.

---

## 🏗️ System Architecture & Edge Processing

To ensure privacy and optimize I/O on the edge device, the client-server architecture enforces strict memory-only audio handling on the Pi.

### 1. The Client (Raspberry Pi)
The Pi handles all immediate environment interactions without relying on persistent local storage (aside from daily text logs).
* **Ring-Buffer Wake Word Capture:** The system runs a continuous `openwakeword` loop. When "Hey Jarvis" is detected, the preceding 2-second audio ring buffer is captured in-memory and shipped to the Mac via HTTP for auto-labeling and true/false positive storage. No WAV files touch the Pi's SD card.
* **In-Memory Command Saving:** Following the wake word, VAD (Voice Activity Detection) records the user's command until silence is detected. The raw PCM data is converted to WAV bytes in RAM and sent to the server.
* **Zero-Latency Hardware Controls (Vosk Fast-Path):** To eliminate server-side latency for basic commands, the Pi runs a lightweight Vosk Kaldi recognizer. If Vosk detects a fast-path keyword (`stop`, `pause`, `resume`, or `skip`), it instantly halts or modifies the `cvlc` audio process locally and asynchronously updates the Mac's state. 

### 2. The Brain (Mac Server)
The server acts as the heavy-inference hub. It manages the SQLite/JSON databases, runs the local LLM, and synthesizes localized spoken intros using Piper TTS.

* **Semantic Slow-Path (Whisper):** If the Vosk fast-path does not trigger, the Mac runs faster-whisper (`large-v3`) for semantic transcription to understand complex requests (e.g., "Play something dark and atmospheric").
* **General Voice Assistant Capabilities (Gemma2:9b):** Beyond audio retrieval, T.A.R.A. functions as a fully capable, general-purpose voice assistant. If a transcribed query falls outside of music selection or system controls, it is dynamically routed to the LLM. Users can ask for general knowledge facts, request a short bedtime story for kids, or hold standard conversational queries—all processed locally and read back via TTS.
* **Time & Timer Management:** The server includes dedicated, fast-text endpoints for utility functions. Users can ask for the current time or set asynchronous countdown timers. When a timer fires, T.A.R.A. ducks the current audio volume, announces the alert over the music, and smoothly restores the original volume.

---

## 📚 The Data Ingestion Pipeline (`build_library.py`)

Before playback can occur, raw audio files must be analyzed, parameterized, and embedded into the thermodynamic hypersphere. The `build_library.py` script is a 4-layer analytical pipeline:

1. **Acoustic Feature Extraction (Librosa):** Analyzes the first 120 seconds of a track to extract rhythmic stability (onset variance, tempo), dynamic range (RMS variance), harmonic-to-percussive ratios, and spectral contrast. 
2. **Vocal Transcription (faster-whisper):** Scans the audio for lyrics. It calculates a `no_speech_prob` average to cleanly separate vocal tracks from instrumentals.
3. **CLAP Audio Embedding:** Instead of relying on just the intro, the script extracts 10-second clips from the start, middle, and end of the track. These clips are passed through `laion/clap-htsat-unfused`, and their embeddings are averaged into a single 512-dimensional vector, giving the physics engine a complete representation of the song's arc.
4. **LLM Classification (Gemma2:9b):** The acoustic features and transcribed lyrics are fed into a local LLM prompt. The model outputs a strict JSON payload quantifying the track's psychological affect (arousal, valence, tension, drive) and prose descriptions of its narrative and thematic setting.

---

## 🚀 Setup and Execution

**Requirements:**
* **Server:** macOS/Linux, 16GB+ RAM (Apple Silicon recommended for LLM/Whisper inference).
* **Client:** Raspberry Pi 4/5, USB Microphone, Bluetooth Speaker.
* **Dependencies:** `librosa`, `faster-whisper`, `openwakeword`, `transformers`, `torch`, `sounddevice`, `ollama`, `vosk`.

**Usage:**
1. Run `python3 build_library.py` on the server to ingest your local music folder.
2. Start the Mac brain: `uvicorn tara_server:app --host 0.0.0.0 --port 8000`
3. Launch the Pi client: `python3 tara_system.py`
