"""
tara_server.py — Mac-side TARA brain.
Changes in this version:
  - USB local playback: songs live on a USB drive connected to the Pi.
    The Mac no longer streams audio — it only sends metadata (filepath,
    title, artist, spoken_intro).  The Pi plays directly from USB.
  - PI_MUSIC_ROOT: new constant mapping Mac music root → Pi USB mount point.
    All endpoints that return a filepath call mac_to_pi_path() to translate.
  - Removed: /stream_song, /song_info endpoints, STREAM_CHUNK_BYTES.
  - Grain physics engine (GrainState): autonomous song selection modelled on
    microstructural grain growth (Berdichevsky, Sci. Rep. 2020).
    Mathematical framework:
      · The 512-dim CLAP unit hypersphere is the embedding space (material).
      · Unplayed songs are the disordered matrix; played songs are crystallised.
      · A grain = one vibe cluster, represented by centroid μ and radius r.
      · Microstructural entropy Sₘ (global) = spread of ALL played songs;
        decays each selection as the session converges on a coherent region.
      · Grain entropy Sᵍ (local) = spread of songs within the current grain;
        must INCREASE each pick — we choose frontier songs, not the centroid.
      · Annealing temperature T = time-of-day × user-profile factor;
        controls grain boundary width (how strictly "in-grain" is defined).
      · Selection rule: pick the candidate that simultaneously decreases Sₘ
        and increases Sᵍ — the grain boundary song.
      · Skip = recrystallization: Sₘ resets high, new grain nucleates far
        from shattered centroid in low-energy (unplayed) matrix territory.
    All computation is pure NumPy matrix algebra on a pre-built embeddings
    matrix cached at startup — typical execution <50ms.
  - /report_playback: accepts Pi-side filepath; remaps to Mac path for DB.
  - All other functionality preserved (CLAP, Whisper, Piper, Ollama,
    profile manager, timers, health, corrections, TTS endpoint, etc.).
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import wave
import uuid
import re
import json
import random
import shutil
from collections import deque
from datetime import datetime
import threading
import time
import numpy as np
from faster_whisper import WhisperModel
import ollama
from piper import PiperVoice

app = FastAPI()

# ── Paths ─────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USB DRIVE — set PI_MUSIC_ROOT to the mount point of the USB drive on  │
# │  the Pi (must match USB_MUSIC_PATH in tara_system.py).                 │
# │  MUSIC_FOLDER is the Mac-side source; it stays as the canonical DB key.│
# │  mac_to_pi_path() translates before sending any filepath to the Pi.    │
# └─────────────────────────────────────────────────────────────────────────┘
MUSIC_FOLDER   = "/Users/vnpawan/PandoraMusic"   # Mac-side; DB canonical key
PI_MUSIC_ROOT  = "/media/vnpawan/MUSICBOX"        # ← change to match USB_MUSIC_PATH on Pi

DB_PATH               = os.path.join(MUSIC_FOLDER, "music_library.json")
WAKE_WORD_LOGS_FOLDER = "/Users/vnpawan/Documents/TARA_WakeWordLogs_heyjarvis"
PROFILE_PATH          = "/Users/vnpawan/Documents/TARA_UserProfile/user_profile.json"
INTERACTION_STORE_PATH= "/Users/vnpawan/Documents/TARA_Interactions"

# Wake clip labeled sub-folders (populated by /log_wake_clip auto-labeler)
WAKE_UNLABELED_DIR       = os.path.join(WAKE_WORD_LOGS_FOLDER, "unlabeled")
WAKE_LABEL_TP_DIR        = os.path.join(WAKE_WORD_LOGS_FOLDER, "labeled", "true_positive")
WAKE_LABEL_FP_DIR        = os.path.join(WAKE_WORD_LOGS_FOLDER, "labeled", "false_positive")
WAKE_LABEL_UNCERTAIN_DIR = os.path.join(WAKE_WORD_LOGS_FOLDER, "labeled", "uncertain")

for d in [
    WAKE_WORD_LOGS_FOLDER, INTERACTION_STORE_PATH,
    os.path.dirname(PROFILE_PATH),
    WAKE_UNLABELED_DIR, WAKE_LABEL_TP_DIR,
    WAKE_LABEL_FP_DIR, WAKE_LABEL_UNCERTAIN_DIR,
]:
    os.makedirs(d, exist_ok=True)

CLAP_MODEL_ID   = "laion/clap-htsat-unfused"
CLAP_CANDIDATES = 3
FUZZY_THRESHOLD = 88

# ── Initialization ────────────────────────────────────────────────────────────

print("🧠 Booting TARA Server...")
whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
piper_voice   = PiperVoice.load("en_US-lessac-medium.onnx")

print("🎵 Loading CLAP model...")
try:
    import torch
    from transformers import ClapModel, ClapProcessor
    clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    clap_model     = ClapModel.from_pretrained(CLAP_MODEL_ID)
    clap_model.eval()
    CLAP_AVAILABLE = True
    print("✅ CLAP loaded.")
except Exception as e:
    CLAP_AVAILABLE = False
    clap_model = clap_processor = None
    print(f"⚠️  CLAP failed ({e}). Falling back to keyword matching.")

try:
    from rapidfuzz import fuzz, process as fuzz_process
    RAPIDFUZZ_AVAILABLE = True
    print("✅ rapidfuzz loaded.")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️  rapidfuzz not installed.")

print("✅ TARA Brain is Online.")

# ── Session state ─────────────────────────────────────────────────────────────

last_played_context: dict | None = None
timer_alert_queue:   deque       = deque()
alert_queue_lock                 = threading.Lock()

# Session-scoped played set — grows until the user says "stop".
# All filepaths (Mac-side) played or skipped since the last stop are stored here
# so they are never repeated within a session.
session_played_fps:    set = set()
session_played_lock        = threading.Lock()
session_songs_played_count = 0   # incremented in record_play; used for streak-aware intros

# Minimum cosine similarity below which two songs are considered "different enough"
# to both appear in a session (prevents near-duplicate picks even for songs not
# in session_played_fps yet).
SESSION_SIMILARITY_THRESHOLD = 0.97

# ── Surprise-me keywords ──────────────────────────────────────────────────────
# When the user says any of these, TARA deliberately picks from the LOW-scoring
# end of the grain candidates — the opposite of what it normally does.
SURPRISE_KEYWORDS = [
    "surprise me", "surprise", "you pick", "you choose", "your choice",
    "anything", "whatever", "random", "i don't know", "i don't care",
    "you decide", "just pick", "pick for me", "something random",
    "no preference", "dealer's choice",
]
# ══════════════════════════════════════════════════════════════════════════════
# GRAIN PHYSICS ENGINE  (v2 — faithful grain-growth model)
# ══════════════════════════════════════════════════════════════════════════════
#
# Conceptual mapping (Berdichevsky, Sci. Rep. 2020):
#
#   Embedding space   →  512-dim CLAP unit hypersphere
#   Matrix            →  unplayed songs (disordered, not yet crystallised)
#   Grain             →  current vibe cluster: centroid μ, radius r
#   Grain growth      →  each completion pulls μ forward and widens r slightly;
#                         Sₘ (global) falls because the played set becomes less
#                         spread; Sᵍ (local) rises because we pick frontier songs
#   Recrystallization →  skip shatters grain; new grain nucleates in the matrix
#                         (unplayed region) farthest from the shattered centroid
#   Annealing T       →  time-of-day × profile factor; controls grain boundary
#                         width (how strictly "frontier" is enforced)
#
# Selection rule (enforced in score_candidates):
#   For each unplayed candidate v, compute:
#     d(v)   = μ · v                     (cosine proximity to grain centre)
#     δSᵍ(v) = d(v) − d̄                 (how much v expands grain spread)
#              where d̄ = mean cosine of all songs already in this grain
#     in_grain(v) = d(v) ≥ r            (grain boundary filter)
#
#   Score(v) = δSᵍ(v) · exp(−ΔSₘ(v) / T) · profile_weight(v)
#
#     δSᵍ(v) > 0  → v is farther from μ than the current mean → Sᵍ↑ ✓
#     ΔSₘ(v)      → how much adding v would increase global spread (penalised)
#                   approximated as the distance of v from the played centroid
#     T            → annealing temperature; high T → loose boundary, more
#                   exploration; low T → tight boundary, committed vibe
#
# ── Tunable constants ─────────────────────────────────────────────────────────

GRAIN_ETA         = 0.10   # centroid EMA rate (weight of new song on μ update)
GRAIN_R_INIT      = 0.50   # initial grain radius (cosine similarity floor)
GRAIN_R_GROWTH    = 0.02   # how much r expands per completion (grain coarsens)
GRAIN_R_MAX       = 0.90   # grain radius ceiling (never consumes whole space)
GRAIN_T_BASE      = 0.10   # residual annealing temperature (never fully frozen)
GRAIN_GAMMA       = 0.70   # temperature dissipation rate per completion
GRAIN_DT          = 2.00   # thermal spike magnitude on skip
GRAIN_SM_FLOOR    = 0.05   # global entropy floor (session never fully collapses)
GRAIN_K_PRESCREEN = 60     # pre-screen size before full frontier scoring
GRAIN_TOD_WEIGHTS = {      # time-of-day annealing multiplier on T
    "morning":   1.40,     # more exploration — fresh start
    "afternoon": 1.00,     # neutral
    "evening":   0.70,     # settling into a vibe
    "night":     0.40,     # committed, tight grain
}


class GrainState:
    """
    Holds the full thermodynamic state of the music session.
    Thread-safe via a single lock.

    Key state variables:
      μ    — unit-vector centroid of the current grain in CLAP space
      r    — grain radius (cosine similarity threshold defining the grain)
      d̄    — mean cosine of all in-grain songs to μ (tracks grain interior)
      Sₘ   — microstructural entropy: spread of ALL played songs (global)
      T    — annealing temperature (modulated by time-of-day)
      n    — consecutive completions in current grain (resets on skip)
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self.mu           = None          # (512,) unit vector — grain centroid
        self.r            = GRAIN_R_INIT  # grain radius (cosine floor)
        self.d_bar        = 0.0           # mean cosine of in-grain songs to μ
        self.n            = 0             # consecutive completions in grain
        self.T            = GRAIN_T_BASE  # annealing temperature
        self.Sm           = 1.0           # global microstructural entropy
        self.played_embs  = []            # list of (512,) unit vecs — all played songs
        # Pre-built full embeddings matrix for fast scoring
        self._emb_matrix  = None          # (N_songs, 512)
        self._emb_fps     = []
        self._emb_dirty   = True

    # ── Embeddings matrix cache ───────────────────────────────────────────────

    def mark_dirty(self):
        with self._lock:
            self._emb_dirty = True

    def _rebuild_matrix(self, songs: list):
        fps, vecs = [], []
        for s in songs:
            emb = s.get("clap_embedding")
            if emb and len(emb) == 512:
                v = np.array(emb, dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    fps.append(s["filepath"])
                    vecs.append(v / n)
        if vecs:
            self._emb_matrix = np.stack(vecs, axis=0)
            self._emb_fps    = fps
        else:
            self._emb_matrix = None
            self._emb_fps    = []
        self._emb_dirty = False
        print(f"[GRAIN] Embeddings matrix built: {len(fps)} songs")

    # ── Global entropy Sₘ ─────────────────────────────────────────────────────

    def _compute_Sm(self) -> float:
        """
        Sₘ = average pairwise cosine distance among all played songs.

        On the unit hypersphere, cosine distance = 1 − (vᵢ · vⱼ).
        A session of diverse songs has Sₘ → 1; a session locked in one vibe
        has Sₘ → 0.

        For efficiency we compute:
          Sₘ = 1 − (1/N²) · ‖Σvᵢ‖²  +  1/N   (bias correction for self-pairs)
        which is exact and O(N·D) instead of O(N²·D).
        """
        if len(self.played_embs) < 2:
            return 1.0
        V   = np.stack(self.played_embs, axis=0)   # (N, 512)
        N   = V.shape[0]
        s   = V.sum(axis=0)                         # (512,)  Σvᵢ
        Sm  = 1.0 - (float(np.dot(s, s)) / (N * N)) + 1.0 / N
        return float(np.clip(Sm, GRAIN_SM_FLOOR, 1.0))

    # ── Core scoring — the grain-growth selection rule ────────────────────────

    def score_candidates(
            self,
            songs: list,
            profile_weights: dict,
            exclude_fps: set | None = None,
    ) -> list[tuple[float, dict]]:
        """
        Score unplayed candidates by the grain-growth selection rule:

          Score(v) = δSᵍ(v) · exp(−ΔSₘ(v) / T) · profile_weight(v)

        where:
          δSᵍ(v) = max(0, d(v) − d̄)
            d(v)  = μ · v  (cosine of candidate to grain centroid)
            d̄     = mean cosine of in-grain played songs to μ
            Positive δSᵍ means v sits farther from μ than the average played
            song — it lies on the frontier and widens the grain (Sᵍ↑).

          ΔSₘ(v)  = 1 − (μ_played · v)
            μ_played = normalised mean of ALL played embeddings (played centroid)
            This approximates how much adding v increases global spread.
            Songs close to the played centroid have small ΔSₘ (Sₘ↓ / stable).

          T       = annealing temperature (time-of-day modulated).
            High T → exp term ≈ 1 → ΔSₘ penalty is weak → wider exploration.
            Low T  → exp term penalises high-ΔSₘ candidates sharply → tight vibe.

          Grain boundary filter (in_grain):
            Only candidates with d(v) ≥ r are eligible. r is the grain radius.
            Songs inside the grain (d > d̄) are the frontier; songs outside (d < r)
            are a different grain or matrix territory.
        """
        if exclude_fps is None:
            exclude_fps = set()

        with self._lock:
            if self._emb_dirty:
                self._rebuild_matrix(songs)

            fp_to_song = {s["filepath"]: s for s in songs}

            if self._emb_matrix is None or len(self._emb_fps) == 0:
                eligible = [s for s in songs if s.get("filepath") not in exclude_fps]
                return sorted(
                    [(profile_weights.get(s.get("filepath", ""), 1.0), s)
                     for s in eligible],
                    key=lambda x: -x[0],
                )

            # ── 0. Mask out played / excluded songs (the matrix = unplayed) ──
            mask = np.array(
                [fp not in exclude_fps for fp in self._emb_fps], dtype=bool
            )
            if mask.sum() == 0:
                mask[:] = True   # full library played — prevent crash

            emb_sub = self._emb_matrix[mask]           # (N_unplayed, 512)
            fps_sub = [fp for fp, m in zip(self._emb_fps, mask) if m]

            # ── 1. Cold start: initialise grain centroid from first pick ──────
            if self.mu is None:
                mu_raw = emb_sub.mean(axis=0)
                norm   = np.linalg.norm(mu_raw)
                self.mu = mu_raw / norm if norm > 0 else emb_sub[0].copy()
                self.d_bar = float((emb_sub @ self.mu).mean()) if len(emb_sub) else 0.0
                print(f"[GRAIN] Cold-start: μ from {len(emb_sub)} unplayed songs")

            mu = self.mu   # (512,)

            # ── 2. Pre-screen: top-K candidates by cosine to μ ───────────────
            #    (brings candidates inside / near the grain boundary)
            dots_all = emb_sub @ mu                       # (N_unplayed,)
            K        = min(GRAIN_K_PRESCREEN, len(dots_all))
            top_idx  = np.argpartition(dots_all, -K)[-K:]
            top_idx  = top_idx[np.argsort(-dots_all[top_idx])]

            cand_emb = emb_sub[top_idx]                   # (K, 512)
            cand_fps = [fps_sub[i] for i in top_idx]
            d_cand   = dots_all[top_idx]                  # (K,)  cosine to μ

            # ── 3. Grain boundary filter: keep candidates with d(v) ≥ r ──────
            #    r is the minimum cosine similarity to be "in this grain"
            in_grain_mask = d_cand >= self.r
            if in_grain_mask.sum() == 0:
                # No candidates inside grain — relax r by half and retry
                in_grain_mask = d_cand >= (self.r * 0.5)
            if in_grain_mask.sum() == 0:
                in_grain_mask[:] = True   # total fallback

            cand_emb = cand_emb[in_grain_mask]
            cand_fps = [fp for fp, m in zip(cand_fps, in_grain_mask) if m]
            d_cand   = d_cand[in_grain_mask]

            # ── 4. δSᵍ(v) = max(0, d(v) − d̄)  ──────────────────────────────
            #    Positive when candidate is farther from μ than the current
            #    interior mean → it sits on the frontier → grain entropy rises.
            d_bar    = self.d_bar
            dSg      = np.maximum(0.0, d_cand - d_bar)   # (K_filtered,)

            # ── 5. ΔSₘ(v) = 1 − (μ_played · v) ─────────────────────────────
            #    μ_played is the normalised centroid of all played songs.
            #    Small ΔSₘ → v is close to the played centroid → global spread
            #    does not increase → microstructural entropy drops (good).
            if self.played_embs:
                played_stack  = np.stack(self.played_embs, axis=0)  # (P, 512)
                mu_played_raw = played_stack.mean(axis=0)            # (512,)
                norm_p        = np.linalg.norm(mu_played_raw)
                mu_played     = mu_played_raw / norm_p if norm_p > 0 else mu
            else:
                mu_played = mu

            dSm = 1.0 - (cand_emb @ mu_played)           # (K_filtered,)  ≥ 0

            # ── 6. Annealing: exp(−ΔSₘ / T) ─────────────────────────────────
            #    T controls grain boundary sharpness.
            #    High T  → Boltzmann factor ≈ 1 for all candidates (loose).
            #    Low T   → only small-ΔSₘ candidates survive (tight).
            T       = max(self.T, 1e-6)
            boltz   = np.exp(-dSm / T)                    # (K_filtered,)

            # ── 7. Profile weights ────────────────────────────────────────────
            prof_wts = np.array(
                [profile_weights.get(fp, 1.0) for fp in cand_fps],
                dtype=np.float32,
            )

            # ── 8. Final score ────────────────────────────────────────────────
            #    Score(v) = δSᵍ(v) · exp(−ΔSₘ(v)/T) · profile_weight(v)
            final = dSg * boltz * prof_wts                # (K_filtered,)

            result = sorted(
                [(float(final[i]), fp_to_song[cand_fps[i]])
                 for i in range(len(cand_fps)) if cand_fps[i] in fp_to_song],
                key=lambda x: -x[0],
            )
            return result

    # ── State update: completion ───────────────────────────────────────────────

    def update_completion(self, embedding: np.ndarray | None):
        """
        Called after a song ends naturally.  Grain grows:

          μ  ← normalise((1−η)·μ + η·v)
               Centroid drifts toward the completed song (EMA on hypersphere).

          r  ← min(r + Δr, r_max)
               Grain radius expands: the grain is coarser after each song.

          d̄  ← (d̄·n + μ·v) / (n+1)
               Running mean cosine of in-grain songs; updated incrementally.

          Sₘ ← recomputed from all played embeddings (global entropy drops
               as played songs cluster into a tighter region).

          T  ← T · exp(−γ)
               Temperature dissipates: annealing progresses.

          n  ← n + 1
        """
        with self._lock:
            self.n += 1
            self.T  = self.T * np.exp(-GRAIN_GAMMA)

            if embedding is not None:
                v    = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                    self.played_embs.append(v.copy())

                    if self.mu is not None:
                        # Update d̄ incrementally before updating μ
                        d_v        = float(np.dot(self.mu, v))
                        self.d_bar = (self.d_bar * (self.n - 1) + d_v) / self.n

                        # Update μ via EMA on the hypersphere
                        mu_raw    = (1.0 - GRAIN_ETA) * self.mu + GRAIN_ETA * v
                        norm_mu   = np.linalg.norm(mu_raw)
                        self.mu   = mu_raw / norm_mu if norm_mu > 0 else self.mu

                    # Grow grain radius
                    self.r = min(self.r + GRAIN_R_GROWTH, GRAIN_R_MAX)

            # Recompute global Sₘ
            self.Sm = self._compute_Sm()

            print(
                f"[GRAIN] Completion #{self.n}: Sₘ={self.Sm:.3f} "
                f"r={self.r:.3f} d̄={self.d_bar:.3f} T={self.T:.3f}"
            )

    # ── State update: skip / recrystallization ────────────────────────────────

    def update_skip(self, embedding: np.ndarray | None, songs: list, profile_weights: dict):
        """
        Skip = dynamic recrystallization.  The current grain shatters:

          T  ← T · exp(−γ) + ΔT     (thermal spike — system is disturbed)
          Sₘ ← 1.0                   (full disorder — grain boundary energy released)
          n  ← 0                     (streak resets)
          r  ← GRAIN_R_INIT          (grain radius collapses back to seed size)
          d̄  ← 0.0                   (interior mean resets)

        Nucleation of new grain in the matrix (unplayed territory):
          score_c = (1 − μ_shattered · v_c)    (far from shattered grain)
                  · (1 − ΔSₘ(v_c))             (close to unplayed centroid → low energy)
          New μ = argmax score_c over all unplayed embeddings.

        The skipped song embedding is added to played_embs so it is never
        revisited, but the shattered grain's region is abandoned.
        """
        with self._lock:
            self.T   = self.T * np.exp(-GRAIN_GAMMA) + GRAIN_DT
            self.Sm  = 1.0
            self.n   = 0
            self.r   = GRAIN_R_INIT
            self.d_bar = 0.0

            shattered_mu = self.mu.copy() if self.mu is not None else None

            if embedding is not None:
                v    = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                    self.played_embs.append(v.copy())

            # Nucleation: find new grain seed in unplayed matrix
            if self._emb_matrix is not None and shattered_mu is not None:
                played_set = set()
                # Build unplayed mask
                unplayed_mask = np.ones(len(self._emb_fps), dtype=bool)
                if self.played_embs:
                    played_stack  = np.stack(self.played_embs, axis=0)   # (P, 512)
                    # Mark any song with cosine > 0.99 to a played embedding as played
                    sims_to_played = self._emb_matrix @ played_stack.T   # (N, P)
                    unplayed_mask  = sims_to_played.max(axis=1) < 0.99

                if unplayed_mask.sum() == 0:
                    unplayed_mask[:] = True   # all played — use everything

                unplayed_embs = self._emb_matrix[unplayed_mask]   # (U, 512)
                unplayed_fps  = [fp for fp, m in zip(self._emb_fps, unplayed_mask) if m]

                # Distance from shattered grain (farther = better seed)
                dist_from_shattered = 1.0 - (unplayed_embs @ shattered_mu)   # (U,)

                # Proximity to unplayed centroid (in-matrix = low Sₘ cost)
                mu_unplayed_raw = unplayed_embs.mean(axis=0)
                norm_u          = np.linalg.norm(mu_unplayed_raw)
                mu_unplayed     = mu_unplayed_raw / norm_u if norm_u > 0 else unplayed_embs[0]
                prox_to_matrix  = unplayed_embs @ mu_unplayed                 # (U,)

                nucl_scores = dist_from_shattered * prox_to_matrix            # (U,)
                best_idx    = int(np.argmax(nucl_scores))
                self.mu     = unplayed_embs[best_idx].copy()
                print(f"[GRAIN] Recrystallization → new μ at '{unplayed_fps[best_idx]}'")
            else:
                self.mu = None

            print(
                f"[GRAIN] Skip: Sₘ={self.Sm:.3f} T={self.T:.3f} r={self.r:.3f}"
            )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "n":            self.n,
                "Sm":           round(self.Sm, 4),
                "T":            round(self.T, 4),
                "r":            round(self.r, 4),
                "d_bar":        round(self.d_bar, 4),
                "n_played":     len(self.played_embs),
                "has_centroid": self.mu is not None,
            }


# Module-level singleton — persists for the lifetime of the server process
grain = GrainState()


def reset_session():
    """
    Clear all session state.  Called when the user says 'stop'.
    Resets both the session played set and the grain physics engine so the
    next session starts completely fresh.
    """
    global last_played_context, session_songs_played_count
    with session_played_lock:
        session_played_fps.clear()
    session_songs_played_count = 0
    last_played_context = None
    with grain._lock:
        grain.mu          = None
        grain.n           = 0
        grain.T           = GRAIN_T_BASE
        grain.Sm          = 1.0
        grain.r           = GRAIN_R_INIT
        grain.d_bar       = 0.0
        grain.played_embs = []
        grain._emb_dirty  = True
    print("[SESSION] 🔄 Session reset — played set cleared, grain reset.")

# ══════════════════════════════════════════════════════════════════════════════
# PROFILE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

profile_lock = threading.Lock()


def _load_profile() -> dict:
    print(f"[PROFILE] Loading profile from: {PROFILE_PATH}")
    if not os.path.exists(PROFILE_PATH):
        print(f"[PROFILE] ⚠️  Profile file not found — returning default profile.")
        return _default_profile()
    try:
        with open(PROFILE_PATH) as f:
            profile = json.load(f)
        print(f"[PROFILE] ✅ Profile loaded. Keys: {list(profile.keys())}")
        return profile
    except Exception as e:
        print(f"[PROFILE] ❌ Failed to load profile: {e}. Returning default.")
        return _default_profile()


def _save_profile(profile: dict):
    tmp = PROFILE_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(profile, f, indent=2)
        os.replace(tmp, PROFILE_PATH)
        print(f"[PROFILE] ✅ Profile saved successfully.")
    except Exception as e:
        print(f"[PROFILE] ❌ Failed to save profile: {e}")
        raise


def _default_profile() -> dict:
    return {
        "affect_weights": {
            "arousal": 0.5, "valence": 0.5, "tension": 0.5, "drive": 0.5,
        },
        "time_of_day": {
            "morning":   {"arousal": 0.5, "valence": 0.5, "tension": 0.5, "drive": 0.5},
            "afternoon": {"arousal": 0.5, "valence": 0.5, "tension": 0.5, "drive": 0.5},
            "evening":   {"arousal": 0.5, "valence": 0.5, "tension": 0.5, "drive": 0.5},
            "night":     {"arousal": 0.5, "valence": 0.5, "tension": 0.5, "drive": 0.5},
        },
        "genre_counts":      {},
        "song_skips":        {},
        "song_completions":  {},
        "corrections":       [],
        "user_context": (
            "The user is Indian and listens to a mix of "
            "Hindi film music and English songs."
        ),
        "total_interactions": 0,
        "last_updated": "",
    }


def _get_time_bucket() -> str:
    h = datetime.now().hour
    if   6 <= h < 12: return "morning"
    elif 12 <= h < 18: return "afternoon"
    elif 18 <= h < 22: return "evening"
    else:              return "night"


def _decay_weight(old: float, new_signal: float, alpha: float = 0.15) -> float:
    return round(old * (1 - alpha) + new_signal * alpha, 4)


def profile_record_play(filepath: str, elapsed: float, completed: bool, db: dict):
    print(f"[PLAYBACK] Recording play: filepath='{filepath}', elapsed={elapsed:.1f}s, completed={completed}")
    with profile_lock:
        profile   = _load_profile()
        song_data = db.get("songs", {}).get(filepath, {})
        c         = song_data.get("classification", {})
        affect    = c.get("affect", {})
        genres    = c.get("genres", [])

        if completed:          signal = 1.0
        elif elapsed < 30:     signal = 0.0
        elif elapsed < 60:     signal = 0.2
        else:                  signal = 0.4

        for dim in ["arousal", "valence", "tension", "drive"]:
            sv = affect.get(dim)
            if sv is not None:
                try:
                    sv = float(sv)
                    profile["affect_weights"][dim] = _decay_weight(
                        profile["affect_weights"][dim],
                        sv if signal >= 0.5 else 1.0 - sv,
                    )
                except (ValueError, TypeError):
                    pass

        bucket = _get_time_bucket()
        for dim in ["arousal", "valence", "tension", "drive"]:
            sv = affect.get(dim)
            if sv is not None:
                try:
                    sv = float(sv)
                    profile["time_of_day"][bucket][dim] = _decay_weight(
                        profile["time_of_day"][bucket][dim],
                        sv if signal >= 0.5 else 1.0 - sv,
                        alpha=0.12,
                    )
                except (ValueError, TypeError):
                    pass

        for genre in genres:
            profile["genre_counts"].setdefault(genre, {"plays": 0, "skips": 0})
            if completed or elapsed >= 60:
                profile["genre_counts"][genre]["plays"] += 1
            else:
                profile["genre_counts"][genre]["skips"] += 1

        if completed:
            profile["song_completions"][filepath] = \
                profile["song_completions"].get(filepath, 0) + 1
        elif elapsed < 30:
            profile["song_skips"][filepath] = \
                profile["song_skips"].get(filepath, 0) + 1

        profile["last_updated"] = datetime.now().isoformat()
        _save_profile(profile)


def profile_record_correction(heard: str, intended: str):
    with profile_lock:
        profile = _load_profile()
        matched = False
        for entry in profile["corrections"]:
            if entry["heard"].lower() == heard.lower():
                entry["intended"] = intended
                entry["count"]    = entry.get("count", 1) + 1
                matched           = True
                break
        if not matched:
            profile["corrections"].append(
                {"heard": heard, "intended": intended, "count": 1}
            )
        profile["last_updated"] = datetime.now().isoformat()
        _save_profile(profile)


def profile_get_candidate_weights(songs: list) -> dict:
    with profile_lock:
        profile = _load_profile()
    weights      = {}
    skips        = profile.get("song_skips", {})
    completions  = profile.get("song_completions", {})
    genre_counts = profile.get("genre_counts", {})
    for song in songs:
        fp = song.get("filepath", "")
        w  = 1.0
        sc = skips.get(fp, 0)
        cc = completions.get(fp, 0)
        if sc > 3:   w *= 0.3
        elif sc > 1: w *= 0.6
        if cc >= 3:  w  = min(w * 1.2, 1.5)
        for genre in song.get("classification", {}).get("genres", []):
            gc    = genre_counts.get(genre, {})
            total = gc.get("plays", 0) + gc.get("skips", 0)
            if total > 5:
                ratio = gc.get("plays", 0) / total
                if ratio > 0.7:   w = min(w * 1.15, 1.5)
                elif ratio < 0.3: w *= 0.7
        weights[fp] = round(w, 3)
    return weights


def profile_get_user_context() -> str:
    with profile_lock:
        profile = _load_profile()
    return profile.get("user_context", "")


def profile_get_correction_vocab() -> list[str]:
    with profile_lock:
        profile = _load_profile()
    return [
        e["intended"]
        for e in profile.get("corrections", [])
        if e.get("count", 0) >= 2
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace('*', '').replace('_', '').replace('#', '')
    return text.strip()


def synthesize_to_file(text: str, output_path: str):
    print(f"[TTS] Synthesizing: '{text[:100]}' → '{output_path}'")
    try:
        with wave.open(output_path, "wb") as wav_file:
            piper_voice.synthesize_wav(text, wav_file)
        size = os.path.getsize(output_path) if os.path.exists(output_path) else -1
        print(f"[TTS] ✅ WAV written. File size: {size} bytes")
    except Exception as e:
        print(f"[TTS] ❌ Synthesis failed for '{text[:60]}': {e}")
        raise


def cleanup_task(files: list):
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            print(f"[CLEANUP] Removed temp file: {f}")


def load_db() -> dict:
    print(f"[DB] Loading music library from: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"[DB] ⚠️  DB not found at {DB_PATH}. Returning empty library.")
        return {"songs": {}}
    try:
        with open(DB_PATH) as f:
            db = json.load(f)
        print(f"[DB] ✅ Loaded {len(db.get('songs', {}))} songs.")
        return db
    except Exception as e:
        print(f"[DB] ❌ Failed to load DB: {e}")
        return {"songs": {}}


# ── Path translation (Mac ↔ Pi USB) ──────────────────────────────────────────
# The DB uses Mac-side absolute paths as canonical keys.
# Before sending any filepath to the Pi, translate it to the Pi USB path.
# On /report_playback the Pi sends its USB path back; translate to Mac path
# for DB lookup.

def mac_to_pi_path(mac_path: str) -> str:
    """
    Translate a Mac-side music path to the equivalent Pi USB path.
    e.g. /Users/vnpawan/PandoraMusic/folder/song.mp3
      →  /media/pi/PandoraMusic/folder/song.mp3
    Returns the original path unchanged if it doesn't start with MUSIC_FOLDER
    (shouldn't happen in normal operation, but safe fallback).
    """
    mac_root = MUSIC_FOLDER.rstrip("/")
    pi_root  = PI_MUSIC_ROOT.rstrip("/")
    if mac_path.startswith(mac_root):
        return pi_root + mac_path[len(mac_root):]
    print(f"[PATH] ⚠️  mac_to_pi_path: '{mac_path}' does not start with '{mac_root}'")
    return mac_path


def pi_to_mac_path(pi_path: str) -> str:
    """
    Translate a Pi USB path back to the Mac-side path for DB lookup.
    e.g. /media/pi/PandoraMusic/folder/song.mp3
      →  /Users/vnpawan/PandoraMusic/folder/song.mp3
    """
    mac_root = MUSIC_FOLDER.rstrip("/")
    pi_root  = PI_MUSIC_ROOT.rstrip("/")
    if pi_path.startswith(pi_root):
        return mac_root + pi_path[len(pi_root):]
    print(f"[PATH] ⚠️  pi_to_mac_path: '{pi_path}' does not start with '{pi_root}'")
    return pi_path


def save_db(db: dict):
    try:
        with open(DB_PATH, 'w') as f:
            json.dump(db, f, indent=2)
        print(f"[DB] ✅ Library saved.")
    except Exception as e:
        print(f"[DB] ❌ Failed to save DB: {e}")
        raise


def record_play(db: dict, filepath: str):
    global last_played_context, session_songs_played_count
    with session_played_lock:
        session_played_fps.add(filepath)  # Track for the entire session
    session_songs_played_count += 1

    if filepath in db["songs"]:
        db["songs"][filepath].setdefault("play_history", [])
        db["songs"][filepath]["play_history"].append(datetime.now().isoformat())
        save_db(db)
        song = db["songs"][filepath]
        c = song.get("classification", {})
        last_played_context = {
            "title": c.get("title", os.path.basename(filepath)),
            "artist": c.get("artist", "unknown"),
            "filepath": filepath,
            "classification": c,
        }
        grain.mark_dirty()  # embeddings matrix may be stale after DB write
        print(f"   📝 Context updated: {last_played_context['title']} by {last_played_context['artist']}")
    else:
        print(f"[RECORD] ⚠️  Filepath not found in DB: '{filepath}'")


def last_played(song: dict) -> datetime | None:
    history = song.get("play_history", [])
    if not history:
        return None
    return datetime.fromisoformat(history[-1])


def build_whisper_prompt(db: dict) -> str:
    titles  = []
    artists = set()
    for song in db.get("songs", {}).values():
        c = song.get("classification", {})
        if c.get("title"):  titles.append(c["title"])
        if c.get("artist"): artists.add(c["artist"])
    random.shuffle(titles)
    parts = []
    if artists: parts.append("Artists: " + ", ".join(sorted(artists)))
    if titles:  parts.append("Songs: "   + ", ".join(titles))
    corrections = profile_get_correction_vocab()
    if corrections: parts.append("Also: " + ", ".join(corrections))
    prompt = ". ".join(parts)
    return prompt[:800] if len(prompt) > 800 else prompt


# ── Intent keywords ───────────────────────────────────────────────────────────

TIME_KEYWORDS     = ["what time", "current time", "what's the time",
                     "tell me the time", "time is it"]
TIMER_KEYWORDS    = ["set a timer", "set timer", "timer for", "remind me in",
                     "wake me in", "alert me in", "countdown", "set an alarm",
                     "alarm for"]
MUSIC_KEYWORDS    = ["play", "music", "song", "track", "put on",
                     "something to listen", "play me", "i want to hear",
                     "haven't heard", "not heard in a while", "play something",
                     "surprise me", "you pick", "you choose", "your choice",
                     "anything", "whatever", "you decide", "just pick",
                     "pick for me", "something random", "dealer's choice"]
UNHEARD_KEYWORDS  = ["haven't heard", "not heard", "haven't listened",
                     "long time", "while", "forget"]
STOP_KEYWORDS     = ["stop", "stop the music", "stop playing",
                     "turn it off", "kill it"]
PAUSE_KEYWORDS    = ["pause", "pause the music", "hold on",
                     "wait", "pause please"]
RESUME_KEYWORDS   = ["resume", "continue", "play again",
                     "unpause", "keep going", "carry on"]
SKIP_KEYWORDS     = ["skip", "next song", "next track", "play something else",
                     "another song", "different song", "change the song"]
CONTINUATION_KEYWORDS = [
    "similar", "like this", "like that", "more of", "same vibe", "same mood",
    "like the last", "another one like", "more upbeat", "slower",
    "something different", "completely different", "more energetic", "calmer",
    "more relaxing", "something heavier", "something lighter",
]
SOURCE_STRIP = [
    "play", "something", "from", "the", "movie", "film", "album", "by",
    "artist", "a", "song", "any", "some", "music", "track", "me",
    "please", "can you", "put on", "an",
]


# ── Song intro / follow-up helpers ───────────────────────────────────────────

import random


def is_surprise_request(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SURPRISE_KEYWORDS)


def build_spoken_intro(
    title:    str,
    artist:   str,
    reason:   str,
    *,
    affect:       dict | None = None,
    is_surprise:  bool        = False,
    grain_snap:   dict | None = None,
) -> str:
    """
    Context-aware spoken intro for TARA.

    Pulls in four layers of context so every intro feels personally chosen:
      1. Surprise flag  — wildcard flavour when the user said "surprise me"
      2. Time of day    — energy appropriate to morning / afternoon / evening / night
      3. Session depth  — opener acknowledges how deep into the session we are
      4. Grain entropy  — tighter language when deep in a vibe, looser when exploring
      5. Affect dims    — arousal/valence colour the description of the song itself
    """
    snap        = grain_snap or {}
    Sm          = snap.get("Sm",    0.5)
    r           = snap.get("r",     GRAIN_R_INIT)
    bucket      = _get_time_bucket()
    n_played    = session_songs_played_count   # how many songs so far this session

    # ── 1. Surprise pick ──────────────────────────────────────────────────────
    if is_surprise:
        surprise_openers = [
            "Alright, throwing the algorithm out the window.",
            "You asked for random — here's something from way out in left field.",
            "Okay, closing my eyes and pointing.",
            "Dealer's choice. Let's see how this lands.",
            "Ignoring everything I know about your taste for one song.",
        ]
        if artist:
            surprise_transitions = [
                f"This is {title} by {artist}. Might be exactly right, might be completely wrong.",
                f"Bringing you {artist} with {title}. No guarantees.",
                f"Let's try {title} from {artist} and see what happens.",
            ]
        else:
            surprise_transitions = [
                f"This is {title}. Might be exactly right, might be completely wrong.",
                f"Let's try {title} and see what happens.",
            ]
        return f"{random.choice(surprise_openers)} {random.choice(surprise_transitions)}"

    # ── 2. Time-of-day opener ─────────────────────────────────────────────────
    tod_openers = {
        "morning": [
            "Good morning energy, coming right up.",
            "Starting the day off properly.",
            "Morning mode: activated.",
        ],
        "afternoon": [
            "Keeping the afternoon rolling.",
            "Midday momentum, let's go.",
            "Perfect for the afternoon stretch.",
        ],
        "evening": [
            "Evening session, let's set the tone.",
            "Winding into the evening nicely.",
            "Good evening vibe incoming.",
        ],
        "night": [
            "Late night selection, dialled in.",
            "Night mode. Just you and the music.",
            "Something for the quiet hours.",
        ],
    }
    opener = random.choice(tod_openers.get(bucket, tod_openers["afternoon"]))

    # ── 3. Session-depth acknowledgement ──────────────────────────────────────
    if n_played == 0:
        depth_line = "Here's where we begin."
    elif n_played < 3:
        depth_line = "Just getting warmed up."
    elif n_played < 8:
        depth_line = f"Song number {n_played + 1}, and the session is finding its shape."
    elif Sm < 0.25:
        depth_line = "Deep in the zone now — staying close to the vibe."
    else:
        depth_line = f"{n_played + 1} songs in and still going strong."

    # ── 4. Affect-coloured song description ───────────────────────────────────
    descriptors = []
    if affect:
        try:
            arousal = float(affect.get("arousal", 0.5))
            valence = float(affect.get("valence", 0.5))
            tension = float(affect.get("tension", 0.5))
            drive   = float(affect.get("drive",   0.5))
            if arousal > 0.70:   descriptors.append("high-energy")
            elif arousal < 0.30: descriptors.append("gentle")
            if valence > 0.70:   descriptors.append("uplifting")
            elif valence < 0.30: descriptors.append("introspective")
            if tension > 0.70:   descriptors.append("intense")
            if drive   > 0.70:   descriptors.append("driving")
            elif drive < 0.30:   descriptors.append("laid-back")
        except (TypeError, ValueError):
            pass

    desc_phrase = (", ".join(descriptors[:2]) + " ") if descriptors else ""

    # ── 5. Transition line ────────────────────────────────────────────────────
    if artist:
        transitions = [
            f"This is {desc_phrase}{title} by {artist}.",
            f"Up next — {artist} with the {desc_phrase}{title}.",
            f"Bringing you {desc_phrase}{title} from {artist}.",
        ]
    else:
        transitions = [
            f"This is {desc_phrase}{title}.",
            f"Up next — {desc_phrase}{title}.",
        ]
    transition = random.choice(transitions)

    return f"{opener} {depth_line} {transition}"


def is_music_request(text: str)    -> bool: return any(kw in text.lower() for kw in MUSIC_KEYWORDS)
def is_stop_request(text: str)     -> bool: return any(kw in text.lower() for kw in STOP_KEYWORDS)
def is_pause_request(text: str)    -> bool: return any(kw in text.lower() for kw in PAUSE_KEYWORDS)
def is_resume_request(text: str)   -> bool: return any(kw in text.lower() for kw in RESUME_KEYWORDS)
def is_skip_request(text: str)     -> bool: return any(kw in text.lower() for kw in SKIP_KEYWORDS)
def is_surprise_request(text: str) -> bool: return any(kw in text.lower() for kw in SURPRISE_KEYWORDS)


# ── Timer logic ───────────────────────────────────────────────────────────────

def parse_timer_duration(text: str) -> int | None:
    text  = text.lower()
    total = 0
    found = False
    for pattern, multiplier in [
        (r'(\d+)\s*hour', 3600),
        (r'(\d+)\s*hr',   3600),
        (r'(\d+)\s*min',  60),
        (r'(\d+)\s*sec',  1),
    ]:
        m = re.search(pattern, text)
        if m:
            total += int(m.group(1)) * multiplier
            found  = True
    return total if found else None


def _duration_readable(secs: int) -> str:
    parts = []
    if secs >= 3600:
        h = secs // 3600
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if (secs % 3600) >= 60:
        m = (secs % 3600) // 60
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if secs % 60:
        s = secs % 60
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " and ".join(parts)


def fire_timer(duration_secs: int, label: str, song_was_playing: bool):
    import time as _time
    _time.sleep(duration_secs)
    readable   = _duration_readable(duration_secs)
    alert_text = f"{readable} {'are' if 'and' in readable else 'is'} up."
    if label:            alert_text += f" {label}."
    alert_text += " Goodbye from timer."
    if song_was_playing: alert_text += " Enjoy the song."
    with alert_queue_lock:
        timer_alert_queue.append({"text": alert_text, "song_was_playing": song_was_playing})
    print(f"⏱️  Timer fired — queued alert: {alert_text}")


def start_timer(duration_secs: int, label: str = "", song_was_playing: bool = False):
    threading.Thread(
        target=fire_timer, args=(duration_secs, label, song_was_playing), daemon=True
    ).start()


# ── CLAP semantic search ──────────────────────────────────────────────────────

def embed_text_query(query: str) -> np.ndarray | None:
    if not CLAP_AVAILABLE:
        return None
    try:
        import torch
        inputs = clap_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = clap_model.get_text_features(**inputs)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            elif hasattr(emb, "last_hidden_state"):
                emb = emb.last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().numpy()
    except Exception as e:
        print(f"[CLAP] ❌ Text embedding failed: {e}")
        return None


def cosine_similarity(a: list, b: np.ndarray) -> float:
    return float(np.dot(np.array(a), b))


def clap_vector_search(user_text: str, songs: list, n: int = CLAP_CANDIDATES) -> list:
    query_vec       = embed_text_query(user_text)
    with_emb        = [s for s in songs if s.get("clap_embedding")]
    without_emb     = [s for s in songs if not s.get("clap_embedding")]
    if query_vec is None or not with_emb:
        return keyword_fallback(user_text, songs, n)
    scored = sorted(
        [(cosine_similarity(s["clap_embedding"], query_vec), s) for s in with_emb],
        key=lambda x: -x[0],
    )
    top_n = [s for _, s in scored[:n]]
    print(f"  🔮 CLAP top-5 scores: {[round(sc, 3) for sc, _ in scored[:5]]}")
    if len(top_n) < n and without_emb:
        top_n += random.sample(without_emb, min(n - len(top_n), len(without_emb)))
    return top_n


def keyword_fallback(user_text: str, songs: list, n: int) -> list:
    t = user_text.lower()
    if any(kw in t for kw in UNHEARD_KEYWORDS):
        return sorted(songs, key=lambda s: last_played(s) or datetime.min)[:n]
    scored = []
    for s in songs:
        c        = s.get("classification", {})
        good_for = [g.lower() for g in c.get("good_for", [])]
        genres   = [g.lower() for g in c.get("genres", [])]
        theme    = c.get("theme_description", "").lower()
        score    = sum(2 for f in good_for + genres if f and f in t)
        score   += sum(1 for w in t.split() if len(w) > 4 and w in theme)
        if score > 0:
            scored.append((score, s))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:n]]
    return random.sample(songs, min(n, len(songs)))


# ── Request classifier ────────────────────────────────────────────────────────

def extract_search_term(text: str) -> str:
    words    = text.lower().split()
    filtered = [w for w in words if w not in SOURCE_STRIP]
    return " ".join(filtered).strip()


def fuzzy_source_match(user_text: str, songs: list) -> list[tuple[int, dict]]:
    if not RAPIDFUZZ_AVAILABLE:
        return []
    search_term = extract_search_term(user_text)
    if len(search_term) < 3:
        return []
    print(f"  🔍 Fuzzy search term: '{search_term}'")
    matched = []
    seen    = set()
    for s in songs:
        fp     = s.get("filepath", "")
        c      = s.get("classification", {})
        title  = c.get("title", "")
        artist = c.get("artist", "") or ""
        best = max(
            fuzz.token_set_ratio(search_term, os.path.basename(fp).lower()),
            fuzz.token_set_ratio(search_term, title.lower()),
            fuzz.token_set_ratio(search_term, artist.lower()),
        )
        if best >= FUZZY_THRESHOLD and fp not in seen:
            matched.append((best, s))
            seen.add(fp)
    if matched:
        matched.sort(key=lambda x: -x[0])
        return matched
    return []


def classify_request(text: str, songs: list) -> tuple[str, list]:
    t               = text.lower()
    is_continuation = any(kw in t for kw in CONTINUATION_KEYWORDS)
    matched         = fuzzy_source_match(text, songs)
    if matched and not is_continuation:
        return "source", matched
    if is_continuation:
        return "continuation", []
    return "vibe", []


# ── Song picker ───────────────────────────────────────────────────────────────

def build_song_summary(i: int, s: dict) -> str:
    c          = s.get("classification", {})
    lp         = last_played(s)
    lp_str     = lp.strftime("%Y-%m-%d") if lp else "never"
    play_count = len(s.get("play_history", []))
    title      = c.get("title", "?")
    artist     = c.get("artist", "unknown")
    affect     = c.get("affect", {})
    affect_str = (
        f"arousal={affect.get('arousal','?')} valence={affect.get('valence','?')} "
        f"tension={affect.get('tension','?')} drive={affect.get('drive','?')}"
    ) if affect else f"energy={c.get('energy_level','?')}"
    theme      = c.get("theme_description") or ", ".join(c.get("mood", []))
    narrative  = c.get("narrative", {})
    setting    = narrative.get("setting", "")
    arc        = narrative.get("emotional_arc", "")
    inst       = c.get("instrumentation", {})
    instruments= ", ".join(inst.get("primary", [])) if inst else ""
    inst_char  = inst.get("character", "") if inst else ""
    parts = [
        f'[{i}] "{title} - {artist}"',
        f'affect: {affect_str}',
        f'theme: {theme[:120]}'         if theme       else None,
        f'setting: {setting} | {arc}'   if setting or arc else None,
        f'instruments: {instruments} ({inst_char})' if instruments else None,
        f'good_for: {c.get("good_for")}',
        f'last_played: {lp_str} | plays: {play_count}',
    ]
    return " | ".join(p for p in parts if p)


def build_context_summary(ctx: dict) -> str:
    c      = ctx.get("classification", {})
    affect = c.get("affect", {})
    affect_str = (
        f"arousal={affect.get('arousal','?')} valence={affect.get('valence','?')} "
        f"tension={affect.get('tension','?')} drive={affect.get('drive','?')}"
    ) if affect else f"energy={c.get('energy_level','?')}"
    return (
        f'"{ctx["title"]} by {ctx["artist"]}" | '
        f'affect: {affect_str} | '
        f'genres: {", ".join(c.get("genres", []))} | '
        f'theme: {c.get("theme_description", "")[:100]}'
    )


def _apply_profile_weights(candidates: list, weights: dict) -> list:
    scored = [
        ((len(candidates) - i) * weights.get(s.get("filepath", ""), 1.0), s)
        for i, s in enumerate(candidates)
    ]
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]


def pick_song(
    user_text: str, db: dict,
    request_type: str = "vibe",
    source_candidates: list = None,
    is_skip: bool = False,
    is_surprise: bool = False,
) -> dict | None:
    print(f"[PICK] request_type='{request_type}', is_skip={is_skip}, is_surprise={is_surprise}")
    songs = list(db["songs"].values())
    if not songs:
        return None
    candidate_weights = profile_get_candidate_weights(songs)

    # ── Surprise path — pick from the LOW end of the grain scores ────────────
    # This is the opposite of normal operation: we deliberately score all
    # candidates and pick from the bottom quartile, forcing genuine novelty.
    if is_surprise:
        with session_played_lock:
            _played_s = set(session_played_fps)
        eligible_surprise = [s for s in songs if s.get("filepath") not in _played_s] or songs
        scored_surprise = grain.score_candidates(
            eligible_surprise, candidate_weights, exclude_fps=_played_s
        )
        if scored_surprise:
            # Take the bottom 25% (at least 5 candidates) — these are the
            # farthest-from-current-vibe songs, which is exactly what we want
            n_bottom = max(5, len(scored_surprise) // 4)
            bottom_pool = scored_surprise[-n_bottom:]
            # Uniform random within the surprise pool (don't weight by score)
            chosen_s = random.choice([s for _, s in bottom_pool])
        else:
            chosen_s = random.choice(eligible_surprise)
        return {"song": chosen_s, "reason": "surprise pick", "is_surprise": True}

    # ── Source path ───────────────────────────────────────────────────────────
    if request_type == "source" and source_candidates:
        with session_played_lock:
            _played = set(session_played_fps)
        last_fp  = last_played_context.get("filepath") if last_played_context else None
        eligible = [(sc, s) for sc, s in source_candidates
                    if s.get("filepath") not in _played] \
                   or source_candidates  # fall back to all if every match was already played
        eligible = [(sc * candidate_weights.get(s.get("filepath", ""), 1.0), s)
                    for sc, s in eligible]
        eligible.sort(key=lambda x: -x[0])
        top_score = eligible[0][0]
        top_tier  = [s for sc, s in eligible if sc >= top_score * 0.95]
        chosen    = random.choice(top_tier)
        return {"song": chosen, "reason": "matched your source request"}

    # ── Continuation path ─────────────────────────────────────────────────────
    if request_type == "continuation" or is_skip:
        direction  = user_text if not is_skip else "play something different"
        with session_played_lock:
            _played = set(session_played_fps)
        # Pre-filter songs already played this session; fall back to all if exhausted
        eligible_songs = [s for s in songs if s.get("filepath") not in _played] or songs
        candidates = clap_vector_search(direction, eligible_songs)
        candidates = _apply_profile_weights(candidates, candidate_weights)

        if last_played_context:
            ctx_summary    = build_context_summary(last_played_context)
            song_summaries = [build_song_summary(i, s) for i, s in enumerate(candidates)]
            bucket         = _get_time_bucket()
            with profile_lock:
                tod_weights = _load_profile()["time_of_day"].get(bucket, {})

            prompt = f"""You are TARA's music selector.

The song that just played was:
{ctx_summary}

The user now says: "{direction}"

Current time of day: {bucket}
User's {bucket} affect preference: {tod_weights}

Here are candidate songs:
{chr(10).join(song_summaries)}

Instructions:
- Understand the user's direction relative to the previous song.
- "Similar" → match affect profile closely.
- "More upbeat / energetic" → higher arousal and drive.
- "Slower / calmer" → lower arousal and tension.
- "Something different" → contrast the previous song's affect.
- Prefer songs matching the user's time-of-day affect preference.
- Avoid replaying the previous song. Prefer songs not played recently.

Respond ONLY with JSON, no markdown:
{{"index": <number>, "reason": "<one sentence>"}}"""

            try:
                response = ollama.chat(model="gemma2:9b", messages=[
                    {'role': 'system', 'content': 'You are a music selector. Always respond with valid JSON only.'},
                    {'role': 'user',   'content': prompt},
                ])
                raw = response['message']['content'].strip()
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"): raw = raw[4:]
                result = json.loads(raw.strip())
                idx    = int(result["index"])
                if 0 <= idx < len(candidates):
                    return {"song": candidates[idx], "reason": result.get("reason", "")}
            except json.JSONDecodeError:
                return {"song": random.choice(candidates), "reason": "random continuation pick (JSON error)"}
            except Exception as e:
                print(f"[LLM] ❌ Continuation LLM failed: {e}")
                return {"song": random.choice(candidates), "reason": "random continuation pick"}

        return {"song": random.choice(candidates), "reason": "continuation pick"}

    # ── Vibe path ─────────────────────────────────────────────────────────────
    with session_played_lock:
        _played_vibe = set(session_played_fps)
    eligible_songs_vibe = [s for s in songs if s.get("filepath") not in _played_vibe] or songs
    candidates = clap_vector_search(user_text, eligible_songs_vibe)
    candidates = _apply_profile_weights(candidates, candidate_weights)
    print(f"🎯 {len(candidates)} candidates from {len(songs)} songs")

    bucket = _get_time_bucket()
    with profile_lock:
        tod_weights = _load_profile()["time_of_day"].get(bucket, {})

    song_summaries = [build_song_summary(i, s) for i, s in enumerate(candidates)]
    prompt = f"""You are TARA's music selector. The user said: "{user_text}"

Current time of day: {bucket}
User's {bucket} affect preference (learned from history): {tod_weights}

Here are the candidate songs with their emotional and thematic profiles:
{chr(10).join(song_summaries)}

Instructions:
- Pick the SINGLE best song for this request.
- Consider the user's mood, context, and any time/activity cues.
- Use the time-of-day preference to break ties.
- Prefer songs not played recently unless specifically requested.
- If the request is vague, use the time-of-day affect preference.

Respond ONLY with JSON, no markdown:
{{"index": <number>, "reason": "<one sentence>"}}"""

    try:
        response = ollama.chat(model="gemma2:9b", messages=[
            {'role': 'system', 'content': 'You are a music selector. Always respond with valid JSON only.'},
            {'role': 'user',   'content': prompt},
        ])
        raw = response['message']['content'].strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        result = json.loads(raw.strip())
        idx    = int(result["index"])
        if idx < 0 or idx >= len(candidates):
            raise ValueError(f"Index {idx} out of range")
        return {"song": candidates[idx], "reason": result.get("reason", "")}
    except json.JSONDecodeError:
        return {"song": random.choice(candidates), "reason": "random pick (JSON error)"}
    except Exception as e:
        print(f"[LLM] ❌ Song picker LLM failed: {e} — picking randomly")
        return {"song": random.choice(candidates), "reason": "random pick"}


# ── Voice response ─────────────────────────────────────────────────────────────

def ask_tara(user_text: str) -> str:
    user_ctx = profile_get_user_context()
    system   = (
        "You are TARA. Be extremely brief. No markdown or emojis. "
        f"Context about the user: {user_ctx}"
    )
    try:
        response = ollama.chat(model="gemma2:9b", messages=[
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': user_text},
        ])
        return response['message']['content']
    except Exception as e:
        print(f"[TARA] ❌ ollama.chat failed: {e}")
        return "Sorry, I had trouble thinking of a response."


# ── Timer/Time text handler ───────────────────────────────────────────────────

def _handle_time_or_timer_text(text: str) -> str | None:
    t_lower = text.lower()
    if any(kw in t_lower for kw in TIME_KEYWORDS):
        now = datetime.now().strftime("%I:%M %p")
        return clean_text(f"The current time is {now}.")
    if any(kw in t_lower for kw in TIMER_KEYWORDS):
        duration = parse_timer_duration(text)
        if duration and duration > 0:
            readable = _duration_readable(duration)
            start_timer(duration, song_was_playing=False)
            return clean_text(f"Timer set for {readable}.")
        return clean_text("Sorry, I couldn't understand the timer duration.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    print(f"[HEALTH] /health polled.")
    return JSONResponse({
        "status":      "ok",
        "last_played": last_played_context.get("title") if last_played_context else None,
        "timestamp":   datetime.now().isoformat(),
        "grain":       grain.snapshot(),
    })


# ── TTS endpoint ──────────────────────────────────────────────────────────────

@app.post("/tts")
async def tts_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Lightweight TTS for spoken song intros. Accepts JSON {text} → audio/wav."""
    req_id   = str(uuid.uuid4())
    temp_out = f"tts_{req_id}.wav"
    try:
        body = await request.json()
        text = body.get("text", "").strip()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)
    if not text:
        return JSONResponse({"status": "error", "message": "No text provided"}, status_code=400)
    print(f"[TTS] /tts: '{text[:80]}'")
    try:
        synthesize_to_file(clean_text(text), temp_out)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    background_tasks.add_task(cleanup_task, [temp_out])
    return FileResponse(temp_out, media_type="audio/wav")


# ── Wake clip receiver (replaces Pi-side ring buffer storage) ─────────────────

_WAKE_LABEL_VARIANTS = {
    "hey jarvis", "hey travis", "hey harris", "hey paris",
    "jarvis", "hey jars", "hey charvis", "hey garvis",
    "hey service", "hey nervous", "hey harvest",
}


@app.post("/log_wake_clip")
async def log_wake_clip(
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Receive a ring-buffer WAV clip + JSON metadata from the Pi.
    Auto-label it with Whisper and store in the appropriate labeled sub-folder.
    This replaces the Pi-side _auto_label_clip loop entirely.
    """
    print("[WAKE_CLIP] /log_wake_clip received.")
    try:
        form      = await request.form()
        wav_field = form.get("file")
        meta_field= form.get("metadata")

        if wav_field is None:
            return JSONResponse({"status": "error", "message": "no file field"}, status_code=400)

        wav_bytes = await wav_field.read()
        meta      = {}
        if meta_field:
            try:
                raw_meta = await meta_field.read() if hasattr(meta_field, "read") else meta_field
                meta = json.loads(raw_meta)
            except Exception as e:
                print(f"[WAKE_CLIP] ⚠️  Could not parse metadata: {e}")

        # Write WAV to temp file for Whisper
        req_id   = str(uuid.uuid4())
        temp_wav = f"wake_tmp_{req_id}.wav"
        with open(temp_wav, "wb") as f:
            f.write(wav_bytes)

        # RMS check
        import struct, math
        samples = struct.unpack_from(f"<{(len(wav_bytes)-44)//2}h", wav_bytes, 44) \
                  if len(wav_bytes) > 44 else ()
        rms = math.sqrt(sum(s*s for s in samples) / len(samples)) / 32768 \
              if samples else 0.0
        print(f"[WAKE_CLIP] RMS={rms:.4f}")

        if rms < 0.005:
            label      = "false_positive"
            transcript = ""
        else:
            segments, _ = whisper_model.transcribe(
                temp_wav,
                beam_size=5,
                language="en",
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
                initial_prompt="Hey Jarvis",
            )
            transcript = "".join(s.text for s in segments).strip().lower()
            print(f"[WAKE_CLIP] Whisper transcript: '{transcript}'")
            if any(v in transcript for v in _WAKE_LABEL_VARIANTS):
                label = "true_positive"
            elif transcript == "":
                label = "false_positive"
            else:
                label = "uncertain"

        dest_dir = {
            "true_positive":  WAKE_LABEL_TP_DIR,
            "false_positive": WAKE_LABEL_FP_DIR,
            "uncertain":      WAKE_LABEL_UNCERTAIN_DIR,
        }[label]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_name  = f"wake_{timestamp}.wav"
        dest_wav  = os.path.join(dest_dir, wav_name)
        dest_meta = dest_wav.replace(".wav", ".json")

        os.rename(temp_wav, dest_wav)

        meta.update({
            "label":             label,
            "whisper_transcript": transcript,
            "auto_label_rms":    round(rms, 4),
            "stored_at":         datetime.now().isoformat(),
        })
        with open(dest_meta, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[WAKE_CLIP] ✅ Stored as {label}: {wav_name}")
        return JSONResponse({"status": "ok", "label": label, "transcript": transcript})

    except Exception as e:
        print(f"[WAKE_CLIP] ❌ Error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Autonomous next song (grain physics) ─────────────────────────────────────

@app.get("/auto_next_song")
async def auto_next_song():
    """
    Called by Pi after a song ends naturally.
    Returns the next song as JSON — no LLM, no disk I/O beyond DB read.
    Typical execution: <50ms.

    Uses GrainState (grain) which holds the full thermodynamic session state:
      W(v) = exp(κ · μ·v) · exp(−β·V(v)) · profile_weight(v)
    Samples from the top-K candidates weighted by W.
    """
    t0 = time.time()
    print("[AUTO_NEXT] /auto_next_song called")

    db = load_db()
    if not db["songs"]:
        print("[AUTO_NEXT] No songs in library.")
        return JSONResponse({"status": "no_songs"})

    songs    = list(db["songs"].values())

    # ── Inject time-of-day annealing temperature ──────────────────────────────
    # T controls grain boundary sharpness (how strictly "frontier" is enforced).
    # We modulate it by the time-of-day multiplier and the profile's affect
    # certainty so mornings explore wider and late nights commit to a vibe.
    bucket     = _get_time_bucket()
    tod_mult   = GRAIN_TOD_WEIGHTS.get(bucket, 1.0)
    with grain._lock:
        # Clamp so T never falls below the residual base * tod_mult
        grain.T = max(grain.T, GRAIN_T_BASE * tod_mult)

    candidate_weights = profile_get_candidate_weights(songs)

    # Exclude every song played/skipped this session
    with session_played_lock:
        exclude = set(session_played_fps)
    scored = grain.score_candidates(songs, candidate_weights, exclude_fps=exclude)

    if not scored:
        print("[AUTO_NEXT] ⚠️  No scored candidates — falling back to random")
        chosen = random.choice(songs)
    else:
        # Build a set of embeddings for recently played songs to enforce diversity
        with session_played_lock:
            played_fps_snap = set(session_played_fps)
        played_embs = []
        for fp in played_fps_snap:
            sd = db["songs"].get(fp, {})
            raw = sd.get("clap_embedding")
            if raw and len(raw) == 512:
                v = np.array(raw, dtype=np.float32)
                n_ = np.linalg.norm(v)
                if n_ > 0:
                    played_embs.append(v / n_)
        played_matrix = np.stack(played_embs, axis=0) if played_embs else None  # (P, 512)

        def _too_similar(song: dict) -> bool:
            """Return True if song is too close to any already-played song."""
            if played_matrix is None:
                return False
            raw = song.get("clap_embedding")
            if not raw or len(raw) != 512:
                return False
            v = np.array(raw, dtype=np.float32)
            n_ = np.linalg.norm(v)
            if n_ == 0:
                return False
            v = v / n_
            sims = played_matrix @ v  # (P,)
            return bool(np.any(sims >= SESSION_SIMILARITY_THRESHOLD))

        # Weighted sample from top-5, preferring songs dissimilar to played ones
        top = scored[:5]
        diverse_top = [(s, song) for s, song in top if not _too_similar(song)]
        if not diverse_top:
            diverse_top = top  # all are too similar — allow repeats rather than crash
        weights = [max(s, 1e-9) for s, _ in diverse_top]
        total   = sum(weights)
        weights = [w / total for w in weights]
        chosen  = random.choices([song for _, song in diverse_top], weights=weights, k=1)[0]

    filepath = chosen["filepath"]
    c        = chosen.get("classification", {})
    title    = c.get("title",  os.path.basename(filepath))
    artist   = c.get("artist", "")
    affect   = c.get("affect", {})

    snap   = grain.snapshot()
    reason = "auto"   # not surfaced in speech anymore — intro is context-built

    spoken_intro = build_spoken_intro(
        title, artist, reason,
        affect=affect,
        grain_snap=snap,
    )
    record_play(db, filepath)

    elapsed_ms = (time.time() - t0) * 1000
    print(
        f"[AUTO_NEXT] ✅ '{title}' | Sₘ={snap['Sm']:.3f} r={snap['r']:.3f} "
        f"d̄={snap['d_bar']:.3f} T={snap['T']:.3f} | {elapsed_ms:.0f}ms"
    )

    return JSONResponse({
        "status":       "play_local",
        "filepath":     mac_to_pi_path(filepath),
        "title":        title,
        "artist":       artist,
        "reason":       reason,
        "spoken_intro": spoken_intro,
        "grain_state":  snap,
    })


# ── Interaction logger ────────────────────────────────────────────────────────

@app.post("/log_interaction")
async def log_interaction(request: Request):
    """Receive WAV bytes + metadata from Pi and store for training."""
    print(f"[LOG_INTERACTION] Request received.")
    try:
        form       = await request.form()
        today      = datetime.now().strftime("%Y-%m-%d")
        meta_raw   = None
        session_id = None

        for field_name, field_value in form.items():
            if field_name == "metadata":
                try:
                    content    = await field_value.read() if hasattr(field_value, 'read') else field_value
                    meta_raw   = json.loads(content)
                    session_id = meta_raw.get("session_id", uuid.uuid4().hex[:12])
                except Exception as e:
                    session_id = uuid.uuid4().hex[:12]
                    print(f"[LOG_INTERACTION] ❌ Metadata parse failed: {e}")

        if not session_id:
            session_id = uuid.uuid4().hex[:12]

        session_dir = os.path.join(INTERACTION_STORE_PATH, today, session_id)
        os.makedirs(session_dir, exist_ok=True)

        for field_name in ["command_wav", "trigger_wav"]:
            fv = form.get(field_name)
            if fv and hasattr(fv, 'read'):
                content   = await fv.read()
                suffix    = "command.wav" if field_name == "command_wav" else "trigger.wav"
                dest_path = os.path.join(session_dir, suffix)
                with open(dest_path, "wb") as f:
                    f.write(content)
                print(f"[LOG_INTERACTION] ✅ Saved {field_name} ({len(content)} bytes)")

        if meta_raw:
            meta_raw["stored_at"] = datetime.now().isoformat()
            with open(os.path.join(session_dir, "metadata.json"), "w") as f:
                json.dump(meta_raw, f, indent=2)

        with profile_lock:
            profile = _load_profile()
            profile["total_interactions"] = profile.get("total_interactions", 0) + 1
            profile["last_updated"]       = datetime.now().isoformat()
            _save_profile(profile)

        print(f"📦 Interaction stored: {today}/{session_id}")
        return JSONResponse({"status": "ok", "session_id": session_id})

    except Exception as e:
        print(f"[LOG_INTERACTION] ❌ Unhandled error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Playback reporter ─────────────────────────────────────────────────────────

class PlaybackReport(BaseModel):
    filepath:  str
    title:     str
    elapsed:   float
    completed: bool
    timestamp: str = ""


@app.post("/report_playback")
async def report_playback(body: PlaybackReport):
    """
    Receive play completion/skip signal from Pi.
    body.filepath is the Pi-side USB path; translate to Mac path for DB lookup.
    Also drives grain state updates:
      - Completed → grain grows (update_completion)
      - Skipped   → recrystallization (update_skip)
    """
    print(f"[REPORT_PLAYBACK] title='{body.title}', elapsed={body.elapsed:.1f}s, completed={body.completed}")
    try:
        mac_filepath = pi_to_mac_path(body.filepath)
        print(f"[REPORT_PLAYBACK] Pi path '{body.filepath}' → Mac path '{mac_filepath}'")

        db = load_db()

        # Extract CLAP embedding for the played/skipped song
        song_data = db.get("songs", {}).get(mac_filepath, {})
        raw_emb   = song_data.get("clap_embedding")
        embedding = None
        if raw_emb and len(raw_emb) == 512:
            v    = np.array(raw_emb, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                embedding = v / norm

        # Update grain physics state
        if body.completed:
            grain.update_completion(embedding)
        else:
            songs             = list(db["songs"].values())
            candidate_weights = profile_get_candidate_weights(songs)
            grain.update_skip(embedding, songs, candidate_weights)

        # Update user profile
        profile_record_play(
            filepath  = mac_filepath,
            elapsed   = body.elapsed,
            completed = body.completed,
            db        = db,
        )
        label = "completed" if body.completed else f"skipped at {body.elapsed:.0f}s"
        print(f"📊 Playback reported: '{body.title}' — {label} | grain={grain.snapshot()}")
        return JSONResponse({"status": "ok", "grain_state": grain.snapshot()})
    except Exception as e:
        print(f"[REPORT_PLAYBACK] ❌ Error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Correction reporter ───────────────────────────────────────────────────────

class CorrectionReport(BaseModel):
    heard:     str
    intended:  str
    timestamp: str = ""


@app.post("/report_correction")
async def report_correction(body: CorrectionReport):
    print(f"[REPORT_CORRECTION] heard='{body.heard}' → intended='{body.intended}'")
    try:
        profile_record_correction(heard=body.heard, intended=body.intended)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        print(f"[REPORT_CORRECTION] ❌ Error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Timer alert endpoint ──────────────────────────────────────────────────────

@app.get("/timer_alert")
async def timer_alert():
    with alert_queue_lock:
        if not timer_alert_queue:
            return JSONResponse({"status": "none"})
        alert = timer_alert_queue.popleft()
    alert_text = clean_text(alert["text"])
    req_id     = str(uuid.uuid4())
    out_path   = f"alert_{req_id}.wav"
    print(f"[TIMER_ALERT] Synthesizing alert: '{alert_text}'")
    try:
        synthesize_to_file(alert_text, out_path)
        return FileResponse(
            out_path, media_type="audio/wav",
            headers={"X-Alert-Text": alert_text}
        )
    except Exception as e:
        print(f"[TIMER_ALERT] ❌ Alert synthesis failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)})


# ── Fast text command endpoint ────────────────────────────────────────────────

class FastTextRequest(BaseModel):
    text: str


@app.post("/fast_text_command")
async def fast_text_command(body: FastTextRequest, background_tasks: BackgroundTasks):
    """Zero-latency time/timer endpoint — text in, WAV out, no Whisper."""
    text = body.text.strip()
    if not text:
        return JSONResponse({"status": "error", "message": "empty text"})
    req_id    = str(uuid.uuid4())
    temp_out  = f"fast_{req_id}.wav"
    tara_text = _handle_time_or_timer_text(text)
    if tara_text is None:
        return JSONResponse({
            "status":  "error",
            "message": "No time/timer intent found. Use /process_audio for music.",
        })
    print(f"⚡ Fast response: {tara_text}")
    try:
        synthesize_to_file(tara_text, temp_out)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})
    background_tasks.add_task(cleanup_task, [temp_out])
    return FileResponse(temp_out, media_type="audio/wav")


# ── Main audio processing endpoint ───────────────────────────────────────────

@app.post("/process_audio")
async def process_audio(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
):
    req_id   = str(uuid.uuid4())
    temp_in  = f"in_{req_id}.wav"
    temp_out = f"out_{req_id}.wav"

    mid_song = request.headers.get("X-Mid-Song", "false").lower() == "true"
    print(f"\n{'='*60}")
    print(f"[PROCESS] /process_audio — req_id={req_id}, mid_song={mid_song}")

    raw_bytes = await file.read()
    with open(temp_in, "wb") as f:
        f.write(raw_bytes)

    # ── Whisper transcription ─────────────────────────────────────────────────
    db_for_prompt  = load_db()
    whisper_prompt = build_whisper_prompt(db_for_prompt)

    try:
        segments, info = whisper_model.transcribe(
            temp_in,
            beam_size=10,
            language="en",
            initial_prompt=whisper_prompt,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
            hotwords="Jarvis, play, pause, stop, resume, skip, timer, volume",
        )
        transcription = "".join([s.text for s in segments]).strip()
        print(f"[WHISPER] ✅ '{transcription}'")
    except Exception as e:
        print(f"[WHISPER] ❌ Transcription failed: {e}")
        background_tasks.add_task(cleanup_task, [temp_in])
        return JSONResponse(
            {"status": "error", "message": f"Whisper failed: {e}"},
            headers={"X-Whisper-Transcript": ""},
        )

    if not transcription:
        background_tasks.add_task(cleanup_task, [temp_in])
        return JSONResponse(
            {"status": "no_speech"},
            headers={"X-Whisper-Transcript": ""},
        )

    print(f"👂 Heard: {transcription} {'[mid-song]' if mid_song else ''}")
    t_lower       = transcription.lower()
    extra_headers = {"X-Whisper-Transcript": transcription[:200]}

    # ── Mid-song restricted mode ──────────────────────────────────────────────
    if mid_song:
        if is_stop_request(transcription):
            reset_session()
            background_tasks.add_task(cleanup_task, [temp_in])
            return JSONResponse({"action": "stop"}, headers=extra_headers)

        if is_pause_request(transcription):
            background_tasks.add_task(cleanup_task, [temp_in])
            return JSONResponse({"action": "pause"}, headers=extra_headers)

        if is_resume_request(transcription):
            background_tasks.add_task(cleanup_task, [temp_in])
            return JSONResponse({"action": "resume"}, headers=extra_headers)

        if is_skip_request(transcription):
            db = load_db()

            # Grain recrystallization for voice-triggered skip
            last_fp_skip = last_played_context.get("filepath") if last_played_context else None
            if last_fp_skip:
                with session_played_lock:
                    session_played_fps.add(last_fp_skip)  # mark skipped song as used
                skip_song = db.get("songs", {}).get(last_fp_skip, {})
                raw_emb = skip_song.get("clap_embedding")
                skip_emb = None
                if raw_emb and len(raw_emb) == 512:
                    v = np.array(raw_emb, dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        skip_emb = v / norm
                candidate_weights_skip = profile_get_candidate_weights(list(db["songs"].values()))
                grain.update_skip(skip_emb, list(db["songs"].values()), candidate_weights_skip)

            # Score candidates excluding the full session played set
            with session_played_lock:
                _skip_exclude = set(session_played_fps)
            scored = grain.score_candidates(
                list(db["songs"].values()),
                candidate_weights_skip if 'candidate_weights_skip' in locals() else profile_get_candidate_weights(
                    list(db["songs"].values())),
                exclude_fps=_skip_exclude
            )

            if not scored:
                chosen = random.choice(list(db["songs"].values()))
            else:
                top = scored[:5]
                weights = [max(s, 1e-9) for s, _ in top]
                total = sum(weights)
                weights = [w / total for w in weights]
                chosen = random.choices([s for _, s in top], weights=weights, k=1)[0]

            result = {
                "song": chosen,
                "reason": "Recrystallizing the vibe."
            }

            if result:
                song = result["song"]
                filepath = song["filepath"]
                c = song.get("classification", {})
                title = c.get("title", os.path.basename(filepath))
                artist = c.get("artist", "")
                reason = result["reason"]
                record_play(db, filepath)
                background_tasks.add_task(cleanup_task, [temp_in])
                return JSONResponse({
                    "status": "play_local",
                    "action": "skip",
                    "filepath": mac_to_pi_path(filepath),
                    "title": title,
                    "artist": artist,
                    "reason": reason,
                    "spoken_intro": build_spoken_intro(
                        title, artist, reason,
                        affect=c.get("affect", {}),
                        grain_snap=grain.snapshot(),
                    ),
                }, headers=extra_headers)

        if any(kw in t_lower for kw in TIME_KEYWORDS):
            now       = datetime.now().strftime("%I:%M %p")
            tara_text = clean_text(f"The current time is {now}.")
            try:
                synthesize_to_file(tara_text, temp_out)
            except Exception as e:
                background_tasks.add_task(cleanup_task, [temp_in])
                return JSONResponse({"status": "error", "message": str(e)})
            background_tasks.add_task(cleanup_task, [temp_in, temp_out])
            return FileResponse(temp_out, media_type="audio/wav",
                                headers={**extra_headers, "X-Mid-Song-Response": "true"})

        if any(kw in t_lower for kw in TIMER_KEYWORDS):
            duration = parse_timer_duration(transcription)
            if duration and duration > 0:
                readable  = _duration_readable(duration)
                tara_text = clean_text(f"Timer set for {readable}.")
                start_timer(duration, song_was_playing=True)
            else:
                tara_text = clean_text("Sorry, I couldn't understand the timer duration.")
            try:
                synthesize_to_file(tara_text, temp_out)
            except Exception as e:
                background_tasks.add_task(cleanup_task, [temp_in])
                return JSONResponse({"status": "error", "message": str(e)})
            background_tasks.add_task(cleanup_task, [temp_in, temp_out])
            return FileResponse(temp_out, media_type="audio/wav",
                                headers={**extra_headers, "X-Mid-Song-Response": "true"})

        if is_music_request(transcription):
            tara_text = clean_text(
                "I'll finish this song first. You can skip, pause, or stop anytime."
            )
            try:
                synthesize_to_file(tara_text, temp_out)
            except Exception as e:
                background_tasks.add_task(cleanup_task, [temp_in])
                return JSONResponse({"status": "error", "message": str(e)})
            background_tasks.add_task(cleanup_task, [temp_in, temp_out])
            return FileResponse(temp_out, media_type="audio/wav",
                                headers={**extra_headers, "X-Mid-Song-Response": "true"})

        tara_text = clean_text(
            "While a song is playing I can help with: stop, pause, resume, skip, time, and timers."
        )
        try:
            synthesize_to_file(tara_text, temp_out)
        except Exception as e:
            background_tasks.add_task(cleanup_task, [temp_in])
            return JSONResponse({"status": "error", "message": str(e)})
        background_tasks.add_task(cleanup_task, [temp_in, temp_out])
        return FileResponse(temp_out, media_type="audio/wav",
                            headers={**extra_headers, "X-Mid-Song-Response": "true"})

    # ── Normal mode ───────────────────────────────────────────────────────────

    if any(kw in t_lower for kw in TIME_KEYWORDS):
        now       = datetime.now().strftime("%I:%M %p")
        tara_text = clean_text(f"The current time is {now}.")
        try:
            synthesize_to_file(tara_text, temp_out)
        except Exception as e:
            background_tasks.add_task(cleanup_task, [temp_in])
            return JSONResponse({"status": "error", "message": str(e)})
        background_tasks.add_task(cleanup_task, [temp_in, temp_out])
        return FileResponse(temp_out, media_type="audio/wav", headers=extra_headers)

    if any(kw in t_lower for kw in TIMER_KEYWORDS):
        duration = parse_timer_duration(transcription)
        if duration and duration > 0:
            readable  = _duration_readable(duration)
            tara_text = clean_text(f"Timer set for {readable}.")
            start_timer(duration, song_was_playing=False)
        else:
            tara_text = clean_text("Sorry, I couldn't understand the timer duration.")
        try:
            synthesize_to_file(tara_text, temp_out)
        except Exception as e:
            background_tasks.add_task(cleanup_task, [temp_in])
            return JSONResponse({"status": "error", "message": str(e)})
        background_tasks.add_task(cleanup_task, [temp_in, temp_out])
        return FileResponse(temp_out, media_type="audio/wav", headers=extra_headers)

    if is_music_request(transcription):
        db = load_db()
        if not db["songs"]:
            tara_text = clean_text(
                "I don't have a music library yet. Please run build_library first."
            )
            try:
                synthesize_to_file(tara_text, temp_out)
            except Exception as e:
                background_tasks.add_task(cleanup_task, [temp_in])
                return JSONResponse({"status": "error", "message": str(e)})
            background_tasks.add_task(cleanup_task, [temp_in, temp_out])
            return FileResponse(temp_out, media_type="audio/wav", headers=extra_headers)

        songs    = list(db["songs"].values())
        _surprise = is_surprise_request(transcription)

        if _surprise:
            # Bypass classify_request entirely — go straight to surprise pick
            result = pick_song(transcription, db, is_surprise=True)
        else:
            request_type, source_candidates = classify_request(transcription, songs)
            result = pick_song(
                transcription, db,
                request_type=request_type,
                source_candidates=source_candidates,
            )

        if result:
            song         = result["song"]
            filepath     = song["filepath"]
            c            = song.get("classification", {})
            title        = c.get("title", os.path.basename(filepath))
            artist       = c.get("artist", "")
            reason       = result["reason"]
            affect       = c.get("affect", {})
            request_type = request_type if not _surprise else "surprise"

            print(f"🎵 Playing: {title}{' by ' + artist if artist else ''}")
            print(f"   Reason:  {reason}")
            print(f"   Path:    {filepath}")

            record_play(db, filepath)
            background_tasks.add_task(cleanup_task, [temp_in])
            return JSONResponse({
                "status":       "play_local",
                "filepath":     mac_to_pi_path(filepath),
                "title":        title  or "",
                "artist":       artist or "",
                "reason":       reason or "",
                "request_type": request_type,
                "spoken_intro": build_spoken_intro(
                    title, artist, reason,
                    affect=affect,
                    is_surprise=_surprise,
                    grain_snap=grain.snapshot(),
                ),
            }, headers=extra_headers)

    # Normal conversation fallback
    tara_text = clean_text(ask_tara(transcription))
    print(f"🤖 TARA: {tara_text}")
    try:
        synthesize_to_file(tara_text, temp_out)
    except Exception as e:
        import traceback; traceback.print_exc()
        background_tasks.add_task(cleanup_task, [temp_in])
        return JSONResponse({"status": "error", "message": str(e)})
    background_tasks.add_task(cleanup_task, [temp_in, temp_out])
    return FileResponse(temp_out, media_type="audio/wav", headers=extra_headers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)