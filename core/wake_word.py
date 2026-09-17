"""
Local wake-word detection for AVELIA.

The spoken phrase is a custom-trained "Hey Avelia" model, not the pretrained
"Hey Avelia" model openwakeword ships out of the box. The trained model is an
ordinary ONNX file dropped in at config/wakewords/hey_avelia.onnx (see
WAKE_MODEL_PATH below) — openwakeword happily loads a custom model by file
path exactly the way it loads one of its own pretrained ones, so nothing else
about the detection pipeline changes.

Design goals:
  • ZERO cost when the feature is off — openwakeword is imported ONLY inside
    start()/install helpers, never at module load. If the user never enables
    wake word, none of this touches the app.
  • ZERO latency on the audio path — the microphone callback only ever does a
    cheap, non-blocking queue push (feed()); the actual model inference runs in
    this module's own background thread, so the real-time audio thread and the
    Gemini stream are never slowed.
  • Fully local & offline — audio fed here never leaves the machine; there is no
    network call except the one-time download (of openwakeword's shared
    melspectrogram/embedding feature-extraction models) the user triggers from
    the UI. The wake model itself never touches the network — it's already on
    disk.

openwakeword's shared melspectrogram/embedding models are a few MB each and run
comfortably on a CPU; the custom "Hey Avelia" model is a similarly small ONNX
file trained separately (see the project's wake-word training notes).
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

# Custom-trained AVELIA wake model. WAKE_MODEL_NAME is just a label used for
# logging and for matching the right key in the scores dict openwakeword
# returns (it keys detections by the model's file stem); WAKE_MODEL_PATH is
# where the actual .onnx file must live. Drop the file trained via the
# openWakeWord Colab/Kaggle notebook at this exact path — nothing downloads it
# automatically, since it's specific to this install, not a public pretrained
# model.
WAKE_MODEL_NAME = "hey_avelia"
WAKE_MODEL_PATH = Path(__file__).resolve().parent.parent / "config" / "wakewords" / "hey_avelia.onnx"
# Score in [0,1]; above this counts as a detection. Tunable per environment.
DEFAULT_THRESHOLD = 0.5
# Mic frames arrive at 16 kHz int16; this is just the detector's input rate.
SAMPLE_RATE = 16000


def is_installed() -> bool:
    """True if the openwakeword package is importable (no model check)."""
    try:
        import importlib.util
        return importlib.util.find_spec("openwakeword") is not None
    except Exception:
        return False


def is_ready() -> bool:
    """True if openwakeword is installed, its shared feature-extraction models
    are present, AND the custom Hey Avelia model file is on disk.

    This is a cheap, DETERMINISTIC file-existence check. It deliberately does NOT
    construct a Model to probe readiness — doing that is slow and, worse, can clash
    with the detector's own Model when it's already running, which intermittently
    returned False and made the UI flicker to 'not downloaded'. Never raises.
    """
    if not is_installed():
        return False
    if not WAKE_MODEL_PATH.is_file():
        return False
    try:
        import openwakeword
        models_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
        if not models_dir.is_dir():
            return False
        has_mel = (any(models_dir.glob("melspectrogram*.onnx"))
                   or any(models_dir.glob("melspectrogram*.tflite")))
        has_emb = (any(models_dir.glob("embedding_model*.onnx"))
                   or any(models_dir.glob("embedding_model*.tflite")))
        return bool(has_mel and has_emb)
    except Exception:
        return False


def install_and_download(logger: Callable[[str], None] = print,
                         notify: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """
    One-click setup for the UI button: pip-install openwakeword if missing, then
    download openwakeword's shared feature-extraction models (melspectrogram +
    embedding — the custom Hey Avelia model itself is never downloaded here, it
    must already be sitting at WAKE_MODEL_PATH). Returns (ok, message). Never
    raises — every failure is reported through the returned message and the
    logger.
    """
    _tell = notify or (lambda _msg: None)
    try:
        if not is_installed():
            logger("Wake word: installing openwakeword (one-time)…")
            _tell("Wake word: installing openwakeword (one-time)…")
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "openwakeword"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
                return False, f"pip install failed: {tail[0][:160]}"
        # Download the shared melspectrogram/embedding feature-extraction models
        # (these are the same for every wake word, custom or pretrained).
        logger("Wake word: downloading feature-extraction models…")
        _tell("Wake word: downloading feature-extraction models…")
        try:
            import openwakeword.utils as _u
            try:
                _u.download_models(["melspectrogram", "embedding_model"])
            except TypeError:
                _u.download_models()   # older signature downloads the default set
        except Exception as e:
            return False, f"model download failed: {e}"

        if not WAKE_MODEL_PATH.is_file():
            return False, (f"custom wake model not found — place hey_avelia.onnx at "
                            f"{WAKE_MODEL_PATH}")
        if not is_ready():
            return False, "installed, but the wake model could not be loaded."
        logger("Wake word: ready.")
        return True, "Wake word installed and ready."
    except Exception as e:
        return False, f"setup error: {e}"


class WakeWordDetector:
    """
    Runs the wake model in a dedicated thread. The mic thread calls feed() with
    raw int16 frames; detections invoke on_detect() (called from this thread —
    the callback must marshal to whatever loop/UI it needs).
    """

    def __init__(self, on_detect: Callable[[], None],
                 threshold: float = DEFAULT_THRESHOLD,
                 logger: Callable[[str], None] = print,
                 notify: Callable[[str], None] | None = None):
        self._on_detect = on_detect
        self._threshold = threshold
        self._logger    = logger
        # See PluginRegistry: `logger` is the console and gets everything,
        # `notify` is the activity log and gets only what the user must act on.
        self._notify    = notify or (lambda _msg: None)
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._thread: threading.Thread | None = None
        self._running = False
        self._model = None
        self._ready = False

    def start(self) -> bool:
        """Load the model and spawn the inference thread. Returns True on success.
        Safe to call again — a no-op if already running. Never raises."""
        if self._running:
            return True
        if not WAKE_MODEL_PATH.is_file():
            self._logger(f"Wake word: model file missing — {WAKE_MODEL_PATH}")
            self._notify("Wake word unavailable — use the WAKE NOW button.")
            return False
        try:
            from openwakeword.model import Model
            self._model = Model(wakeword_models=[str(WAKE_MODEL_PATH)], inference_framework="onnx")
        except Exception as e:
            self._logger(f"Wake word: could not load model — {e}")
            self._notify("Wake word unavailable — use the WAKE NOW button.")
            self._model = None
            return False
        self._running = True
        self._ready = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="WakeWordThread")
        self._thread.start()
        self._logger("Wake word: listening for 'Hey Avelia'.")
        return True

    def stop(self) -> None:
        self._running = False
        # unblock the thread if it's waiting on the queue
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._model = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def feed(self, frame_int16) -> None:
        """Called from the mic callback (real-time thread). Must stay cheap and
        never block — the frame is copied and dropped if the queue is backed up."""
        if not self._running:
            return
        try:
            # frame_int16 is a numpy int16 array (possibly 2-D mono) — flatten to 1-D
            data = frame_int16[:, 0].copy() if getattr(frame_int16, "ndim", 1) > 1 else frame_int16.copy()
            self._queue.put_nowait(data)
        except queue.Full:
            pass
        except Exception:
            pass

    def _loop(self) -> None:
        import numpy as np
        while self._running:
            try:
                frame = self._queue.get()
                if frame is None or not self._running:
                    break
                scores = self._model.predict(np.asarray(frame, dtype=np.int16))
                score = 0.0
                if isinstance(scores, dict):
                    # match the avelia model regardless of exact key suffix
                    for k, v in scores.items():
                        if "avelia" in k.lower():
                            score = max(score, float(v))
                    if score == 0.0 and scores:
                        score = max(float(v) for v in scores.values())
                if score >= self._threshold:
                    # drain any backlog so we don't double-fire on the same utterance
                    self._drain()
                    try:
                        self._on_detect()
                    except Exception as e:
                        self._logger(f"Wake word: on_detect error — {e}")
            except Exception as e:
                self._logger(f"Wake word: inference error — {e}")

    def _drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass