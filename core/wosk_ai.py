"""
core/wosk_ai.py — the ECAPA-TDNN voice profile from Abror's earlier Wosk
project, ported into Avelia unchanged in substance.

The model, the profile (abror_profile.npy) and the 0.50 threshold are all the
ones that were already working. What changed is only what had to change to live
inside a long-running Qt application instead of a standalone script:

  * Paths resolve from the project root, not the current working directory.
    `SpeakerRecognition.from_hparams(savedir="wosk/models/...")` only found the
    model when Avelia happened to be started from the project folder; launched
    from a shortcut or a frozen build, it silently re-downloaded.

  * The Hugging Face offline switch and the hf_hub_download patch are applied
    inside the loader, not at import. Setting os.environ at module import makes
    them process-wide, and Avelia imports a lot of other things.

  * verify_array() scores a numpy array directly, so the live gate does not
    write a wav file to disk on every attempt. check_voice(path) is kept as it
    was for the CLI and for any old code that calls it.

Nothing here retrains or re-enrolls anything: it loads wosk/models/ as-is.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

SR = 16000
MIN_SECONDS = 3.0        # ECAPA needs this much speech to score reliably
SILENCE_RMS = 0.008      # below this the mic heard nothing worth scoring
THRESHOLD = 0.50         # the value the Wosk profile was already using


def _base_dir() -> Path:
    """Project root, whether run from source or from a frozen build."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


MODEL_DIR = _base_dir() / "wosk" / "models" / "speechbrain_model"
PROFILE_PATH = _base_dir() / "wosk" / "models" / "abror_profile.npy"


class WoskAI:
    """Loads the existing ECAPA encoder and the stored abror_profile.npy.

    Constructing this loads a PyTorch model — it takes seconds. Never build it
    on the asyncio loop or the Qt thread; hand it to an executor.
    """

    def __init__(self, model_dir: Path | str = MODEL_DIR,
                 profile_path: Path | str = PROFILE_PATH):
        self.model_dir = Path(model_dir)
        self.profile_path = Path(profile_path)
        self.SR = SR
        self.etalon_vector: np.ndarray | None = None
        self._infer_lock = threading.Lock()

        if not self.profile_path.exists():
            raise FileNotFoundError(
                f"Voice profile not found: {self.profile_path}. "
                f"Copy the 'wosk' folder from the old project into the Avelia root."
            )
        self.etalon_vector = np.load(self.profile_path)

        self.encoder = self._load_encoder()

    # -- model ---------------------------------------------------------------

    def _load_encoder(self):
        # Offline mode and the download patch belong here, not at import time:
        # the model is already on disk, and these switches are process-wide.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        try:
            import torchaudio
            if not hasattr(torchaudio, "set_audio_backend"):
                # Newer torchaudio dropped this API; speechbrain still calls it.
                torchaudio.set_audio_backend = lambda backend: None
                torchaudio.get_audio_backend = lambda: "soundfile"
        except ImportError:
            pass

        self._patch_hf_download()

        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import SpeakerRecognition

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Speechbrain model folder not found: {self.model_dir}. "
                f"Copy the 'wosk' folder from the old project into the Avelia root."
            )

        return SpeakerRecognition.from_hparams(
            source=str(self.model_dir),
            savedir=str(self.model_dir),
        )

    @staticmethod
    def _patch_hf_download() -> None:
        """The 'custom.py 404 + Windows symlink' workaround, applied once.

        speechbrain asks the hub for an optional custom.py that this model does
        not ship; on Windows the resulting fetch used to crash the load. The
        patch hands back a stub file instead. Applied idempotently so importing
        this module twice does not wrap the function twice.
        """
        try:
            import huggingface_hub
        except ImportError:
            return
        if getattr(huggingface_hub.hf_hub_download, "_wosk_patched", False):
            return

        original = huggingface_hub.hf_hub_download

        def patched(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            wants_custom = (
                (len(args) > 1 and args[1] == "custom.py")
                or kwargs.get("filename") == "custom.py"
            )
            if wants_custom:
                stub = _base_dir() / "wosk" / "models" / "custom_bypass.py"
                stub.parent.mkdir(parents=True, exist_ok=True)
                if not stub.exists():
                    stub.write_text("# Wosk AI: hub 404 / symlink workaround\n",
                                    encoding="utf-8")
                return str(stub)
            return original(*args, **kwargs)

        patched._wosk_patched = True
        huggingface_hub.hf_hub_download = patched

    # -- scoring -------------------------------------------------------------

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        import torch
        wav = torch.tensor(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        # One encoder shared between the startup gate and any later check, so
        # inference is serialised.
        with self._infer_lock, torch.no_grad():
            return self.encoder.encode_batch(wav).squeeze().cpu().numpy()

    def verify_array(self, audio: np.ndarray) -> tuple[bool, float, str]:
        """Score raw mono float32 audio at 16 kHz.

        Returns (accepted, similarity, reason) where reason is
        ok | rejected | too_short | silence.
        """
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size < MIN_SECONDS * self.SR:
            return False, 0.0, "too_short"

        y = y - float(np.mean(y))          # DC offset, as in the original
        chunk = y[: int(MIN_SECONDS * self.SR)]

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < SILENCE_RMS:
            return False, 0.0, "silence"

        test_vector = self._embed(chunk)
        denom = np.linalg.norm(self.etalon_vector) * np.linalg.norm(test_vector)
        if denom == 0:
            return False, 0.0, "silence"
        similarity = float(np.dot(self.etalon_vector, test_vector) / denom)

        return similarity >= THRESHOLD, similarity, "ok" if similarity >= THRESHOLD else "rejected"

    def check_voice(self, test_audio_path) -> str:
        """Original string-returning API, kept for the old call sites."""
        try:
            import librosa
            y, _ = librosa.load(str(test_audio_path), sr=self.SR, mono=True)
            accepted, similarity, reason = self.verify_array(y)
            if reason == "too_short":
                return "Xato: Ovoz kamida 3 soniya bo'lishi kerak!"
            if reason == "silence":
                return "Wosk AI: Mikrofonda hech qanday nutq aniqlanmadi (Jimjitlik)."
            if accepted:
                return f"Ha, tizim tasdiqlandi. Bu siz — ABROR! | Sim: {similarity:.4f}"
            return f"Kechirasiz, men sizni tanimadim. Ovoz mos kelmadi. | Sim: {similarity:.4f}"
        except Exception as e:
            return f"Tahlilda xatolik: {e}"