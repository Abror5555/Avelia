"""
core/voice_lock.py — the startup voice gate.

This is the block that used to live inline in AveliaLive.run(), moved out and
fixed in four places. The behaviour Abror had is kept: record 4 seconds, score
against the Wosk profile, retry until it matches.

WHAT WAS WRONG WITH THE INLINE VERSION

1. The model loaded on the event loop.
       wosk_system = WoskAI()
   That constructor loads PyTorch and speechbrain — several seconds — and it
   was the one thing NOT handed to an executor, while the 4-second recording
   was. So the UI froze during loading and stayed responsive during recording,
   which is exactly backwards. Both run in the executor here.

2. A failure meant full access.
       except Exception as bio_err:
           self.ui.write_log(...)
   caught the whole loop, logged, and then fell through into the rest of run().
   A broken microphone, a missing profile or an unplugged headset therefore
   opened Avelia with no check at all — the one case where the lock mattered
   most. The policy is now explicit and named: ON_ERROR.

3. The result was a string match.
       if "Bu siz — ABROR!" in auth_result
   Rewording that message anywhere in wosk.py would have silently disabled the
   lock. verify_array() returns a boolean now.

4. The temp wav file survived failures.
   wosk_live_auth.wav was deleted only on success, so every rejected attempt
   left a recording of someone's voice in the project folder. No file is
   written at all now — the audio goes straight from sounddevice to the model.

WHAT THIS IS NOT
    A microphone lock is not authentication: a recording of Abror played at the
    mic passes it. It keeps a guest from talking to Avelia; it is not a
    password, and core/confirm.py still guards the irreversible actions.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np

RECORD_SECONDS = 4.0
RETRY_DELAY = 3.0

# What to do when the voice system itself fails — no profile, no model, no
# microphone. Not the same question as "was that Abror?", and it deserves its
# own answer rather than whatever the try/except happened to do.
#
#   "open"  — start unverified, with a loud log line. The old behaviour.
#   "stop"  — refuse to start. Correct if the lock is the point.
ON_ERROR = "open"

# 0 = retry forever, as before. Set to 3 if being locked out by a dead
# microphone sounds worse than a stranger getting three tries.
MAX_ATTEMPTS = 0


async def require_owner(
    ui,
    device=None,
    seconds: float = RECORD_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
    retry_delay: float = RETRY_DELAY,
    on_error: str = ON_ERROR,
) -> bool:
    """Hold the startup sequence until the owner's voice is recognised.

    `device` is the input device index from core/audio_devices.py — the gate
    must listen on the same microphone the rest of Avelia uses, or it records
    silence from the wrong one and rejects a perfectly good voice.

    Returns True to continue starting, False to abort.
    """
    loop = asyncio.get_running_loop()

    def _log(msg: str) -> None:
        try:
            ui.write_log(msg)
        except Exception:
            pass

    def _state(name: str) -> None:
        try:
            ui.set_state(name)
        except Exception:
            pass

    # ── load the model off the loop ─────────────────────────────────────────
    _log("SYS: Ovoz biometriyasi yuklanmoqda...")
    _state("THINKING")
    try:
        from core.wosk_ai import WoskAI
        wosk = await loop.run_in_executor(None, WoskAI)
    except Exception as e:
        _log(f"ERR: Ovoz modeli yuklanmadi — {e}")
        if on_error == "stop":
            _log("SYS: Ovoz qulfi ochilmadi, tizim to'xtatildi.")
            return False
        _log("SYS: DIQQAT — Avelia ovoz tasdiqlanmasdan ishga tushdi.")
        return True

    attempt = 0
    while True:
        attempt += 1
        _state("PROCESSING")
        _log(f"SYS: [OVOZ QULFI] Shaxsni tasdiqlash uchun {seconds:.0f} soniya gapiring...")

        try:
            audio = await loop.run_in_executor(
                None, _record, seconds, wosk.SR, device
            )
        except Exception as e:
            # A recording failure is a hardware problem, not a rejection, so it
            # must not be reported to Abror as "I don't recognise you".
            _log(f"ERR: Mikrofondan yozib olinmadi — {e}")
            if on_error == "stop":
                return False
            _log("SYS: DIQQAT — Avelia ovoz tasdiqlanmasdan ishga tushdi.")
            return True

        _log("SYS: Ovoz barmoq izi tekshirilmoqda...")
        try:
            accepted, similarity, reason = await loop.run_in_executor(
                None, wosk.verify_array, audio
            )
        except Exception as e:
            _log(f"ERR: Ovoz tahlilida xato — {e}")
            if on_error == "stop":
                return False
            return True

        if accepted:
            _log(f"SYS: Kirish tasdiqlandi ({similarity:.3f}). Xush kelibsiz, Abror.")
            _state("THINKING")
            return True

        if reason == "silence":
            _log("SYS: Mikrofon jim — hech qanday nutq eshitilmadi.")
        elif reason == "too_short":
            _log("SYS: Nutq juda qisqa — to'liqroq gapiring.")
        else:
            _log(f"SYS: Ovoz mos kelmadi ({similarity:.3f}). Qayta urinilmoqda...")

        if max_attempts and attempt >= max_attempts:
            _log(f"SYS: {max_attempts} urinishdan keyin tasdiqlanmadi.")
            return on_error == "open"

        await asyncio.sleep(retry_delay)


def _record(seconds: float, sample_rate: int, device=None) -> np.ndarray:
    """Blocking 4-second capture. Runs in an executor, never on the loop.

    Called BEFORE Avelia opens its own microphone stream. Two streams on one
    device fail on Windows WASAPI, so this must stay ahead of the Gemini
    connection in run() — which is where it already was.
    """
    import sounddevice as sd

    kwargs = {"samplerate": sample_rate, "channels": 1, "dtype": "float32"}
    if device is not None:
        kwargs["device"] = device

    data = sd.rec(int(seconds * sample_rate), **kwargs)
    sd.wait()
    return np.asarray(data, dtype=np.float32).reshape(-1)