import os
# TIZIMNI 100% MUTLOQ OFLAYN REJIMGA O'TKAZISH
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import librosa
import torch

# =====================================================================
# 🛠️ MONKEY PATCH 1: torchaudio backend muammosini yamash
# =====================================================================
import torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
    torchaudio.get_audio_backend = lambda: "soundfile" 

# =====================================================================
# 🛠️ MONKEY PATCH 2: Hugging Face Windows Symlink va 404 Crashni PROACTIVE yamash
# =====================================================================
import huggingface_hub
orig_hf_hub_download = huggingface_hub.hf_hub_download

def patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
        
    is_custom = False
    if len(args) > 1 and args[1] == 'custom.py': is_custom = True
    if kwargs.get('filename') == 'custom.py': is_custom = True
        
    if is_custom:
        dummy_path = os.path.abspath("custom_bypass.py")
        if not os.path.exists(dummy_path):
            with open(dummy_path, "w") as f:
                f.write("# Wosk AI: Windows Symlink va 404 muammosini aylanib o'tish yamog'i\n")
        return dummy_path

    return orig_hf_hub_download(*args, **kwargs)

huggingface_hub.hf_hub_download = patched_hf_hub_download
# =====================================================================

from speechbrain.pretrained import SpeakerRecognition 

class WoskAI:
    def __init__(self):
        print("🤖 Barqaror ECAPA-TDNN v0.5.15 Ovoz Encoderi yuklanmoqda...")
        # 💡 FIX: Manzil 'wosk/models/...' ga o'zgartirildi
        self.encoder = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="wosk/models/speechbrain_model"
        )
        self.etalon_vector = None
        # 💡 FIX: Manzil 'wosk/models/...' ga o'zgartirildi
        self.profile_path = "wosk/models/abror_profile.npy"
        self.SR = 16000
        
        if os.path.exists(self.profile_path):
            self.etalon_vector = np.load(self.profile_path)
            print("✅ Tayyor 'Abror' etalon ovoz profili tizimga yuklandi.")
        else:
            self.generate_abror_profile()

    def generate_abror_profile(self):
        print("\n🔄 'abror_vois' papkasidagi 40 ta fayldan profil yaratilmoqda...")
        vectors = []
        
        if not os.path.exists("abror_vois"):
            print("❌ Xato: 'abror_vois' papkasi topilmadi!")
            return
            
        audio_files = [os.path.join("abror_vois", f) for f in os.listdir("abror_vois") if f.endswith(".wav")]
        
        for path in audio_files:
            try:
                y, sr = librosa.load(path, sr=self.SR)
                y = y - np.mean(y) # DC Offset tozalash
                waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    embedding = self.encoder.encode_batch(waveform)
                    vectors.append(embedding.squeeze().cpu().numpy())
            except Exception as e:
                continue
                
        if vectors:
            self.etalon_vector = np.mean(vectors, axis=0)
            # 💡 FIX: Yangi ichki papkani yaratish
            os.makedirs("wosk/models", exist_ok=True)
            np.save(self.profile_path, self.etalon_vector)
            print("✅ 'abror_profile.npy' muvaffaqiyatli muhrlandi! Tizim tayyor.\n")

    def check_voice(self, test_audio_path):
        try:
            y, sr = librosa.load(test_audio_path, sr=self.SR)
            if len(y) < self.SR * 3:
                return "Xato: Ovoz kamida 3 soniya bo'lishi kerak!"
            
            y = y - np.mean(y)
            chunk = y[:self.SR * 3]
            
            rms_energy = np.sqrt(np.mean(chunk**2))
            print(f"[INFO] Audio RMS Energiyasi: {rms_energy:.5f}")
            
            if rms_energy < 0.008:
                return "🤖 Wosk AI: Mikrofonda hech qanday nutq aniqlanmadi (Jimjitlik)."

            waveform = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                test_vector = self.encoder.encode_batch(waveform).squeeze().cpu().numpy()

            dot_product = np.dot(self.etalon_vector, test_vector)
            norm_etalon = np.linalg.norm(self.etalon_vector)
            norm_test = np.linalg.norm(test_vector)
            similarity = float(dot_product / (norm_etalon * norm_test))
            
            if similarity >= 0.50:
                confidence = 90.0 + (similarity - 0.50) * 20.0
                confidence = min(100.0, confidence)
                return f"Ha, tizim tasdiqlandi. Bu siz — ABROR! ({confidence:.1f}%) | Sim: {similarity:.4f}"
            else:
                confidence = max(0.0, similarity * 100)
                return f"Kechirasiz, men sizni tanimadim. Ovoz mos kelmadi. | Sim: {similarity:.4f}"
                
        except Exception as e:
            return f"Tahlilda xatolik: {str(e)}"