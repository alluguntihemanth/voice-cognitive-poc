import librosa
import numpy as np

def extract_audio_features(path, sr=16000):
    y, sr = librosa.load(path, sr=sr)
    
    # Basic stats
    duration = librosa.get_duration(y=y, sr=sr)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Tempo (proxy for speech rate)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return {
        "duration": duration,
        "zcr": zcr,
        "rms": rms,
        "pitch": pitch,
        "tempo": tempo
    }
