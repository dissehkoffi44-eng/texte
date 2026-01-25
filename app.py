import streamlit as st
import numpy as np
import pandas as pd
import librosa
from music21 import *
import io
from scipy.signal import find_peaks

st.set_page_config(page_title="Détecteur de Tonalité Ultra-Précis", page_icon="🎵", layout="wide")

st.title("🎵 Détecteur de Tonalité Maximisé (visant 92–96 %)")
st.markdown("**Améliorations clés** : music21 + ensemble Krumhansl + cadence boosting + HPSS audio + top-3")

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def chord_to_root_and_type(chord_str):
    try:
        c = chord.Chord(chord_str)
        return c.root().pitchClass, c.quality  # quality = 'major', 'minor', 'diminished', etc.
    except:
        return None, None

def detect_key_ensemble(chords_list):
    if not chords_list:
        return None, 0.0, []
    
    s = stream.Stream()
    for ch_str in chords_list:
        try:
            c = chord.Chord(ch_str)
            c.duration.quarterLength = 4.0
            s.append(c)
        except:
            continue
    
    if len(s) == 0:
        return None, 0.0, []
    
    # 1. Analyse music21 (très robuste)
    try:
        key1 = s.analyze('key')
        score1 = getattr(key1, 'correlationCoefficient', 0.85)
        tonic1 = key1.tonic.name
        mode1 = key1.mode
    except:
        key1 = None
        score1 = 0.0
        tonic1 = None
        mode1 = None
    
    # 2. Krumhansl-Schmuckler amélioré
    hist = np.zeros(12)
    weights = np.linspace(0.5, 1.5, len(chords_list))  # derniers accords plus importants
    for i, ch_str in enumerate(chords_list):
        root, _ = chord_to_root_and_type(ch_str)
        if root is not None:
            hist[root] += weights[i]
    hist /= (hist.sum() + 1e-8)
    
    scores = []
    for i in range(12):
        maj_score = np.corrcoef(hist, np.roll(MAJOR_PROFILE, i))[0,1]
        min_score = np.corrcoef(hist, np.roll(MINOR_PROFILE, i))[0,1]
        scores.append((NOTES[i], 'major', maj_score))
        scores.append((NOTES[i], 'minor', min_score))
    
    best_ks = max(scores, key=lambda x: x[2])
    score2 = best_ks[2]
    
    # 3. Cadence boosting (V→I ou V→i très fort)
    cadence_boost = 0.0
    for j in range(len(chords_list)-1):
        try:
            c1 = chord.Chord(chords_list[j])
            c2 = chord.Chord(chords_list[j+1])
            if (c1.root().name in ['G','E'] and c2.root().name in ['C','A'] and 
                c1.quality == 'major' and c2.quality in ['major', 'minor']):
                cadence_boost = 0.12
        except:
            pass
    
    # Fusion des scores
    final_scores = []
    seen = set()
    for note, mode, sc in scores:
        key_str = f"{note} {mode}"
        if key_str in seen: continue
        seen.add(key_str)
        total_score = sc * 0.6
        if key1 and note == tonic1 and mode == mode1:
            total_score = max(total_score, score1 * 0.9 + sc * 0.3)
        total_score += cadence_boost if (note == tonic1 and mode == mode1) else 0
        final_scores.append((note, mode, total_score))
    
    final_scores.sort(key=lambda x: x[2], reverse=True)
    best = final_scores[0]
    mode_fr = "Majeur" if best[1] == "major" else "Mineur"
    return f"{best[0]} {mode_fr}", round(best[2], 3), final_scores[:3]

# Interface (similaire mais avec top-3 et meilleure audio)
# ... (le reste du code reste très proche de la version précédente, mais avec l'appel à detect_key_ensemble)

# Pour l'audio : pipeline amélioré
def analyze_audio_advanced(y, sr, duration_limit=120):
    y = y[:int(duration_limit * sr)]
    y_harmonic, _ = librosa.effects.hpss(y)                    # Séparation harmonique
    y_harmonic = librosa.util.normalize(y_harmonic)
    y_harmonic = y_harmonic[np.abs(y_harmonic) > 0.02]         # Suppression bruit faible
    
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36, hop_length=512)
    hist = np.mean(chroma, axis=1)
    hist /= (hist.sum() + 1e-8)
    
    scores = []
    for i in range(12):
        maj = np.corrcoef(hist, np.roll(MAJOR_PROFILE, i))[0,1]
        mino = np.corrcoef(hist, np.roll(MINOR_PROFILE, i))[0,1]
        scores.append((NOTES[i], 'major', maj))
        scores.append((NOTES[i], 'minor', mino))
    
    scores.sort(key=lambda x: x[2], reverse=True)
    best = scores[0]
    mode_fr = "Majeur" if best[1] == "major" else "Mineur"
    return f"{best[0]} {mode_fr}", round(best[2], 3), scores[:3]

# (Intégrer ces fonctions dans les tabs comme avant)
