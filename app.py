import streamlit as st
import numpy as np
import pandas as pd
import librosa
from music21 import *
import io
from scipy.signal import find_peaks

# ────────────────────────────────────────────────
#  TESTS D'IMPORTS – placés TOUT EN HAUT
# ────────────────────────────────────────────────
try:
    from music21 import chord, stream, chord  # double chord pour forcer l'import
    st.success("music21 importé avec succès !")
except Exception as e:
    st.error(f"Erreur music21 : {e}")
    st.stop()   # ← arrête tout de suite pour voir l'erreur

try:
    import librosa
    st.success("librosa OK")
except Exception as e:
    st.error(f"Erreur librosa : {e}")
    st.stop()

# Si on arrive ici → les deux bibliothèques principales sont importées

st.set_page_config(page_title="Détecteur de Tonalité Ultra-Précis", page_icon="🎵", layout="wide")

st.title("🎵 Détecteur de Tonalité Maximisé (visant 92–96 %)")
st.markdown("**Améliorations clés** : music21 + ensemble Krumhansl + cadence boosting + HPSS audio + top-3")

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def chord_to_root_and_type(chord_str):
    try:
        c = chord.Chord(chord_str)
        return c.root().pitchClass, c.quality
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
    
    # 1. Analyse music21
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
    weights = np.linspace(0.5, 1.5, len(chords_list))
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
    
    # 3. Cadence boosting simple
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
    
    # Fusion
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

def analyze_audio_advanced(y, sr, duration_limit=120):
    y = y[:int(duration_limit * sr)]
    y_harmonic, _ = librosa.effects.hpss(y)
    y_harmonic = librosa.util.normalize(y_harmonic)
    y_harmonic = y_harmonic[np.abs(y_harmonic) > 0.02]
    
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

# ────────────────────────────────────────────────
#                 INTERFACE
# ────────────────────────────────────────────────

tab_audio, tab_chords = st.tabs(["Analyse Audio", "Analyse Accords (bientôt)"])

with tab_audio:
    st.markdown("Charge un fichier audio (mp3, wav, ogg, flac...)")
    
    audio_file = st.file_uploader(
        "Sélectionne ton fichier audio",
        type=["mp3", "wav", "ogg", "flac", "m4a"],
        help="Durée recommandée : < 2 minutes pour des résultats rapides",
        key="audio_upload"
    )
    
    if audio_file is not None:
        try:
            # Lecture sécurisée via BytesIO
            audio_bytes = audio_file.read()
            audio_io = io.BytesIO(audio_bytes)
            
            with st.spinner("Analyse en cours..."):
                y, sr = librosa.load(audio_io, sr=None)
                st.success(f"Audio chargé – durée : {len(y)/sr:.1f} secondes")
                
                key, conf, top3 = analyze_audio_advanced(y, sr)
                
                st.subheader(f"Résultat principal : **{key}**")
                st.write(f"Confiance : **{conf:.3f}**")
                
                st.markdown("**Top 3 propositions :**")
                for note, mode, score in top3:
                    m = "Majeur" if mode == "major" else "Mineur"
                    st.write(f"- {note} {m} → {score:.3f}")
                    
        except Exception as e:
            st.error(f"Erreur pendant le traitement audio :\n{str(e)}")
    else:
        st.info("En attente du fichier audio...")

with tab_chords:
    st.info("Fonctionnalité analyse accords / MIDI en cours de développement...")
    st.write("(Tu peux déjà tester la partie audio ci-dessus)")

if st.button("Rafraîchir la page (debug)"):
    st.rerun()
