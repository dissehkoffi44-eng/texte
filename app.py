import streamlit as st
import numpy as np
import pandas as pd
import librosa
from music21 import *
import io
from scipy.signal import find_peaks

# ────────────────────────────────────────────────
#  TESTS D'IMPORTS
# ────────────────────────────────────────────────
try:
    from music21 import chord, stream
    st.success("music21 importé avec succès !")
except Exception as e:
    st.error(f"Erreur music21 : {e}")
    st.stop()

try:
    import librosa
    st.success("librosa OK")
except Exception as e:
    st.error(f"Erreur librosa : {e}")
    st.stop()

st.set_page_config(page_title="Détecteur de Tonalité Ultra-Précis", page_icon="🎵", layout="wide")

st.title("🎵 Détecteur de Tonalité Maximisé (v2)")
st.markdown("**Améliorations appliquées** : HPSS léger + percussions • chroma_cqt optimisé • pondération temporelle • bins_per_octave=12")

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Optionnel : profil Albrecht (plus moderne) - décommentez si vous voulez tester
# MAJOR_ALBRECHT = np.array([7.38, 2.15, 3.82, 2.48, 5.12, 4.21, 2.71, 6.15, 2.55, 4.02, 2.41, 3.33])
# MINOR_ALBRECHT  = np.array([7.45, 2.62, 3.68, 5.55, 2.88, 3.71, 2.62, 5.48, 4.12, 2.95, 3.51, 3.41])

def chord_to_root_and_type(chord_str):
    try:
        c = chord.Chord(chord_str)
        return c.root().pitchClass, c.quality
    except:
        return None, None

def detect_key_ensemble(chords_list):
    # (inchangé pour l'instant)
    if not chords_list:
        return None, 0.0, []
    # ... (votre fonction existante)
    return "Fonctionnalité en cours", 0.0, []

def analyze_audio_advanced(y, sr, duration_limit=150):
    y = y[:int(duration_limit * sr)]
    
    # HPSS léger + réinjection des percussions (recommandé)
    y_harmonic, y_perc = librosa.effects.hpss(y, margin=(1.5, 8.0))
    y_harmonic = y_harmonic + 0.12 * y_perc                     # garde un peu de rythme
    y_harmonic = librosa.util.normalize(y_harmonic)

    # Chroma CQT optimisé
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=4096,
        bins_per_octave=12,      # ← beaucoup plus stable
        n_bins=84,               # 7 octaves
        norm=2,
        tuning=None
    )

    # Pondération temporelle : les dernières secondes comptent plus
    n_frames = chroma.shape[1]
    if n_frames > 1:
        decay = np.exp(np.linspace(0, -2.5, n_frames))   # décroissance douce
        decay /= decay.sum()
        hist = np.dot(chroma, decay)
    else:
        hist = np.mean(chroma, axis=1)

    hist /= (hist.sum() + 1e-10)

    # Calcul des scores
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
    st.markdown("Charge un fichier audio (mp3, wav, ogg, flac, m4a)")
    
    audio_file = st.file_uploader(
        "Sélectionne ton fichier audio",
        type=["mp3", "wav", "ogg", "flac", "m4a"],
        help="Durée recommandée : < 2 minutes",
        key="audio_upload"
    )
    
    if audio_file is not None:
        try:
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

if st.button("Rafraîchir la page (debug)"):
    st.rerun()
