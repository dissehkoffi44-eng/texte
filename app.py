import streamlit as st
import numpy as np
import pandas as pd
import librosa
from music21 import *
import io

st.set_page_config(page_title="Détecteur de Tonalité Avancé", page_icon="🎵", layout="wide")
st.title("🎵 Détecteur de Tonalité Avancé")
st.markdown("Analyse par grille d'accords (par sections) + analyse audio avec librosa")

# ===================== PROFILS DE CLÉS (Krumhansl-Schmuckler) =====================
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def pitch_class_histogram(chords):
    hist = np.zeros(12)
    for ch in chords:
        try:
            c = chord.Chord(ch)
            for p in c.pitches:
                pc = p.pitchClass
                hist[pc] += 1
        except:
            continue
    return hist / (hist.sum() + 1e-8)

def correlation_score(hist, profile):
    return np.corrcoef(hist, profile)[0, 1]

def detect_key_with_confidence(chords):
    if not chords:
        return None, 0.0
    hist = pitch_class_histogram(chords)
    scores = []
    for i in range(12):
        shifted_major = np.roll(MAJOR_PROFILE, i)
        shifted_minor = np.roll(MINOR_PROFILE, i)
        score_maj = correlation_score(hist, shifted_major)
        score_min = correlation_score(hist, shifted_minor)
        scores.append((NOTES[i], 'major', score_maj))
        scores.append((NOTES[i], 'minor', score_min))
    best = max(scores, key=lambda x: x[2])
    mode_fr = "Majeur" if best[1] == "major" else "Mineur"
    return f"{best[0]} {mode_fr}", round(best[2], 3)

def parse_chords(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    all_chords = []
    sections = []
    current_section = []
    for line in lines:
        if line.startswith("#") or line == "":  # ligne vide = nouvelle section
            if current_section:
                sections.append(current_section)
                current_section = []
            continue
        for sep in ["|", ",", "-", "/"]:
            line = line.replace(sep, " ")
        chords = [c.strip() for c in line.split() if c.strip()]
        current_section.extend(chords)
        all_chords.extend(chords)
    if current_section:
        sections.append(current_section)
    return all_chords, sections

# ===================== INTERFACE =====================
tab1, tab2, tab3 = st.tabs(["📝 Grille manuelle", "📁 Upload fichier accords", "🎤 Upload audio"])

with tab1:
    st.subheader("Saisie manuelle de grille d'accords")
    example = "C G Am F\n\nG D Em Bm\n\nC G F C"
    text = st.text_area("Collez votre grille (séparez les sections par une ligne vide)", example, height=200)
    if st.button("Analyser (manuelle)", type="primary"):
        all_chords, sections = parse_chords(text)
        if all_chords:
            global_key, conf = detect_key_with_confidence(all_chords)
            st.success(f"**Tonalité globale : {global_key}** (confiance : {conf})")
            if len(sections) > 1:
                st.write("**Détection par sections :**")
                for i, sec in enumerate(sections, 1):
                    key, c = detect_key_with_confidence(sec)
                    st.write(f"Section {i} ({len(sec)} accords) → **{key}** (confiance {c})")

with tab2:
    st.subheader("Upload fichier .txt ou .csv")
    uploaded_file = st.file_uploader("Choisissez un fichier .txt ou .csv", type=["txt", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            text = df.to_string(index=False)
        else:
            text = uploaded_file.read().decode("utf-8")
        st.text_area("Contenu détecté", text[:500] + ("..." if len(text)>500 else ""), height=150)
        if st.button("Analyser le fichier"):
            all_chords, sections = parse_chords(text)
            global_key, conf = detect_key_with_confidence(all_chords)
            st.success(f"**Tonalité globale : {global_key}** (confiance : {conf})")
            if len(sections) > 1:
                for i, sec in enumerate(sections, 1):
                    key, c = detect_key_with_confidence(sec)
                    st.write(f"Section {i} → **{key}** (confiance {c})")

with tab3:
    st.subheader("Upload fichier audio (MP3 / WAV)")
    audio_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav", "ogg"])
    duration_limit = st.slider("Durée max analysée (secondes)", 30, 180, 90)
    
    if audio_file and st.button("Analyser l'audio"):
        with st.spinner("Analyse audio en cours... (peut prendre 10-30 secondes)"):
            try:
                y, sr = librosa.load(io.BytesIO(audio_file.read()), duration=duration_limit, sr=22050)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=24)
                hist = np.mean(chroma, axis=1)
                hist = hist / (hist.sum() + 1e-8)
                
                scores = []
                for i in range(12):
                    shifted_maj = np.roll(MAJOR_PROFILE, i)
                    shifted_min = np.roll(MINOR_PROFILE, i)
                    scores.append((NOTES[i], 'major', np.corrcoef(hist, shifted_maj)[0,1]))
                    scores.append((NOTES[i], 'minor', np.corrcoef(hist, shifted_min)[0,1]))
                
                best = max(scores, key=lambda x: x[2])
                mode_fr = "Majeur" if best[1] == "major" else "Mineur"
                st.success(f"**Tonalité détectée par audio : {best[0]} {mode_fr}** (confiance : {round(best[2], 3)})")
                st.metric("Durée analysée", f"{len(y)/sr:.1f} secondes")
                
            except Exception as e:
                st.error(f"Erreur lors de l'analyse audio : {str(e)}")

st.caption("App développée avec music21 + librosa • Profils Krumhansl-Schmuckler • Détection par sections activée")
