import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Key Master Pro - Temperley Weighted", layout="wide")

# Système Camelot
CAMELOT_MAP = {
    "C Major": "8B", "C Minor": "5A", "C# Major": "3B", "C# Minor": "12A",
    "Db Major": "3B", "D Major": "10B", "D Minor": "7A", "D# Major": "5B",
    "Eb Major": "5B", "Eb Minor": "2A", "E Major": "12B", "E Minor": "9A",
    "F Major": "7B", "F Minor": "4A", "F# Major": "2B", "F# Minor": "11A",
    "Gb Major": "2B", "G Major": "9B", "G Minor": "6A", "G# Major": "4B",
    "Ab Major": "4B", "Ab Minor": "1A", "A Major": "11B", "A Minor": "8A",
    "Bb Major": "6B", "Bb Minor": "3A", "B Major": "1B", "B Minor": "10A"
}

# --- PROFILS ---
KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Temperley (Le modèle privilégié ici)
TEMP_MAJ = [0.75, 0.06, 0.49, 0.08, 0.67, 0.46, 0.10, 0.72, 0.10, 0.37, 0.06, 0.40]
TEMP_MIN = [0.71, 0.08, 0.48, 0.62, 0.05, 0.46, 0.11, 0.75, 0.40, 0.07, 0.13, 0.33]

BELL_MAJ = [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.76, 1.89, 12.88, 1.24, 14.74]
BELL_MIN = [18.16, 1.29, 11.34, 14.67, 0.99, 11.31, 1.20, 17.27, 11.47, 1.02, 0.86, 11.35]

def analyze_hybrid_weighted(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
    chroma_vals = np.mean(chroma, axis=1)
    chroma_vals = (chroma_vals - np.mean(chroma_vals)) / np.std(chroma_vals)
    
    notes_en = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    notes_fr = ['Do', 'Do#', 'Ré', 'Ré#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
    
    results = []

    # COEFFICIENTS DE PONDÉRATION
    W_TEMP = 0.50  # 50% de poids pour Temperley
    W_KS = 0.25    # 25% pour Krumhansl
    W_BELL = 0.25  # 25% pour Bellman

    for i in range(12):
        # Calcul des scores individuels
        s_ks_maj = np.corrcoef(chroma_vals, np.roll(KS_MAJ, i))[0, 1]
        s_ks_min = np.corrcoef(chroma_vals, np.roll(KS_MIN, i))[0, 1]
        
        s_temp_maj = np.corrcoef(chroma_vals, np.roll(TEMP_MAJ, i))[0, 1]
        s_temp_min = np.corrcoef(chroma_vals, np.roll(TEMP_MIN, i))[0, 1]
        
        s_bell_maj = np.corrcoef(chroma_vals, np.roll(BELL_MAJ, i))[0, 1]
        s_bell_min = np.corrcoef(chroma_vals, np.roll(BELL_MIN, i))[0, 1]
        
        # Moyenne pondérée
        final_maj = (s_temp_maj * W_TEMP) + (s_ks_maj * W_KS) + (s_bell_maj * W_BELL)
        final_min = (s_temp_min * W_TEMP) + (s_ks_min * W_KS) + (s_bell_min * W_BELL)
        
        results.append((final_maj, f"{notes_fr[i]} Majeur", f"{notes_en[i]} Major"))
        results.append((final_min, f"{notes_fr[i]} Mineur", f"{notes_en[i]} Minor"))

    results.sort(key=lambda x: x[0], reverse=True)
    return results, chroma_vals, notes_fr, tuning

# --- INTERFACE ---
st.title("🎼 AI Key Master Pro")
st.subheader("Moteur Hybride Optimisé (Pondération Temperley 50%)")

file = st.file_uploader("Upload (FLAC, MP3, WAV)", type=["flac", "mp3", "wav"])

if file:
    ext = file.name.split('.')[-1]
    tmp_path = f"temp_audio.{ext}"
    with open(tmp_path, "wb") as f:
        f.write(file.getbuffer())
    
    st.audio(file)
    
    try:
        with st.spinner("Analyse avec priorité Temperley..."):
            res, chromas, note_labels, tune = analyze_hybrid_weighted(tmp_path)
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Tonalité Détectée", res[0][1])
            with c2:
                camelot = CAMELOT_MAP.get(res[0][2], "??")
                st.metric("Code Camelot", camelot)
            with c3:
                hz = 440 * (2**(tune/12))
                st.metric("Diapason", f"{hz:.1f} Hz")
                
            st.divider()
            l, r = st.columns([2, 1])
            with l:
                st.subheader("📊 Spectre Harmonique")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(note_labels, chromas, color='#FF4B4B')
                ax.set_facecolor('#0E1117')
                fig.patch.set_facecolor('#0E1117')
                ax.tick_params(colors='white')
                st.pyplot(fig)
            with r:
                st.subheader("🎯 Confiance Temperley+")
                for i in range(5):
                    conf = (res[i][0] + 1) / 2 * 100
                    st.write(f"**{res[i][1]}**")
                    st.progress(int(conf))

    except Exception as e:
        st.error(f"Erreur : {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
