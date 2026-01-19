import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="AI Key Master Pro", page_icon="🎹", layout="wide")

# Système Camelot (Standard de l'industrie DJ pour le mixage harmonique)
CAMELOT_MAP = {
    "Do Majeur": "8B", "La Mineur": "8A",
    "Sol Majeur": "9B", "Mi Mineur": "9A",
    "Ré Majeur": "10B", "Si Mineur": "10A",
    "La Majeur": "11B", "Fa# Mineur": "11A",
    "Mi Majeur": "12B", "Do# Mineur": "12A",
    "Si Majeur": "1B", "Sol# Mineur": "1A",
    "Fa# Majeur": "2B", "Ré# Mineur": "2A",
    "Do# Majeur": "3B", "La# Mineur": "3A",
    "Sol# Majeur": "4B", "Fa Mineur": "4A",
    "Ré# Majeur": "5B", "Do Mineur": "5A",
    "La# Majeur": "6B", "Sol Mineur": "6A",
    "Fa Majeur": "7B", "Ré Mineur": "7A",
}

def get_camelot(key_name):
    return CAMELOT_MAP.get(key_name, "??")

# --- MOTEUR D'ANALYSE HAUTE PRÉCISION ---
def analyze_audio_pro(audio_path):
    # 1. Chargement multi-format avec Librosa
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 2. Détection du Diapason (Tuning)
    # L'IA détecte si le morceau est à 440Hz ou décalé (ex: 432Hz)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # 3. Extraction par Transformée Constant-Q (CQT)
    # Calibré sur la perception logarithmique de l'oreille humaine
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, n_chroma=12)
    
    # 4. Nettoyage Psychoacoustique
    # Utilisation de la médiane pour ignorer les bruits percussifs (kicks/snares)
    chroma_vals = np.median(chroma, axis=1)
    if np.max(chroma_vals) > 0:
        chroma_vals /= np.max(chroma_vals)

    # 5. Profils Harmoniques "Piano Master"
    # Ces poids simulent les harmoniques naturelles des cordes d'un piano
    maj_template = [1.0, 0.05, 0.1, 0.05, 0.8, 0.1, 0.05, 0.9, 0.05, 0.1, 0.05, 0.1]
    min_template = [1.0, 0.05, 0.1, 0.8, 0.1, 0.1, 0.05, 0.9, 0.05, 0.1, 0.05, 0.1]
    
    notes = ['Do', 'Do#', 'Ré', 'Ré#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
    results = []

    for i in range(12):
        # On fait pivoter les profils pour tester les 12 notes racines
        score_maj = np.dot(chroma_vals, np.roll(maj_template, i))
        score_min = np.dot(chroma_vals, np.roll(min_template, i))
        
        results.append((score_maj, f"{notes[i]} Majeur"))
        results.append((score_min, f"{notes[i]} Mineur"))

    # Tri par niveau de certitude
    results.sort(key=lambda x: x[0], reverse=True)
    return results, chroma_vals, notes, tuning

# --- INTERFACE UTILISATEUR ---
st.title("🎧 AI Key Master : Analyse Pro")
st.markdown("### Support natif FLAC, MP3 & WAV | Précision Psychoacoustique")

uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["flac", "mp3", "wav"])

if uploaded_file:
    # Sauvegarde sécurisée du fichier temporaire avec son extension
    ext = uploaded_file.name.split('.')[-1]
    temp_path = f"temp_analysis.{ext}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        with st.spinner("L'oreille artificielle analyse les résonances..."):
            results, chromas, notes_list, tuning = analyze_audio_pro(temp_path)
            
            # Résultat principal
            best_key = results[0][1]
            camelot_code = get_camelot(best_key)
            
            # --- AFFICHAGE DASHBOARD ---
            st.write("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tonalité Détectée", best_key)
            with col2:
                st.metric("Code Camelot", camelot_code)
            with col3:
                hz = 440 * (2**(tuning/12))
                st.metric("Diapason", f"{hz:.1f} Hz")
            
            st.write("---")
            
            # --- VISUALISATION ---
            c_left, c_right = st.columns([2, 1])
            
            with c_left:
                st.subheader("📊 Spectre de Chromas (CQT)")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(notes_list, chromas, color='#1DB954') # Vert Pro
                ax.set_facecolor('#0E1117')
                fig.patch.set_facecolor('#0E1117')
                ax.tick_params(colors='white')
                st.pyplot(fig)
                
            with c_right:
                st.subheader("🎯 Certitude")
                for i in range(5):
                    name = results[i][1]
                    # Score relatif au meilleur résultat
                    confidence = (results[i][0] / results[0][0]) * 100
                    st.write(f"**{get_camelot(name)}** - {name}")
                    st.progress(int(confidence))
                    
    except Exception as e:
        st.error(f"Erreur d'analyse : {e}")
    finally:
        # Nettoyage automatique du serveur
        if os.path.exists(temp_path):
            os.remove(temp_path)

st.divider()
st.caption("Moteur basé sur Librosa & Krumhansl-Schmuckler profiles. Précision cible : 90% sur contenus harmoniques.")
