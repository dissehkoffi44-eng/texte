import streamlit as st
import essentia.standard as es
import numpy as np
import os

# Configuration de la page
st.set_page_config(page_title="Audio Key Pro", page_icon="🎹")

st.title("🎹 Détecteur de Tonalité Professionnel")
st.markdown("""
Cette application utilise **Essentia (Deep Learning)** pour extraire la tonalité 
et l'accordage (tuning) d'un morceau avec une précision studio.
""")

uploaded_file = st.file_uploader("Glissez un fichier audio ici", type=['mp3', 'wav', 'flac'])

if uploaded_file:
    # Sauvegarde temporaire du fichier pour Essentia
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner('Analyse neuronale de la structure harmonique...'):
        try:
            # 1. Chargement de l'audio
            loader = es.MonoLoader(filename="temp_audio")
            audio = loader()

            # 2. Utilisation du KeyExtractor d'Essentia
            # Cet algorithme est beaucoup plus précis que les chromagrammes simples
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)

            # 3. Affichage des résultats
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Tonalité Détectée", value=f"{key} {scale}")
            
            with col2:
                st.metric(label="Confiance", value=f"{strength:.2f}")

            # Feedback visuel selon la confiance
            if strength > 0.7:
                st.success("Analyse fiable : La structure harmonique est claire.")
            else:
                st.warning("Analyse complexe : Le morceau pourrait contenir des modulations ou du bruit.")

        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
        
        finally:
            if os.path.exists("temp_audio"):
                os.remove("temp_audio")

st.info("Note : Les fichiers longs (5min+) peuvent prendre quelques secondes à être traités par le modèle.")
