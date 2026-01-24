import streamlit as st
from skey import detect_key
import torch
import torchaudio
import os
import tempfile

st.title("Détecteur de Tonalité Musicale (Clé) avec Haute Précision")
st.write("Téléchargez un fichier audio (.mp3 ou .wav) pour déterminer sa tonalité principale en utilisant un modèle SOTA auto-supervisé.")

# Upload du fichier audio
uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav"])

if uploaded_file is not None:
    # Sauvegarde temporaire du fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    # Bouton pour lancer l'analyse
    if st.button("Analyser la tonalité"):
        with st.spinner("Analyse en cours... (cela peut prendre quelques secondes)"):
            try:
                # Détection avec skey (sur CPU par défaut)
                result = detect_key(audio_dir=audio_path, extension=os.path.splitext(audio_path)[1][1:], device="cpu")
                
                # Affichage du résultat
                st.success(f"Tonalité détectée : **{result}**")
                st.info("Note : Ce modèle est robuste mais peut varier si la chanson a des modulations. Pour plus de précision, analysez des segments spécifiques.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {str(e)}. Assurez-vous que le fichier est valide.")

    # Nettoyage du fichier temp
    os.unlink(audio_path)
else:
    st.warning("Veuillez uploader un fichier audio pour commencer.")

# Pied de page
st.markdown("Basé sur [skey de Deezer](https://github.com/deezer/skey) - Modèle auto-supervisé égalant les SOTA supervisés.")
