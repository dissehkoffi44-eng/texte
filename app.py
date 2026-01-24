import streamlit as st
import librosa
import numpy as np
import tempfile
import os

st.title("Détecteur de Tonalité (version simple Librosa)")

uploaded_file = st.file_uploader("Audio (.mp3/.wav)", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    if st.button("Analyser"):
        with st.spinner("..."):
            try:
                y, sr = librosa.load(path, sr=None)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                
                # Profils simples Krumhansl (majeur/mineur)
                major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
                minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
                
                major_corr = np.corrcoef(chroma_mean, major_profile)[0,1]
                minor_corr = np.corrcoef(chroma_mean, minor_profile)[0,1]
                
                keys = ['Do', 'Do#', 'Ré', 'Ré#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
                major_key = keys[np.argmax(major_corr)]
                minor_key = keys[np.argmax(minor_corr)]
                
                if major_corr > minor_corr:
                    result = f"{major_key} majeur (corr: {major_corr:.2f})"
                else:
                    result = f"{minor_key} mineur (corr: {minor_corr:.2f})"
                
                st.success(f"Tonalité : **{result}**")
            except Exception as e:
                st.error(f"Erreur : {e}")
    os.unlink(path)
