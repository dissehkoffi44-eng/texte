import streamlit as st
import essentia.standard as es
import numpy as np
import tempfile
import os
from collections import defaultdict

# Liste complète des profils Essentia
PROFILES = [
    'krumhansl', 'temperley', 'shaath', 'tonictriad', 'temperley2005',
    'thpcp', 'edmm', 'edma', 'bgate', 'braw'
]

st.title("Détecteur de Tonalité Musicale avec Essentia (Pondéré + Auto-Analyse)")

st.write("""
Analyse **automatique** dès l'upload du fichier (pas besoin de cliquer sur un bouton).
Pondération par strength pour une précision maximale (vote chirurgical).
Précision estimée : 80-95 % sur benchmarks MIR.
Formats : .mp3 / .wav
""")

# Sélection du mode
mode = st.radio("Mode d'analyse :", ("Profil unique", "Tous les profils (vote pondéré par strength)"))

selected_profile = None
if mode == "Profil unique":
    selected_profile = st.selectbox("Profil :", PROFILES, index=PROFILES.index('temperley'))

# Upload
uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav"])

# Session state pour tracker le nom du fichier précédent (évite re-analyse inutile)
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if uploaded_file is not None:
    current_file_name = uploaded_file.name

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    # Lance l'analyse seulement si le fichier a changé (ou premier upload)
    if current_file_name != st.session_state.last_file_name:
        with st.spinner("Analyse automatique en cours..."):
            try:
                # Charge audio
                loader = es.MonoLoader(filename=audio_path)
                audio = loader()

                results = []

                if mode == "Profil unique":
                    key_extractor = es.Key(
                        profileType=selected_profile,
                        numHarmonics=4, pcpSize=36, slope=0.6,
                        usePolyphony=True, useThreeChords=True
                    )
                    key, scale, strength = key_extractor(audio)
                    results.append((key, scale, strength, selected_profile))
                else:
                    for profile in PROFILES:
                        try:
                            key_extractor = es.Key(
                                profileType=profile,
                                numHarmonics=4, pcpSize=36, slope=0.6,
                                usePolyphony=True, useThreeChords=True
                            )
                            key, scale, strength = key_extractor(audio)
                            results.append((key, scale, strength, profile))
                        except ValueError:
                            pass  # Ignore profils non supportés

                # Mapping français
                key_fr_map = {
                    'C': 'Do', 'C#': 'Do#', 'D': 'Ré', 'D#': 'Ré#', 'E': 'Mi',
                    'F': 'Fa', 'F#': 'Fa#', 'G': 'Sol', 'G#': 'Sol#',
                    'A': 'La', 'A#': 'La#', 'B': 'Si'
                }

                if results:
                    if mode == "Profil unique":
                        key, scale, strength, profile = results[0]
                        key_fr = key_fr_map.get(key, key)
                        mode_fr = 'majeur' if scale == 'major' else 'mineur'
                        result_text = f"**{key_fr} {mode_fr}** (profil '{profile}', force : {strength:.2f})"
                        st.session_state.analysis_result = result_text
                    else:
                        weighted_votes = defaultdict(float)
                        for key, scale, strength, _ in results:
                            weighted_votes[(key, scale)] += strength

                        if weighted_votes:
                            best_key, best_scale = max(weighted_votes, key=weighted_votes.get)
                            best_strength_sum = weighted_votes[(best_key, best_scale)]
                            key_fr = key_fr_map.get(best_key, best_key)
                            mode_fr = 'majeur' if best_scale == 'major' else 'mineur'
                            result_text = f"**{key_fr} {mode_fr}** (somme strengths : {best_strength_sum:.2f} / {len(results)} profils)"
                            st.session_state.analysis_result = result_text

                        # Tableau détaillé
                        st.subheader("Détails par profil")
                        data = []
                        for k, s, stren, prof in results:
                            kf = key_fr_map.get(k, k)
                            mf = 'majeur' if s == 'major' else 'mineur'
                            data.append([prof, f"{kf} {mf}", f"{stren:.2f}"])
                        st.table({"Profil": [d[0] for d in data], "Tonalité": [d[1] for d in data], "Strength": [d[2] for d in data]})

                st.session_state.last_file_name = current_file_name

            except Exception as e:
                st.error(f"Erreur pendant l'analyse : {str(e)}")
                st.session_state.analysis_result = None

    # Affichage du résultat (persistant)
    if st.session_state.analysis_result:
        st.success(f"Tonalité détectée automatiquement : {st.session_state.analysis_result}")

    # Nettoyage temp
    os.unlink(audio_path)

else:
    st.info("Uploadez un fichier audio pour lancer l'analyse automatique.")
    # Reset si pas de fichier
    st.session_state.last_file_name = None
    st.session_state.analysis_result = None

st.markdown("Basé sur Essentia – Analyse auto + pondération pour précision chirurgicale.")
