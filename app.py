import streamlit as st
from music21 import *
import numpy as np

st.set_page_config(page_title="Détecteur de Tonalité", page_icon="🎵", layout="centered")
st.title("🎵 Détecteur Automatique de Tonalité")
st.markdown("Collez votre grille d'accords → l'app détecte la tonalité en appliquant les règles classiques")

# Exemple par défaut
example = "C G Am F\nC G F C\nDm G C Am\nF G C"

chords_text = st.text_area(
    "Grille d'accords (une ligne par mesure ou tout d'un coup)",
    example,
    height=150
)

if st.button("🔍 Analyser la tonalité", type="primary"):
    if not chords_text.strip():
        st.error("Veuillez entrer des accords")
        st.stop()

    # Nettoyage et parsing
    lines = [line.strip() for line in chords_text.split("\n") if line.strip()]
    all_chords = []
    
    for line in lines:
        # Gérer les séparateurs courants
        for sep in ["|", ",", "-", "/"]:
            line = line.replace(sep, " ")
        chords = [c.strip() for c in line.split() if c.strip()]
        all_chords.extend(chords)

    if not all_chords:
        st.error("Aucun accord valide détecté")
        st.stop()

    # Création du stream music21
    s = stream.Stream()
    for ch in all_chords:
        try:
            # Ajouter des durées pour une meilleure analyse
            c = chord.Chord(ch)
            c.duration.quarterLength = 4.0  # blanche
            s.append(c)
        except Exception:
            st.warning(f"Accord ignoré : {ch}")

    if len(s) == 0:
        st.error("Impossible de créer des accords valides")
        st.stop()

    # Analyse avec l'algorithme de music21 (très puissant)
    try:
        key_result = s.analyze('key')
        tonic = key_result.tonic.name
        mode = key_result.mode  # 'major' ou 'minor'
        
        # Conversion en français
        mode_fr = "Majeur" if mode == "major" else "Mineur"
        tonalite = f"{tonic} {mode_fr}"

        st.success(f"✅ **Tonalité détectée : {tonalite}**")
        
        # Informations supplémentaires
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accords analysés", len(all_chords))
        with col2:
            st.metric("Dernier accord", all_chords[-1] if all_chords else "N/A")
        
        st.info(f"Music21 a utilisé un algorithme basé sur : profil des accords, cadence parfaite, fréquence de la tonique, sensible, etc.")

        # Afficher la grille d'accords en notation romaine (optionnel)
        if st.checkbox("Voir les degrés en notation romaine"):
            try:
                rn = roman.romanNumeralFromChord(s[0], key_result)
                st.write("Exemple premier accord :", rn.figure)
            except:
                st.write("Impossible d'afficher les degrés")

    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {str(e)}")

st.caption("App basée sur music21 • Détection globale (pas section par section)")
