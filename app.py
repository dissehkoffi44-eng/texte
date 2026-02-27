import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import gc
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import pickle
import os
import tempfile
import shutil

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODAL_MODES = ['ionian', 'major', 'lydian', 'mixolydian', 'dorian', 'aeolian', 'minor', 'phrygian', 'locrian']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in MODAL_MODES]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# --- TABLEAU CAMELOT ÉTENDU POUR TOUS LES MODES (CORRECTION DES APPROXIMATIONS) ---
# Basé sur le tableau fourni, avec mapping exact pour chaque mode et tonic.
# Utilise # pour dièses et b pour bémols (enharmoniques séparés par /).
CAMELOT_TABLE = {
    1: {'aeolian': 'G#/Ab', 'dorian': 'C#/Db', 'phrygian': 'D#/Eb', 'locrian': 'A#/Bb', 'ionian': 'B', 'lydian': 'E', 'mixolydian': 'F#/Gb'},
    2: {'aeolian': 'D#/Eb', 'dorian': 'G#/Ab', 'phrygian': 'A#/Bb', 'locrian': 'F', 'ionian': 'F#/Gb', 'lydian': 'B', 'mixolydian': 'C#/Db'},
    3: {'aeolian': 'Bb/A#', 'dorian': 'Eb/D#', 'phrygian': 'F', 'locrian': 'C', 'ionian': 'Db/C#', 'lydian': 'Gb/F#', 'mixolydian': 'Ab/G#'},
    4: {'aeolian': 'F', 'dorian': 'Bb/A#', 'phrygian': 'C', 'locrian': 'G', 'ionian': 'Ab/G#', 'lydian': 'Db/C#', 'mixolydian': 'Eb/D#'},
    5: {'aeolian': 'C', 'dorian': 'F', 'phrygian': 'G', 'locrian': 'D', 'ionian': 'Eb/D#', 'lydian': 'Ab/G#', 'mixolydian': 'Bb/A#'},
    6: {'aeolian': 'G', 'dorian': 'C', 'phrygian': 'D', 'locrian': 'A', 'ionian': 'Bb/A#', 'lydian': 'Eb/D#', 'mixolydian': 'F'},
    7: {'aeolian': 'D', 'dorian': 'G', 'phrygian': 'A', 'locrian': 'E', 'ionian': 'F', 'lydian': 'Bb/A#', 'mixolydian': 'C'},
    8: {'aeolian': 'A', 'dorian': 'D', 'phrygian': 'E', 'locrian': 'B', 'ionian': 'C', 'lydian': 'F', 'mixolydian': 'G'},
    9: {'aeolian': 'E', 'dorian': 'A', 'phrygian': 'B', 'locrian': 'F#/Gb', 'ionian': 'G', 'lydian': 'C', 'mixolydian': 'D'},
    10: {'aeolian': 'B', 'dorian': 'E', 'phrygian': 'F#/Gb', 'locrian': 'C#/Db', 'ionian': 'D', 'lydian': 'G', 'mixolydian': 'A'},
    11: {'aeolian': 'F#/Gb', 'dorian': 'B', 'phrygian': 'C#/Db', 'locrian': 'G#/Ab', 'ionian': 'A', 'lydian': 'D', 'mixolydian': 'E'},
    12: {'aeolian': 'C#/Db', 'dorian': 'F#/Gb', 'phrygian': 'G#/Ab', 'locrian': 'D#/Eb', 'ionian': 'E', 'lydian': 'A', 'mixolydian': 'B'},
}

# --- PROFILS DE RÉFÉRENCE MODAUX COMPLETS ---
# Chaque modèle psychoacoustique est étendu aux 7 modes grecs.
# Les profils Dorian/Phrygien/Lydien/Mixolydien/Locrien sont dérivés des
# profils Krumhansl/Temperley/Bellman par rotation et ajustement des degrés caractéristiques.
PROFILES = {
    "krumhansl": {
        "ionian":     [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "aeolian":    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "dorian":     [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.74, 4.75, 3.98, 4.02, 3.34, 3.17],
        "phrygian":   [6.33, 5.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "lydian":     [6.35, 2.23, 3.48, 2.33, 5.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "mixolydian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 5.29, 2.88],
        "locrian":    [6.33, 5.68, 3.52, 5.38, 2.60, 3.53, 1.54, 1.75, 3.98, 2.69, 3.34, 3.17],
        # Alias classiques maintenus pour rétrocompatibilité
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    },
    "temperley": {
        "ionian":     [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "aeolian":    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
        "dorian":     [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.5, 4.5, 3.5, 3.5, 1.5, 4.0],
        "phrygian":   [5.0, 4.5, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
        "lydian":     [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 4.5, 4.5, 2.0, 3.5, 1.5, 4.0],
        "mixolydian": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 4.0, 4.0],
        "locrian":    [5.0, 4.5, 3.5, 4.5, 2.0, 4.0, 2.0, 2.0, 3.5, 2.0, 1.5, 4.0],
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
    },
    "bellman": {
        "ionian":     [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "aeolian":    [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4],
        "dorian":     [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.88, 17.2, 13.62, 12.81, 1.89, 2.4],
        "phrygian":   [18.16, 13.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 1.89, 2.4],
        "lydian":     [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 13.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "mixolydian": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 12.89, 12.44],
        "locrian":    [18.16, 13.69, 12.99, 13.34, 1.07, 11.15, 1.38, 2.2, 13.62, 1.27, 1.89, 2.4],
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4],
    }
}

# --- MAPPING MODE GREC → FAMILLE CAMELOT ---
# Utilisé par get_safe_camelot() pour projeter tout mode vers A (mineur) ou B (majeur)
MODAL_TO_CAMELOT_TYPE = {
    'ionian':     'major',
    'lydian':     'major',
    'mixolydian': 'major',
    'major':      'major',
    'aeolian':    'minor',
    'dorian':     'minor',
    'phrygian':   'minor',
    'locrian':    'minor',
    'minor':      'minor',
}

# Liste complète des modes actifs (utilisée dans les boucles)
ALL_MODES = ['ionian', 'aeolian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian']

# Tierce caractéristique par famille de mode (majeure +4 / mineure +3)
MODE_THIRD = {
    'ionian': 4, 'lydian': 4, 'mixolydian': 4, 'major': 4,
    'aeolian': 3, 'dorian': 3, 'phrygian': 3, 'locrian': 3, 'minor': 3,
}

# --- PRÉCOMPUTATION DES PROFILS ROULÉS (tous modes) ---
# 3 modèles × 9 modes × 12 toniques = 324 tableaux précalculés au démarrage
PROFILES_ROLLED = {
    p_name: {
        mode: [np.roll(PROFILES[p_name][mode], i) for i in range(12)]
        for mode in ALL_MODES + ['major', 'minor']  # inclut alias classiques
    }
    for p_name in PROFILES
}

# Alias maintenus pour solve_key_sniper_modal (basé sur MODAL_PROFILES_ROLLED)
MODES_PROFILES = {m: PROFILES["krumhansl"][m] for m in ALL_MODES}
MODAL_PROFILES_ROLLED = {
    m_name: [np.roll(MODES_PROFILES[m_name], i) for i in range(12)]
    for m_name in MODES_PROFILES
}

# Mapping mode → famille Camelot (rétrocompatibilité)
MODAL_CAMELOT_FAMILY = MODAL_TO_CAMELOT_TYPE

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.15); color: #f87171;
        padding: 15px; border-radius: 15px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    .metric-box:hover { border-color: #58a6ff; }
    .sniper-badge {
        background: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.7em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---

def arbitrage_expert_universel(chroma, bass_vec, key_cons, key_dom, cam_map, y_harm=None, sr=None, tuning=0.0):
    """
    Arbitrage Expert Universel v14.0 — "The Bass & Dissonance Guard + Sub-Bass Priority"

    Améliorations vs v13.0 :
      - Intègre l'analyse Sub-Bass (40–100Hz) comme juge de paix ultime.
        Si la sub-basse de la Dominante domine celle de la Consonance de 20%+,
        la Dominante est déclarée gagnante (BASS_DOMINANCE).
      - Intègre le vecteur basse (bass_vec) dans le calcul de force harmonique
        (pondération : Quinte 50% + Tierce 30% + Basse 20%).
      - Filtre de note interdite (Anti-Confusion Quinte) :
        détecte les dissonances caractéristiques pour lever les ambiguïtés
        entre tonalités voisines (ex. D minor vs C major via le Si).
      - Seuil de victoire relevé à 15% (plus sélectif que v12.1 à 10%).

    Couvre toujours les trois familles de voisinage Camelot :
      - Voisins directs   (dist_num=1, même mode)
      - Voisins relatifs  (dist_num=0, modes croisés A↔B)
      - Voisins diagonaux (dist_num=1, modes croisés)

    Paramètres
    ----------
    chroma   : np.ndarray — vecteur chroma global (12 valeurs normalisées).
    bass_vec : np.ndarray — vecteur chroma des basses (12 valeurs normalisées).
    key_cons : str — clé Consonance (ex. "A minor").
    key_dom  : str — clé Dominante (ex. "E minor").
    cam_map  : dict — mapping clé → Camelot (ex. CAMELOT_MAP).
    y_harm   : np.ndarray (optionnel) — signal harmonique brut pour analyse sub-basse.
    sr       : int (optionnel) — sample rate pour l'analyse sub-basse.

    Retourne
    --------
    dict avec :
      - 'key'        : str  — la clé gagnante.
      - 'duel_actif' : bool — True si un duel de voisinage a eu lieu.
      - 'dist_num'   : int  — distance numérique Camelot (0-6).
      - 'dist_mode'  : int  — distance de mode (0=même, 1=croisé).
      - 'type'       : str  — type de décision (ex. 'BASS_DOMINANCE').
    """
    cam_c = get_exact_camelot(key_cons)
    cam_d = get_exact_camelot(key_dom)

    # Garde-fou : clés inconnues du référentiel Camelot → pas d'arbitrage
    if not cam_c or not cam_d:
        return {"key": key_cons, "duel_actif": False, "dist_num": 99, "dist_mode": 99, "type": "NONE"}

    # Extraction des coordonnées Camelot
    val_c, mode_c = int(cam_c[:-1]), cam_c[-1]
    val_d, mode_d = int(cam_d[:-1]), cam_d[-1]

    # Distance numérique circulaire (roue de 12 positions, max 6)
    dist_num = abs(val_c - val_d)
    if dist_num > 6:
        dist_num = 12 - dist_num

    # Distance de mode (0 = même mode A/A ou B/B, 1 = modes croisés A/B)
    dist_mode = 0 if mode_c == mode_d else 1

    # --- ZONE DE DUEL UNIVERSELLE ---
    # Couvre : voisins directs, relatifs (enharmoniques de mode) et diagonaux
    if dist_num <= 1 and dist_mode <= 1:
        idx_c = NOTES_LIST.index(key_cons.split()[0])
        idx_d = NOTES_LIST.index(key_dom.split()[0])

        def get_advanced_strength(idx, mode, chroma_vec, b_vec):
            """Force harmonique avancée : Quinte (50%) + Tierce (30%) + Basse (20%)."""
            quinte = chroma_vec[(idx + 7) % 12]
            tierce = chroma_vec[(idx + 3) % 12] if mode == 'minor' else chroma_vec[(idx + 4) % 12]
            basse  = b_vec[idx]
            return (quinte * 0.5) + (tierce * 0.3) + (basse * 0.2)

        force_c = get_advanced_strength(idx_c, key_cons.split()[1], chroma, bass_vec)
        force_d = get_advanced_strength(idx_d, key_dom.split()[1], chroma, bass_vec)

        # --- FILTRE DE NOTE INTERDITE (Anti-Confusion Quinte) ---
        # Cas D minor (7A) vs C major (8B) :
        # Le Si (B) est une 6te de tension pour D minor mais fondamentale de G major (voisin de C).
        # Si le Si est trop présent (> 15% du spectre), D minor est probablement faux.
        if key_cons == "D minor" and key_dom == "C major":
            si_idx = NOTES_LIST.index("B")
            if chroma[si_idx] > 0.15:
                force_d *= 1.25  # Boost C major (ou G major caché)

        # --- TEST DE DOMINANCE SUB-BASSE (Étape B) ---
        # Si le signal y_harm est disponible, on calcule le vecteur sub-basse (40–100Hz).
        # Si la sub-basse de la Dominante dépasse celle de la Consonance de 20%+,
        # la Dominante est déclarée gagnante directement (BASS_DOMINANCE).
        if y_harm is not None and sr is not None:
            sub_vec = get_sub_bass_priority(y_harm, sr, tuning=tuning)
            if sub_vec[idx_d] > sub_vec[idx_c] * 1.2:
                return {"key": key_dom, "duel_actif": True, "dist_num": dist_num, "dist_mode": dist_mode, "type": "BASS_DOMINANCE"}

        # La dominante gagne si elle est spectralement plus forte de 15% (seuil relevé vs v12)
        winner = key_dom if force_d > force_c * 1.15 else key_cons
        return {"key": winner, "duel_actif": True, "dist_num": dist_num, "dist_mode": dist_mode, "type": "SPECTRAL"}

    # Par défaut : hors zone de voisinage → pas de duel
    return {"key": key_cons, "duel_actif": False, "dist_num": dist_num, "dist_mode": dist_mode, "type": "NONE"}

def seconds_to_mmss(seconds):
    if seconds is None:
        return "??:??"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr, tuning=0.0):
    nyq = 0.5 * sr
    b, a = butter(2, 150/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12, tuning=tuning)
    return np.mean(chroma_bass, axis=1)

def get_sub_bass_priority(y, sr, tuning=0.0):
    """
    Analyse Sub-Bass chirurgicale (40Hz–100Hz) — Étape A.
    Filtre plus serré que get_bass_priority (150Hz) pour isoler
    les fréquences sub-graves qui définissent la tonique réelle.
    """
    nyq = 0.5 * sr
    # Filtre plus serré pour le Sub-Bass (40Hz à 100Hz)
    b, a = butter(2, [40/nyq, 100/nyq], btype='band')
    y_sub = lfilter(b, a, y)
    chroma_sub = librosa.feature.chroma_cqt(y=y_sub, sr=sr, n_chroma=12, tuning=tuning)
    return np.mean(chroma_sub, axis=1)

def detect_harmonic_sections(y, sr, duration, step=6, min_harm_duration=20, harm_threshold=0.3, perc_threshold=0.5, tuning=0.0):
    """
    Détecte les sections harmoniques en ignorant les intros/outros avec seulement kicks ou voix parlée.

    FIX BUG : hop_length fixé à 512 (valeur par défaut de librosa) pour garantir
    une indexation cohérente des tableaux chroma et flatness.
    L'ancienne formule sr//2048 produisait des indices incorrects selon le sample rate.
    """
    hop_length = 512  # FIX : valeur fixe et cohérente avec librosa par défaut

    chroma_full = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, hop_length=hop_length, tuning=tuning)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)

    harmonic_starts = []
    segments = range(0, int(duration) - step, step // 2)

    for start in segments:
        idx_start = int(start * sr) // hop_length
        idx_end   = int((start + step) * sr) // hop_length

        seg_chroma = chroma_full[:, idx_start:idx_end]
        chroma_var = np.var(seg_chroma)

        seg_flat = np.mean(flatness[0, idx_start:idx_end])

        if chroma_var > harm_threshold and seg_flat < perc_threshold:
            harmonic_starts.append(start)

    if not harmonic_starts:
        return 0, duration

    harmonic_starts = np.array(harmonic_starts)
    diffs = np.diff(harmonic_starts)
    breaks = np.where(diffs > step)[0]

    sections = np.split(harmonic_starts, breaks + 1)
    longest_section = max(sections, key=len)

    if len(longest_section) * step < min_harm_duration:
        return 0, duration

    harm_start = longest_section[0]
    harm_end = longest_section[-1] + step

    harm_start = max(harm_start, 5)
    harm_end = min(harm_end, duration - 5)

    return harm_start, harm_end

def detect_cadence_resolution(timeline, final_key):
    """
    Détection des cadences de résolution (ex. : V-I) pour valider la tonique.
    Compatible avec les modes grecs (ex. "A dorian", "D mixolydian").
    """
    parts = final_key.split()
    note = parts[0]
    mode = parts[1] if len(parts) > 1 else 'ionian'
    root_idx = NOTES_LIST.index(note)
    dom_idx = (root_idx + 7) % 12
    subdom_idx = (root_idx + 5) % 12
    # Famille harmonique pour les comparaisons cadentielles
    is_minor_family = MODAL_TO_CAMELOT_TYPE.get(mode, 'major') == 'minor'

    resolution_count = 0
    for i in range(1, len(timeline)):
        prev_note = timeline[i-1]["Note"]
        curr_note = timeline[i]["Note"]

        dom_key = f"{NOTES_LIST[dom_idx]} {mode}"
        if is_minor_family:
            if (prev_note == f"{NOTES_LIST[dom_idx]} ionian" or
                prev_note == f"{NOTES_LIST[dom_idx]} major" or
                prev_note == dom_key) and curr_note == final_key:
                resolution_count += 1 if ('major' in prev_note or 'ionian' in prev_note) else 0.5
        else:
            if prev_note == dom_key and curr_note == final_key:
                resolution_count += 1

        subdom_key = f"{NOTES_LIST[subdom_idx]} {mode}"
        if prev_note == subdom_key and curr_note == final_key:
            resolution_count += 0.5

    last_third = len(timeline) // 3
    last_resolutions = sum(
        1 for j in range(len(timeline) - last_third, len(timeline) - 1)
        if timeline[j]["Note"].startswith(NOTES_LIST[dom_idx]) and timeline[j+1]["Note"] == final_key
    )

    cadence_score = resolution_count + (last_resolutions * 2)
    return cadence_score

def get_exact_camelot(key_str):
    """
    Conversion exacte vers la roue de Camelot pour tous les modes (y compris grecs).
    Utilise le tableau étendu CAMELOT_TABLE pour éviter les approximations major/minor.
    Gère les enharmoniques (ex. G# == Ab) et mappe 'major' → 'ionian', 'minor' → 'aeolian'.
    """
    if not key_str or "Unknown" in key_str:
        return "??"
    parts = key_str.strip().split()
    if len(parts) < 2:
        return "??"
    note = parts[0]
    mode = parts[1].lower()
    if mode == 'major':
        mode = 'ionian'
    elif mode == 'minor':
        mode = 'aeolian'
    # Recherche dans le tableau
    for code in range(1, 13):
        tonic_str = CAMELOT_TABLE[code].get(mode, '')
        if tonic_str:
            options = [opt.strip() for opt in tonic_str.split('/')]
            if note in options:
                if mode in ['aeolian', 'dorian', 'phrygian']:
                    suffix = 'A'
                else:
                    suffix = 'B'
                return f"{code}{suffix}"
    # Fallback si non trouvé (rare)
    return "??"

def solve_key_sniper(chroma_vector, bass_vector):
    """
    Moteur de détection modal complet — 7 Modes Grecs × 3 modèles psychoacoustiques × 12 toniques.

    Remplace l'ancienne boucle Major/Minor par une itération sur ALL_MODES.
    Chaque mode dispose de :
      - Son propre profil de corrélation (Krumhansl/Temperley/Bellman modal)
      - Sa tierce caractéristique (majeure +4 ou mineure +3 selon famille)
      - Pondération basse et quinte communes à tous les modes
    """
    best_overall_score = -1
    best_key = "Unknown"

    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)

    for m_name in ALL_MODES:
        for i in range(12):
            profile_scores = []
            third_idx = (i + MODE_THIRD[m_name]) % 12

            for p_name in PROFILES_ROLLED:
                rolled = PROFILES_ROLLED[p_name][m_name][i]
                score = np.corrcoef(cv, rolled)[0, 1]

                # Boost basse : si la tonique est dominante dans les graves
                if bv[i] > 0.6:
                    score += (bv[i] * 0.2)

                # Validation de la Quinte (stable pour presque tous les modes sauf Locrian)
                fifth_idx = (i + 7) % 12
                if m_name != 'locrian' and cv[fifth_idx] > 0.5:
                    score += 0.1

                # Validation de la Tierce spécifique au mode
                if cv[third_idx] > 0.5:
                    score += 0.1

                profile_scores.append(score)

            avg_score = np.mean(profile_scores)
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_key = f"{NOTES_LIST[i]} {m_name}"

    return {"key": best_key, "score": best_overall_score}


def solve_key_sniper_modal(chroma_vector, bass_vector):
    """
    Moteur de détection modale étendu — 7 Modes Grecs × 12 Toniques.

    Détecte non seulement Major/Minor classiques mais aussi Dorian, Phrygien,
    Lydien, Mixolydien et Locrien avec pondération professionnelle :
      - Corrélation de Pearson sur le profil modal complet
      - Boost basse (poids +0.25 si tonique dominante dans les graves)
      - Bonus Quinte (stabilité harmonique +0.10)

    Retourne : dict avec 'key' (ex. "A dorian") et 'score'.
    """
    best_overall_score = -1
    best_res = {"key": "Unknown", "mode": "major", "score": 0}

    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)

    for m_name, profiles in MODAL_PROFILES_ROLLED.items():
        for i in range(12):
            rolled = profiles[i]
            # Corrélation de Pearson entre spectre audio et profil théorique du mode
            score = np.corrcoef(cv, rolled)[0, 1]

            # --- POIDS PROFESSIONNELS ---
            # 1. Priorité à la Basse (si la basse joue la tonique → boost massif)
            if bv[i] > 0.7:
                score += 0.25

            # 2. Détection de la Quinte (stabilité harmonique)
            fifth_idx = (i + 7) % 12
            if cv[fifth_idx] > 0.6:
                score += 0.10

            if score > best_overall_score:
                best_overall_score = score
                # Mapping vers nom lisible (ionian→major, aeolian→minor, sinon nom grec)
                if m_name == "ionian":
                    mode_label = "major"
                elif m_name == "aeolian":
                    mode_label = "minor"
                else:
                    mode_label = m_name
                best_res = {
                    "key": f"{NOTES_LIST[i]} {mode_label}",
                    "raw_mode": m_name,
                    "score": best_overall_score
                }

    return best_res


def get_safe_camelot(key_str):
    return get_exact_camelot(key_str)


def get_camelot_modal(key_str):
    return get_exact_camelot(key_str)

def get_key_score(key, chroma_vector, bass_vector):
    """
    Calcul du score de confiance pour une clé donnée — supporte tous les modes grecs.
    Utilise PROFILES_ROLLED précomputé pour éviter les np.roll() redondants.
    """
    parts = key.split()
    note = parts[0]
    mode = parts[1] if len(parts) > 1 else 'ionian'
    root_idx = NOTES_LIST.index(note)

    # Normalisation des vecteurs chroma et basse
    cv_norm = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv_norm = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)

    third_idx = (root_idx + MODE_THIRD.get(mode, 4)) % 12
    fifth_idx = (root_idx + 7) % 12

    profile_scores = []
    for p_name in PROFILES_ROLLED:
        # Utilise le mode exact s'il est disponible, sinon fallback sur major/minor
        if mode in PROFILES_ROLLED[p_name]:
            rolled = PROFILES_ROLLED[p_name][mode][root_idx]
        else:
            fallback = MODAL_TO_CAMELOT_TYPE.get(mode, 'major')
            rolled = PROFILES_ROLLED[p_name][fallback][root_idx]

        score = np.corrcoef(cv_norm, rolled)[0, 1]

        if bv_norm[root_idx] > 0.6:
            score += (bv_norm[root_idx] * 0.2)

        if mode != 'locrian' and cv_norm[fifth_idx] > 0.5:
            score += 0.1
        if cv_norm[third_idx] > 0.5:
            score += 0.1

        profile_scores.append(score)

    return np.mean(profile_scores)


def process_audio(audio_file, file_name, progress_placeholder):
    status_text = progress_placeholder.empty()
    progress_bar = progress_placeholder.progress(0)

    def update_prog(value, text):
        progress_bar.progress(value)
        status_text.markdown(f"**{text} | {value}%**")

    # FIX ROBUSTESSE : temp_dir créé avant le try pour garantir le nettoyage en cas d'erreur
    temp_dir = tempfile.mkdtemp()

    try:
        update_prog(10, f"Chargement de {file_name}")
        file_bytes = audio_file.getvalue()
        ext = file_name.split('.')[-1].lower()
        if ext == 'm4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            y = samples / (2**15)
            sr = audio.frame_rate
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
        else:
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)

        update_prog(20, "Détection des sections harmoniques")
        duration = librosa.get_duration(y=y, sr=sr)
        # 1. Estimer le décalage d'accordage (tuning) dès le début sur le signal complet
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        harm_start, harm_end = detect_harmonic_sections(y, sr, duration, tuning=tuning)
        update_prog(30, f"Section harmonique détectée : {seconds_to_mmss(harm_start)} à {seconds_to_mmss(harm_end)}")

        idx_harm_start = int(harm_start * sr)
        idx_harm_end = int(harm_end * sr)
        y_harm = y[idx_harm_start:idx_harm_end]
        duration_harm = harm_end - harm_start

        update_prog(40, "Filtrage des fréquences")
        # 2. Affiner l'estimation du tuning sur la section harmonique (plus précis)
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        y_filt = apply_sniper_filters(y_harm, sr)

        # 3. Utiliser ce décalage pour recaler l'analyse harmonique
        # On passe le paramètre 'tuning' au calcul du chromagramme
        chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
        bass_global = get_bass_priority(y_harm, sr, tuning=tuning)

        update_prog(50, "Analyse du spectre harmonique")
        step, timeline, votes = 6, [], Counter()
        segments = range(0, int(duration_harm) - step, 2)

        for i, start in enumerate(segments):
            idx_start, idx_end = int(start * sr), int((start + step) * sr)
            seg = y_filt[idx_start:idx_end]
            if np.max(np.abs(seg)) < 0.01: continue

            c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
            c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
            b_seg = get_bass_priority(y_harm[idx_start:idx_end], sr, tuning=tuning)

            res = solve_key_sniper(c_avg, b_seg)
            res_modal = solve_key_sniper_modal(c_avg, b_seg)

            # Calculer la notation Camelot pour ce segment précis
            segment_camelot = get_exact_camelot(res['key'])
            weight = 3.0 if start > (duration_harm - 15) else 2.0 if start < 10 else 1.0
            votes[res['key']] += int(res['score'] * 100 * weight)
            timeline.append({
                "Temps": harm_start + start,
                "Note": res['key'],
                "Camelot": segment_camelot,  # Nouvelle colonne Camelot
                "Conf": res['score'],
                "Mode": res_modal.get('key', res['key'])
            })

            p_val = 50 + int((i / len(segments)) * 40)
            update_prog(p_val, "Calcul chirurgical en cours")

        update_prog(90, "Synthèse finale et validation de la tonique")
        most_common = votes.most_common(10)
        total_votes = sum(votes.values())

        res_global = solve_key_sniper(chroma_avg, bass_global)
        final_key = res_global['key']
        final_conf = int(res_global['score'] * 100)

        # --- ANALYSE MODALE GLOBALE ---
        res_modal_global = solve_key_sniper_modal(chroma_avg, bass_global)
        modal_key = res_modal_global.get('key', final_key)
        modal_camelot = get_exact_camelot(modal_key)
        modal_raw_mode = res_modal_global.get('raw_mode', 'ionian')
        modal_conf = int(res_modal_global.get('score', 0) * 100)  # Récupération de la confiance modale
        # NOUVEAU : Calcul de la présence du mode dans la timeline
        modal_count = sum(1 for t in timeline if t["Mode"] == modal_key)
        modal_presence = (modal_count / len(timeline) * 100) if len(timeline) > 0 else 0

        cadence_score = detect_cadence_resolution(timeline, final_key)
        if cadence_score < 2 and len(most_common) > 1:
            alt_keys = [k for k, _ in most_common if k != final_key]
            alt_cadences = {ak: detect_cadence_resolution(timeline, ak) for ak in alt_keys}
            best_alt = max(alt_cadences, key=alt_cadences.get)
            if alt_cadences[best_alt] > cadence_score + 1:
                final_key = best_alt
                final_conf = int(get_key_score(final_key, chroma_avg, bass_global) * 100)

        if timeline and timeline[-1]["Note"] == final_key:
            final_conf = min(final_conf + 5, 99)

        dominant_key = most_common[0][0] if most_common else "Unknown"
        dominant_votes = most_common[0][1] if most_common else 0
        dominant_percentage = (dominant_votes / total_votes * 100) if total_votes > 0 else 0
        dominant_conf = int(get_key_score(dominant_key, chroma_avg, bass_global) * 100) if dominant_key != "Unknown" else 0

        # Étape A : Calcul de la présence de la clé retenue par l'algorithme (Consonance)
        final_vote_count = votes.get(final_key, 0)
        final_percentage = (final_vote_count / total_votes * 100) if total_votes > 0 else 0
        dominant_camelot = get_exact_camelot(dominant_key)

        mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / sum(votes.values())) > 0.25
        target_key = most_common[1][0] if mod_detected else None

        target_conf = min(int(get_key_score(target_key, chroma_avg, bass_global) * 100), 99) if mod_detected else None

        modulation_time = None
        target_percentage = 0
        ends_in_target = False

        if mod_detected and target_key:
            target_times = np.array([t["Temps"] for t in timeline if t["Note"] == target_key])
            if len(target_times) > 3:
                Z = linkage(target_times.reshape(-1, 1), method='single')
                clust = fcluster(Z, t=5, criterion='distance')
                max_cluster_size = max(Counter(clust).values()) * 2
                if max_cluster_size < 10:
                    mod_detected = False
            if mod_detected:
                candidates = [t["Temps"] for t in timeline if t["Note"] == target_key and t["Conf"] >= 0.84]
                if candidates:
                    modulation_time = min(candidates)
                else:
                    sorted_times = sorted(target_times)
                    modulation_time = sorted_times[max(0, len(sorted_times) // 3)]

                total_valid = len(timeline)
                if total_valid > 0:
                    target_count = sum(1 for t in timeline if t["Note"] == target_key)
                    target_percentage = (target_count / total_valid) * 100

                if timeline:
                    last_n = max(5, len(timeline) // 10)
                    last_segments = timeline[-last_n:]
                    last_counter = Counter(s["Note"] for s in last_segments)
                    last_key = last_counter.most_common(1)[0][0]
                    ends_in_target = (last_key == target_key)

        update_prog(100, "Analyse terminée")
        status_text.empty()
        progress_bar.empty()

        # ══════════════════════════════════════════════════════════════════════════
        # --- INDICE DE STABILITÉ HARMONIQUE ---
        # Calcule combien de clés significatives se partagent le top des votes.
        # Si ≥ 3 clés captent chacune > 10% des votes → morceau instable.
        # ══════════════════════════════════════════════════════════════════════════
        top_votes = [v for k, v in most_common if v > total_votes * 0.1]
        stability_index = 100 / len(top_votes) if top_votes else 0
        is_unstable = len(top_votes) >= 3  # Plus de 3 clés significatives

        # ══════════════════════════════════════════════════════════════════════════
        # --- MOTEUR DE DÉCISION SNIPER V12.1 — JUGE DE PAIX HARMONIQUE ---
        # ══════════════════════════════════════════════════════════════════════════

        # --- CALCULS DE FORCE ET TEMPS DYNAMIQUE ---
        total_duration    = duration  # durée totale du fichier audio
        dynamic_threshold = min(total_duration * 0.20, 60)  # 20% du morceau, max 60s

        # --- AVANT (Calcul bridé) ---
        # final_power = min(final_conf, 99) * np.sqrt(max(final_percentage, 0))
        # --- APRÈS (Calcul libre pour le Power Score) ---
        # On garde la valeur brute pour le calcul de puissance (sans plafond)
        raw_final_conf    = final_conf
        raw_dominant_conf = dominant_conf
        # Power Score = Confiance brute × √Présence (sans plafond → arbitrage équitable)
        final_power = raw_final_conf    * np.sqrt(max(final_percentage, 0))
        dom_power   = raw_dominant_conf * np.sqrt(max(dominant_percentage, 0))
        power_ratio = dom_power / final_power if final_power > 0 else 0

        # --- SÉCURITÉ ANTI-ERREUR (CORRIGÉE v6.1 : Pas d'écrasement) ---
        # On identifie les anomalies mais on ne remplace PAS final_key.
        # Le moteur de décision (Priorités ci-dessous) choisit quoi afficher.
        # Cela préserve les deux mesures brutes (Consonance ET Dominante) pour l'arbitrage.
        est_fantome = (final_percentage < 5.0 and dominant_percentage > 20.0)
        domination_statistique = (power_ratio > 1.25)
        # [NEUTRALISÉ v6.1] — L'ancienne bascule écrasait final_key, faussant l'arbitrage.
        # La logique de priorité (FORCE SUPRÊME, etc.) gère déjà ces cas proprement.

        # Pré-calcul de l'arbitrage universel (Bass & Dissonance Guard — v14.0)
        decision_pivot = None
        arb_dist_num   = 99
        arb_dist_mode  = 99
        arb_type       = "NONE"
        if final_conf >= 70 and dominant_conf >= 70:  # Seuil abaissé à 70 → plus de duels activés
            arb_result = arbitrage_expert_universel(
                chroma_avg,
                bass_global,   # Vecteur basse pour pondération harmonique avancée
                final_key,
                dominant_key,
                CAMELOT_MAP,
                y_harm,        # Signal harmonique pour analyse sub-basse (40–100Hz)
                sr,            # Sample rate associé
                tuning=tuning  # Décalage d'accordage pour recaler l'analyse sub-basse
            )
            # Le pivot est actif si et seulement si un duel de voisinage a eu lieu
            if arb_result["duel_actif"]:
                decision_pivot = arb_result["key"]
                arb_dist_num   = arb_result["dist_num"]
                arb_dist_mode  = arb_result["dist_mode"]
                arb_type       = arb_result.get("type", "SPECTRAL")

        # 🎯 PRIORITÉ @ : VERROUILLAGE STATISTIQUE (Consonance == Dominante)
        # Si les deux moteurs sont d'accord avec une confiance et présence décente
        if final_key == dominant_key and final_conf >= 70 and final_percentage >= 30:
            confiance_pure_key = final_key
            avis_expert = "💎 VERROUILLAGE STATISTIQUE"
            color_bandeau = "linear-gradient(135deg, #10b981, #059669)"  # Vert Émeraude

        # ⚡ PRIORITÉ 0 : LA FORCE SUPRÊME (Power Score juge suprême)
        # Déclenché si la dominante écrase la consonance (ratio > 1.25)
        # OU si ce sont des voisins et que la dominante est plus solide (ratio > 1.10)
        elif power_ratio > 1.25 or (decision_pivot and power_ratio > 1.10):
            confiance_pure_key = dominant_key
            avis_expert = f"⚡ FORCE SUPRÊME ({round(dominant_percentage, 1)}%)"
            color_bandeau = "linear-gradient(135deg, #7c3aed, #4c1d95)"  # Violet Puissance

        # ⚖️ PRIORITÉ 1 : ARBITRAGE HARMONIQUE (Duel de voisinage — affichage systématique)
        # Déclenché dès qu'un duel spectral a eu lieu sur le Camelot Wheel.
        # Le type de voisinage détermine le libellé affiché dans le bandeau.
        elif decision_pivot is not None:
            confiance_pure_key = decision_pivot
            if arb_type == "BASS_DOMINANCE":
                type_duel = "SUB-BASS DOMINANCE"
            elif arb_dist_num == 0 and arb_dist_mode == 1:
                type_duel = "VOISIN RELATIF"
            elif arb_dist_num == 1 and arb_dist_mode == 1:
                type_duel = "VOISIN DIAGONAL"
            else:
                type_duel = "VOISIN PROCHE"
            avis_expert   = f"⚖️ ARBITRAGE : {type_duel}"
            color_bandeau = "linear-gradient(135deg, #0369a1, #0c4a6e)"  # Bleu Océan

        # 🏁 PRIORITÉ 2 : MODULATION DYNAMIQUE (Proportionnelle)
        elif (mod_detected and ends_in_target and target_percentage >= 25.0
              and modulation_time is not None and modulation_time <= dynamic_threshold):
            confiance_pure_key = target_key
            avis_expert = f"🏁 MODULATION VALIDÉE ({round(modulation_time)}s / {round(total_duration)}s)"
            color_bandeau = "linear-gradient(135deg, #4338ca, #1e1b4b)"  # Violet

        # 💎 PRIORITÉ 3 : ACCORD PARFAIT (Consonance = Dominante, confiance ≥ 85%)
        elif final_key == dominant_key and final_conf >= 85:
            confiance_pure_key = final_key
            avis_expert = "💎 ACCORD PARFAIT"
            color_bandeau = "linear-gradient(135deg, #059669, #064e3b)"  # Vert Émeraude

        # ✅ FALLBACK : ANALYSE STABLE (Avec Test de Légitimité Power Score)
        else:
            # On compare les forces brutes (Confiance × √Présence)
            # Si la Dominante est 15% plus puissante, elle détrône la Consonance
            if dom_power > (final_power * 1.2):
                confiance_pure_key = dominant_key
                avis_expert = f"✅ STABILITÉ DOMINANTE ({round(dominant_percentage, 1)}%)"
                color_bandeau = "linear-gradient(135deg, #065f46, #064e3b)"  # Vert
            else:
                confiance_pure_key = final_key
                avis_expert = "✅ ANALYSE STABLE"
                color_bandeau = "linear-gradient(135deg, #065f46, #064e3b)"  # Vert

        # ══════════════════════════════════════════════════════════════════════════

        res_obj = {
            "key": final_key, "camelot": get_exact_camelot(final_key),
            "conf": min(int(raw_final_conf), 100),  # Plafond uniquement pour l'esthétique
            "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
            "chroma": chroma_avg, "modulation": mod_detected,
            "target_key": target_key, "target_camelot": get_exact_camelot(target_key) if target_key else None,
            "name": file_name,
            "modulation_time_str": seconds_to_mmss(modulation_time) if mod_detected else None,
            "mod_target_percentage": round(target_percentage, 1) if mod_detected else 0,
            "mod_ends_in_target": ends_in_target if mod_detected else False,
            "harm_start": seconds_to_mmss(harm_start), "harm_end": seconds_to_mmss(harm_end),
            "target_conf": target_conf,
            "dominant_key": dominant_key, "dominant_camelot": dominant_camelot,
            "dominant_conf": dominant_conf, "dominant_percentage": round(dominant_percentage, 1),
            "key_presence": round(final_percentage, 1),
            "duration_detected": round(total_duration, 1),
            "final_power": round(final_power, 1),
            "dom_power": round(dom_power, 1),
            "power_ratio": round(power_ratio, 2),
            "confiance_pure": confiance_pure_key,
            "pure_camelot": get_exact_camelot(confiance_pure_key),
            "avis_expert": avis_expert,
            "color_bandeau": color_bandeau,
            # --- DONNÉES MODALES ---
            "modal_key": modal_key,
            "modal_camelot": modal_camelot,
            "modal_raw_mode": modal_raw_mode,
            "modal_conf": modal_conf,
            "modal_presence": round(modal_presence, 1),
            # --- INDICE DE STABILITÉ ---
            "stability_score": round(stability_index, 1),
            "is_unstable": is_unstable,
        }

        # --- RAPPORT TELEGRAM ENRICHI (RADAR + TIMELINE) ---
        if TELEGRAM_TOKEN and CHAT_ID:
            try:
                mod_line = ""
                if mod_detected:
                    perc = res_obj["mod_target_percentage"]
                    end_txt = " 🏁 *FIN SUR MODULATION*" if res_obj['mod_ends_in_target'] else ""
                    mod_line = (
                        f"\n⚠️ *MODULATION →* `{target_key.upper()} ({res_obj['target_camelot']})`"
                        f" | ≈ *{res_obj['modulation_time_str']}*"
                        f" | *PRÉSENCE:* `{perc}%`"
                        f" | *CONFIANCE:* `{res_obj['target_conf']}%`"
                        f"{end_txt}"
                    )

                dom_line = (
                    f"\n🏆 *DOMINANTE:* `{dominant_key.upper()} ({res_obj['dominant_camelot']})`"
                    f" | *PRÉSENCE:* `{res_obj['dominant_percentage']}%`"
                    f" | *CONFIANCE:* `{res_obj['dominant_conf']}%`"
                )
                pure_line = f"\n🔒 *TONALITÉ PURE:* `{res_obj['confiance_pure'].upper()} ({res_obj['pure_camelot']})` | *AVIS:* `{res_obj['avis_expert']}`"
                # Mapping des couleurs vers Emojis pour Telegram
                modal_emojis = {
                    "ionian": "🟢 (Vert)",
                    "aeolian": "🔵 (Bleu)",
                    "dorian": "🟣 (Violet)",
                    "phrygian": "🔴 (Rouge)",
                    "lydian": "🟠 (Orange)",
                    "mixolydian": "💎 (Cyan)",
                    "locrian": "⚫ (Gris)"
                }
                # Récupération de l'émoji correspondant au mode actuel
                current_mode_emoji = modal_emojis.get(res_obj.get('modal_raw_mode'), "⚪")
                modal_line = (
                    f"\n🎼 *MODE DÉTECTÉ :* {current_mode_emoji}"
                    f"\n└ `{res_obj.get('modal_key','—').upper()} ({res_obj.get('modal_camelot','??')})`"
                )

                caption = (
                    f"🎯 *RCDJ228 MUSIC SNIPER*\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"📂 *FICHIER:* `{file_name}`\n"
                    f"🎹 *MEILLEURE CONSONANCE:* `{final_key.upper()} ({res_obj['camelot']})`"
                    f" | *PRÉSENCE:* `{res_obj['key_presence']}%`"
                    f" | *CONFIANCE:* `{res_obj['conf']}%`"
                    + dom_line
                    + pure_line
                    + modal_line
                    + f"{mod_line}\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"🎸 *ACCORDAGE:* `{res_obj['tuning']} Hz` ✅\n"
                    f"🛡️ *SECTION HARMONIQUE:* {res_obj['harm_start']} → {res_obj['harm_end']}"
                )

                # Radar avec labels Camelot (mode majeur par défaut)
                CAMELOT_LABELS_TG = [get_exact_camelot(f"{n} major") for n in NOTES_LIST]
                fig_radar = go.Figure(data=go.Scatterpolar(r=res_obj['chroma'], theta=CAMELOT_LABELS_TG, fill='toself', line_color='#10b981'))
                fig_radar.update_layout(template="plotly_dark", title="SPECTRE HARMONIQUE (Camelot)", polar=dict(radialaxis=dict(visible=False)))
                radar_bytes = fig_radar.to_image(format="png", width=700, height=500)

                # Timeline avec axe Y en notation Camelot
                CAMELOT_ORDER_TG = [f"{i}{m}" for i in range(1, 13) for m in ['A', 'B']]
                df_tl = pd.DataFrame(res_obj['timeline'])
                fig_tl = px.line(
                    df_tl,
                    x="Temps",
                    y="Camelot",
                    markers=True,
                    template="plotly_dark",
                    category_orders={"Camelot": CAMELOT_ORDER_TG},
                    hover_data={"Note": True, "Temps": ":.2f"},
                    title="ÉVOLUTION TEMPORELLE (Camelot)"
                )
                timeline_bytes = fig_tl.to_image(format="png", width=1000, height=450)

                media_group = [
                    {'type': 'photo', 'media': 'attach://radar.png', 'caption': caption, 'parse_mode': 'Markdown'},
                    {'type': 'photo', 'media': 'attach://timeline.png'}
                ]

                files = {
                    'radar.png': radar_bytes,
                    'timeline.png': timeline_bytes
                }

                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup",
                    data={'chat_id': CHAT_ID, 'media': json.dumps(media_group)},
                    files=files
                )

            except Exception as e:
                st.error(f"Erreur d'envoi Telegram : {e}")

        # Sauvegarde disque pour données lourdes
        timeline_path = os.path.join(temp_dir, f"{file_name}_timeline.pkl")
        chroma_path = os.path.join(temp_dir, f"{file_name}_chroma.npy")
        with open(timeline_path, 'wb') as tf:
            pickle.dump(res_obj['timeline'], tf)
        np.save(chroma_path, res_obj['chroma'])

        res_obj['timeline_path'] = timeline_path
        res_obj['chroma_path'] = chroma_path
        res_obj['temp_dir'] = temp_dir
        del res_obj['timeline']
        del res_obj['chroma']

        del y, y_filt
        gc.collect()
        return res_obj

    except Exception as e:
        # FIX ROBUSTESSE : nettoyage garanti du dossier temp même en cas de crash
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e


def get_chord_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 2.0);
        }});
    }};
    """

# --- DASHBOARD PRINCIPAL ---
st.title("🎯 RCDJ228 MUSIC SNIPER")
st.markdown("#### Système d'Analyse Harmonique Modale — 7 Modes Grecs × 12 Toniques")

global_status = st.empty()

uploaded_files = st.file_uploader("📥 Déposez vos fichiers (Audio)", type=['mp3','wav','flac','m4a'], accept_multiple_files=True, key="file_uploader")

if 'analyses' not in st.session_state:
    st.session_state.analyses = {}
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

if uploaded_files:
    global_status.info("Analyse des fichiers en cours...")
    progress_zone = st.container()

    for i, f in enumerate(reversed(uploaded_files)):
        file_name = f.name

        if file_name not in st.session_state.analyses:
            st.session_state.analyzing = True
            analysis_data = process_audio(f, file_name, progress_zone)
            st.session_state.analyses[file_name] = analysis_data
            if len(st.session_state.analyses) > 5:
                oldest_file = next(iter(st.session_state.analyses))
                if 'temp_dir' in st.session_state.analyses[oldest_file] and os.path.exists(st.session_state.analyses[oldest_file]['temp_dir']):
                    shutil.rmtree(st.session_state.analyses[oldest_file]['temp_dir'])
                del st.session_state.analyses[oldest_file]

        if file_name in st.session_state.analyses:
            analysis_data = st.session_state.analyses[file_name]

            with open(analysis_data['timeline_path'], 'rb') as tf:
                timeline = pickle.load(tf)
            chroma = np.load(analysis_data['chroma_path'])

            with st.container():
                st.markdown(f"<div class='file-header'>📂 ANALYSE : {analysis_data['name']}</div>", unsafe_allow_html=True)

                mod_alert = ""
                if analysis_data['modulation']:
                    ends_badge = " &nbsp; 🏁 <b>FIN SUR MODULATION</b>" if analysis_data['mod_ends_in_target'] else ""
                    mod_alert = (
                        f"<div class='modulation-alert'>"
                        f"⚠️ MODULATION : <b>{analysis_data['target_key'].upper()}</b> ({analysis_data['target_camelot']})"
                        f" &nbsp;|&nbsp; PRÉSENCE : <b>{analysis_data['mod_target_percentage']}%</b>"
                        f" &nbsp;|&nbsp; CONFIANCE : <b>{analysis_data['target_conf']}%</b>"
                        f"{ends_badge}"
                        f"</div>"
                    )

                st.markdown(f"""
                    <div class="report-card" style="background:{analysis_data['color_bandeau']};">
                        <p style="letter-spacing:5px; opacity:0.8; font-size:0.7em; margin-bottom:0px;">
                            SNIPER ENGINE v6.1 — MODAL | {analysis_data['avis_expert']}
                        </p>
                        <h1 style="font-size:5em; margin:0px 0; font-weight:900; line-height:1; text-align: center;">
                            {analysis_data['pure_camelot']}
                        </h1>
                        <p style="font-size:2em; font-weight:bold; margin-top:-10px; margin-bottom:20px; opacity:0.9; text-align: center;">
                            {analysis_data['confiance_pure'].upper()}
                        </p>
                        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.2); width:50%; margin: 20px auto;">
                        <div style="display: flex; justify-content: space-around; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; opacity: 0.85; flex-wrap: wrap; gap: 8px;">
                            <div>🎯 CONSONANCE :&nbsp;<b>{analysis_data['key'].upper()} ({analysis_data['camelot']})</b>&nbsp;({analysis_data.get('key_presence', 0)}%&nbsp;|&nbsp;{analysis_data['conf']}%)</div>
                            <div>📊 DOMINANTE :&nbsp;<b>{analysis_data['dominant_key'].upper()}</b>&nbsp;({analysis_data['dominant_camelot']}&nbsp;|&nbsp;{analysis_data['dominant_percentage']}%&nbsp;|&nbsp;{analysis_data['dominant_conf']}%)</div>
                        </div>
                        {mod_alert}
                    </div>
                """, unsafe_allow_html=True)

                if analysis_data.get('is_unstable'):
                    stability_val = analysis_data.get('stability_score', 0)
                    st.markdown(
                        f"<div style='background:rgba(245,158,11,0.12); border:1px solid #f59e0b; border-radius:15px; "
                        f"padding:14px 20px; margin-bottom:12px; font-family:JetBrains Mono,monospace; color:#fbbf24;'>"
                        f"⚠️ <b>ALERTE INSTABILITÉ</b> — Indice de stabilité : <b>{stability_val}</b> &nbsp;|&nbsp; "
                        f"Ce morceau change fréquemment de structure harmonique. Évitez les transitions longues."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                m2, m3 = st.columns(2)
                with m2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{analysis_data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                with m3:
                    btn_id = f"play_{hash(analysis_data['name'])}"
                    components.html(f"""
                        <button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🎹 TESTER L'ACCORD</button>
                        <script>{get_chord_js(btn_id, analysis_data['key'])}</script>
                    """, height=110)

                # --- MODE GREC DÉTECTÉ ---
                raw_mode = analysis_data.get('modal_raw_mode', 'ionian')
                modal_colors = {
                    "ionian": "#10b981", "aeolian": "#3b82f6",
                    "dorian": "#8b5cf6", "phrygian": "#ef4444",
                    "lydian": "#f59e0b", "mixolydian": "#06b6d4", "locrian": "#6b7280"
                }
                modal_descriptions = {
                    "ionian":     "Majeur classique — lumineux, stable",
                    "aeolian":    "Mineur naturel — mélancolique, profond",
                    "dorian":     "Mineur jazz — funky, sophistiqué",
                    "phrygian":   "Flamenco / Metal — sombre, exotique",
                    "lydian":     "Cinématique — rêveur, flottant",
                    "mixolydian": "Blues / Rock — énergique, dominant",
                    "locrian":    "Dissonant — instable, avant-gardiste",
                }
                mc = modal_colors.get(raw_mode, "#6b7280")
                md = modal_descriptions.get(raw_mode, "")
                st.markdown(
                    f"<div class='metric-box' style='border-color:{mc}; margin-bottom:12px;'>"
                    f"<b>🎼 MODE DÉTECTÉ</b><br>"
                    f"<span style='font-size:1.8em; color:{mc}; font-weight:900;'>{analysis_data.get('modal_key','—').upper()}</span>"
                    f"&nbsp;&nbsp;<span style='font-size:1em; color:#94a3b8;'>Camelot : <b>{analysis_data.get('modal_camelot','??')}</b></span><br>"
                    f"<div style='font-size:0.9em; margin-top:5px; color:#94a3b8;'>"
                    f"🎯 CONFIANCE : <b>{analysis_data.get('modal_conf', 0)}%</b> &nbsp;|&nbsp; 📊 PRÉSENCE : <b>{analysis_data.get('modal_presence', 0)}%</b>"
                    f"</div>"
                    f"<span style='font-size:0.75em; color:#94a3b8; font-style:italic;'>{md}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # --- POWER SCORES (debug & transparence) ---
                ps1, ps2, ps3 = st.columns(3)
                with ps1:
                    st.markdown(f"<div class='metric-box'><b>💪 FORCE CONSONANCE</b><br><span style='font-size:1.6em; color:#a78bfa;'>{analysis_data.get('final_power', '—')}</span></div>", unsafe_allow_html=True)
                with ps2:
                    st.markdown(f"<div class='metric-box'><b>💪 FORCE DOMINANTE</b><br><span style='font-size:1.6em; color:#a78bfa;'>{analysis_data.get('dom_power', '—')}</span></div>", unsafe_allow_html=True)
                with ps3:
                    ratio_val = analysis_data.get('power_ratio', 0)
                    ratio_color = "#ef4444" if ratio_val > 1.25 else "#f59e0b" if ratio_val > 1.10 else "#10b981"
                    st.markdown(f"<div class='metric-box'><b>📊 RATIO DE PUISSANCE</b><br><span style='font-size:1.6em; color:{ratio_color};'>{ratio_val}</span></div>", unsafe_allow_html=True)

                c1, c2 = st.columns([2, 1])
                with c1:
                    # Préparer l'ordre des catégories pour que l'axe Y soit bien rangé (1A, 1B, 2A, 2B...)
                    CAMELOT_ORDER = [f"{i}{m}" for i in range(1, 13) for m in ['A', 'B']]
                    df_tl = pd.DataFrame(timeline)
                    fig_tl = px.line(
                        df_tl,
                        x="Temps",
                        y="Camelot",  # Utiliser la colonne Camelot
                        markers=True,
                        template="plotly_dark",
                        category_orders={"Camelot": CAMELOT_ORDER},  # Trier par roue de Camelot
                        hover_data={"Note": True, "Temps": True}  # Garder le nom de la note au survol
                    )
                    fig_tl.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Temps : %{x:.2f}s<br>Camelot : %{y}<extra></extra>")
                    fig_tl.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_tl, use_container_width=True, key=f"timeline_{analysis_data['name']}_{i}")
                with c2:
                    # Créer une liste de labels Camelot pour le radar (basée sur le mode majeur par défaut)
                    CAMELOT_LABELS = [get_exact_camelot(f"{n} major") for n in NOTES_LIST]
                    fig_radar = go.Figure(data=go.Scatterpolar(r=chroma, theta=CAMELOT_LABELS, fill='toself', line_color='#10b981'))
                    fig_radar.update_layout(template="plotly_dark", height=300, margin=dict(l=40, r=40, t=30, b=20), polar=dict(radialaxis=dict(visible=False)), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{analysis_data['name']}_{i}")

                st.markdown("<hr style='border-color: #30363d; margin-bottom:40px;'>", unsafe_allow_html=True)

            del timeline, chroma
            gc.collect()

    st.session_state.analyzing = False
    global_status.success("Tous les fichiers ont été analysés avec succès !")
    gc.collect()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Sniper Control")
    if st.button("🧹 Vider la file d'analyse"):
        for data in list(st.session_state.analyses.values()):
            if 'temp_dir' in data and os.path.exists(data['temp_dir']):
                shutil.rmtree(data['temp_dir'])
        st.session_state.analyses = {}
        st.session_state.analyzing = False
        gc.collect()
        st.rerun()
