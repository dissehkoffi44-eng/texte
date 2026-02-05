import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
import requests
import gc
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import logging
import textwrap
import concurrent.futures  # AJOUT : Pour parallélisation des segments

# --- FORCE FFMPEG PATH (WINDOWS FIX) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODES = ['major', 'minor', 'mixolydian', 'dorian', 'phrygian', 'lydian', 'locrian', 'aeolian', 'ionian']  # Ajout des modes étendus
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in MODES]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
    # Note: Pour modes étendus, mapper approximativement (e.g., mixolydian comme major)
}

# Offsets pour calculer la root de la major parente
MODE_PARENT_OFFSETS = {
    'major': 0,
    'ionian': 0,
    'mixolydian': -7,
    'lydian': -5,
    'minor': -9,
    'aeolian': -9,
    'dorian': -2,
    'phrygian': -4,
    'locrian': -11  # 7e degré
}

# Mise à jour du mapping pour tous les modes (standards et étendus)
for note in NOTES_LIST:
    for mode in MODES:
        # Pour major/minor/ionian/aeolian, utiliser le mapping direct
        if mode in ['ionian', 'aeolian']:
            base_mode = 'major' if mode == 'ionian' else 'minor'
            CAMELOT_MAP[f"{note} {mode}"] = CAMELOT_MAP.get(f"{note} {base_mode}", "??")
        elif mode in ['major', 'minor']:
            continue  # Déjà mappés

        # Pour modes étendus, calculer la root de la major parente
        root_idx = NOTES_LIST.index(note)
        offset = MODE_PARENT_OFFSETS.get(mode, 0)
        parent_root_idx = (root_idx + offset) % 12
        parent_root = NOTES_LIST[parent_root_idx]

        # Détermine si major-like ou minor-like pour choisir B ou A
        if mode in ['major', 'ionian', 'mixolydian', 'lydian']:
            parent_mode = 'major'  # Camelot B
        else:  # minor, aeolian, dorian, phrygian, locrian
            parent_mode = 'minor'  # Camelot A (relative minor de la major parente)
            # Relative minor = -3 demi-tons de la major parente
            relative_minor_idx = (parent_root_idx - 3) % 12
            parent_root = NOTES_LIST[relative_minor_idx]

        camelot_key = f"{parent_root} {parent_mode}"
        camelot = CAMELOT_MAP.get(camelot_key, "??")
        CAMELOT_MAP[f"{note} {mode}"] = camelot + " (modal equiv)" if camelot != "??" else "??"

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "mixolydian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.00],  # Pénalise 7e majeure, boost b7
        "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 4.75, 2.54, 3.98, 2.69, 3.34, 3.17],  # Boost 6 majeure vs minor
        "phrygian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 2.00],  # Pénalise 2 majeure
        "lydian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 5.19, 2.52, 2.39, 3.66, 2.29, 2.88],  # Boost #4
        "locrian": [6.33, 2.68, 3.52, 5.38, 2.60, 2.54, 4.75, 3.53, 3.98, 2.69, 3.34, 3.17],  # Pénalise 5 parfaite
        "aeolian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],  # Identique à minor
        "ionian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]   # Identique à major
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
        "mixolydian": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 3.0],  # Adapté pour b7
        "dorian": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 4.5, 2.0, 3.5, 2.0, 1.5, 4.0],  # Boost 6
        "phrygian": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 3.0],  # Adapté
        "lydian": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 4.5, 2.0, 2.0, 3.5, 1.5, 4.0],  # Boost #4
        "locrian": [5.0, 2.0, 3.5, 4.5, 2.0, 2.0, 4.5, 4.0, 3.5, 2.0, 1.5, 4.0],  # Adapté
        "aeolian": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],  # Identique à minor
        "ionian": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]   # Identique à major
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4],
        "mixolydian": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 10.0],  # Adapté
        "dorian": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 17.2, 1.38, 13.62, 1.27, 12.79, 2.4],  # Boost 6
        "phrygian": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 1.0],  # Adapté
        "lydian": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 16.74, 1.25, 1.56, 12.81, 1.89, 12.44],  # Boost #4
        "locrian": [18.16, 0.69, 12.99, 13.34, 1.07, 1.38, 17.2, 11.15, 13.62, 1.27, 12.79, 2.4],  # Adapté
        "aeolian": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4],  # Identique à minor
        "ionian": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44]   # Identique à major
    }
}

# --- FONCTIONS UTILITAIRES POUR LES CONSEILS MIX ---

def get_neighbor_camelot(camelot_str: str, offset: int) -> str:
    """Retourne le Camelot voisin avec l'offset donné (modulo 12)"""
    if camelot_str in ['??', None, '']:
        return '??'
    try:
        num = int(camelot_str[:-1])
        wheel = camelot_str[-1]  # A ou B
        new_num = ((num - 1 + offset) % 12) + 1
        return f"{new_num}{wheel}"
    except:
        return '??'

def get_compatible_keys(key, threshold=0.85):
    """Retourne liste de keys compatibles basées sur overlap notes."""
    diat_notes = get_diatonic_notes(key)  # Set de 7 notes
    compat = []
    for other_key in [f"{n} {m}" for n in NOTES_LIST for m in MODES]:
        if other_key == key: continue
        other_notes = get_diatonic_notes(other_key)
        overlap = len(diat_notes.intersection(other_notes)) / 7 if diat_notes else 0
        if overlap >= threshold:
            camelot = CAMELOT_MAP.get(other_key, "??")
            compat.append(f"{other_key} ({camelot}) - Overlap: {overlap:.0%}")
    return compat[:5]  # Top 5

def get_mixing_advice(data):
    """
    Génère les conseils de mix suivant EXACTEMENT la checklist fournie
    """
    if not data.get('modulation', False):
        return None

    principal_camelot = data.get('camelot', '??')
    target_key       = data.get('target_key', 'Inconnu')
    target_camelot   = data.get('target_camelot', '??')
    perc             = data.get('mod_target_percentage', 0)
    ends_in_target   = data.get('mod_ends_in_target', False)
    time_str         = data.get('modulation_time_str', '??:??')

    lines = []

    lines.append("**Checklist mix harmonique – ce que tu dois faire :**")

    if ends_in_target:
        lines.append(f"✅ **Oui : le morceau termine dans {target_key.upper()} ({target_camelot})**")
        lines.append("   → **Privilégie cette tonalité pour le track suivant**")
        priority = "target"
    else:
        lines.append(f"⚠️ **Non : ne termine pas en {target_key.upper()} ({target_camelot})**")
        lines.append("   → La tonalité de sortie reste plutôt " + principal_camelot)
        priority = "principal"

    if perc > 45:
        lines.append(f"✅ **Pourcentage très élevé ({perc:.1f}%)** → traite ce track presque comme s'il était en **{target_camelot}**")
        priority = "target"
    elif perc > 25:
        lines.append(f"ℹ️ **Pourcentage significatif ({perc:.1f}%)** → la target est importante")
        lines.append("   → Tu peux sortir après la bascule pour utiliser la target")
    else:
        lines.append(f"🔸 **Pourcentage faible ({perc:.1f}%)** → modulation plutôt ponctuelle")
        lines.append("   → Tu peux rester sur la tonalité principale pour plus de sécurité")

    lines.append(f"⚠️ **Moment de bascule ≈ {time_str}**")
    lines.append("   → **Évite de faire un long mix pile à cet endroit** (chevauchement de tonalités = risque de clash harmonique)")

    if ends_in_target or perc > 40:
        lines.append("")
        lines.append("**🚀 Pour une montée d’énergie volontaire :**")
        lines.append(f"   → Sors sur la fin → enchaîne sur un track **+3** ou **+7** depuis **{target_camelot}**")
        lines.append(f"     Ex : {target_camelot} → **{get_neighbor_camelot(target_camelot, 3)}** ou **{get_neighbor_camelot(target_camelot, 7)}**")
        lines.append("     → C’est une vraie « modulation DJ » qui donne du punch !")

    lines.append("")
    lines.append("**Choix le plus safe pour le track suivant :**")
    if priority == "target":
        lines.append(f"→ **{target_camelot}** ou voisins (±1 sur la même roue A/B)")
    else:
        lines.append(f"→ **{principal_camelot}** ou voisins (±1)")

    lines.append("")
    lines.append("**Compatibilités safe (basé sur overlap notes >85%) :**")
    compat_principal = get_compatible_keys(data['best_verified_key'])
    lines.append(f"Pour principal ({data['best_verified_key']}) : {', '.join(compat_principal)}")
    if data.get('modulation'):
        compat_target = get_compatible_keys(data['target_key'])
        lines.append(f"Pour target ({data['target_key']}) : {', '.join(compat_target)}")
    lines.append("→ Vérifie auditif pour éviter clashes sur intervalles clés (ex : b7 vs 7 maj).")

    return "\n".join(lines)

# --- AJOUTS POUR PIANO COMPANION INTÉGRATION ---

def get_mode_intervals(mode):
    """Retourne les intervalles pour un mode donné."""
    if mode in ['major', 'ionian']:
        return [0, 2, 4, 5, 7, 9, 11]
    elif mode == 'minor' or mode == 'aeolian':
        return [0, 2, 3, 5, 7, 8, 10]
    elif mode == 'mixolydian':
        return [0, 2, 4, 5, 7, 9, 10]
    elif mode == 'dorian':
        return [0, 2, 3, 5, 7, 9, 10]
    elif mode == 'phrygian':
        return [0, 1, 3, 5, 7, 8, 10]
    elif mode == 'lydian':
        return [0, 2, 4, 6, 7, 9, 11]
    elif mode == 'locrian':
        return [0, 1, 3, 5, 6, 8, 10]
    else:
        return []  # Mode inconnu

def get_diatonic_chords(key):
    """Génère les accords diatoniques pour une tonalité/mode, comme dans Piano Companion."""
    if not key or key in ["Unknown", "Atonal"]:
        return []
    
    try:
        note, mode = key.split()
        root_idx = NOTES_LIST.index(note)
    except ValueError:
        return []
    
    scale_intervals = get_mode_intervals(mode)
    if not scale_intervals:
        return []
    
    # Types d'accords diatoniques approximatifs pour modes (basés sur tierce/quinte)
    chord_types = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']  # Défaut maj ; adapter si besoin pour modes
    if mode in ['minor', 'aeolian', 'dorian', 'phrygian']:
        chord_types = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']
    roman_numerals = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'] if 'major' in mode else ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']
    
    scale_notes = [NOTES_LIST[(root_idx + interv) % 12] for interv in scale_intervals]
    
    chords = []
    for i in range(7):
        chord_root = scale_notes[i]
        ctype = chord_types[i]
        if ctype == 'maj':
            third_interval = 4
            fifth_interval = 7
        elif ctype == 'min':
            third_interval = 3
            fifth_interval = 7
        elif ctype == 'dim':
            third_interval = 3
            fifth_interval = 6
        third = NOTES_LIST[(NOTES_LIST.index(chord_root) + third_interval) % 12]
        fifth = NOTES_LIST[(NOTES_LIST.index(chord_root) + fifth_interval) % 12]
        chord_notes = [chord_root, third, fifth]
        chords.append({
            'roman': roman_numerals[i],
            'name': f"{chord_root}{ctype}",
            'notes': ' '.join(chord_notes),
            'notes_list': chord_notes  # Ajout pour test consonance
        })
    
    return chords

def get_diatonic_notes(key):
    """Retourne les notes uniques diatoniques d'une tonalité/mode (pour comparaison)."""
    chords = get_diatonic_chords(key)
    if not chords:
        return set()
    diat_notes = set()
    for chord in chords:
        diat_notes.update(chord['notes'].split())
    return diat_notes

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
        background: rgba(239, 68, 68, 0.20); color: #fca5a5;
        padding: 18px; border-radius: 12px; border: 1px solid #ef4444;
        margin: 20px 0; font-weight: bold; font-family: 'JetBrains Mono', monospace;
        line-height: 1.6; font-size: 1.05em;
    }
    .modulation-alert .detail {
        color: #fbbf24; font-size: 1.1em;
    }
    .modulation-alert .nature {
        font-size: 0.92em; opacity: 0.85; color: #fca5a5;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---
def apply_sniper_filters(y, sr, strict=True, margin=4.0, low_freq=80, high_freq=5000):
    # Made parameters adjustable for retry
    if strict:
        y_harm = librosa.effects.harmonic(y, margin=margin)
        nyq = 0.5 * sr
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = butter(4, [low, high], btype='band')
        return lfilter(b, a, y_harm)
    else:
        return y  # Version non filtrée pour fusion

def get_bass_priority(y, sr, low_cutoff=150):
    nyq = 0.5 * sr
    b, a = butter(2, low_cutoff/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def solve_key_sniper(chroma_vector, bass_vector, atonal_threshold=0.7, bonus_weight=0.20):
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    key_scores = {f"{NOTES_LIST[i]} {mode}": [] for mode in MODES for i in range(12)}  # Extension aux modes
    
    for p_name, p_data in PROFILES.items():
        for mode in MODES:
            for i in range(12):
                if mode not in p_data:
                    base_mode = "major" if mode in ['major', 'ionian', 'mixolydian', 'lydian'] else "minor"
                    profile = p_data[base_mode]
                else:
                    profile = p_data[mode]
                
                score = np.corrcoef(cv, np.roll(profile, i))[0, 1]
                
                # Seuils adaptatifs basés sur variance chroma (plus robuste)
                var_cv = np.var(cv)
                leading_threshold = 0.25 + var_cv * 0.1  # Exemple adaptatif
                
                # Bonus par mode spécifique
                dom_idx = (i + 7) % 12  # V (quinte)
                third_idx = (i + 4 if mode in ['major', 'ionian', 'mixolydian', 'lydian'] else i + 3) % 12
                sixth_idx = (i + 9) % 12  # Pour dorian/minor diff
                
                if mode in ['minor', 'aeolian', 'phrygian', 'locrian']:
                    leading_tone = (i + 11) % 12
                    if cv[leading_tone] > leading_threshold:
                        score *= 1.25  # Réduit pour éviter surboost
                elif mode == 'dorian':
                    if cv[sixth_idx] > 0.40:  # Boost 6 majeure
                        score *= 1.15
                elif mode == 'mixolydian':
                    b7_idx = (i + 10) % 12
                    if cv[b7_idx] > 0.35 and cv[(i + 11) % 12] < 0.20:  # Boost b7, pénalise 7 majeure
                        score *= 1.20
                # Ajoute pour autres modes (lydian: boost #4 = i+6, etc.)
                elif mode == 'lydian':
                    sharp4_idx = (i + 6) % 12
                    if cv[sharp4_idx] > 0.40:
                        score *= 1.15
                elif mode == 'phrygian':
                    b2_idx = (i + 1) % 12
                    if cv[b2_idx] > 0.35:
                        score *= 1.15
                elif mode == 'locrian':
                    dim5_idx = (i + 6) % 12
                    if cv[dim5_idx] < 0.20:  # Pénalise si 5 dim faible
                        score *= 0.85
                
                # Bonus généraux (bass, third, fifth) inchangés, mais ajoute seuil adaptatif
                if bv[i] > max(0.5, 1 - var_cv):
                    score += (bv[i] * bonus_weight)
                
                if cv[third_idx] > max(0.4, 1 - var_cv):
                    score += 0.12
                
                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > max(0.4, 1 - var_cv):
                    score += 0.08
                
                key_name = f"{NOTES_LIST[i]} {mode}"
                key_scores[key_name].append(score)
    
    for key_name, scores in key_scores.items():
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_key = key_name
    
    if best_overall_score < atonal_threshold:
        best_key = "Atonal"
        best_overall_score = 0
    
    return {"key": best_key, "score": best_overall_score}

def seconds_to_mmss(seconds):
    if seconds is None:
        return "??:??"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def test_chord_consonance(chroma_norm, chord_notes_list):
    """Teste la consonance d'un accord en sommant les valeurs chroma normalisées de ses notes."""
    try:
        indices = [NOTES_LIST.index(note) for note in chord_notes_list]
        consonance_score = np.sum([chroma_norm[idx] for idx in indices]) / len(indices)  # Moyenne pour normalisation
        return consonance_score
    except ValueError:
        return 0

# AJOUT : Fonction pour consonance d'une note individuelle (sur toute la durée)
def get_note_consonance(chroma_norm, note):
    """Calcule la consonance d'une note individuelle (sa force normalisée dans la chroma globale)."""
    try:
        idx = NOTES_LIST.index(note)
        return chroma_norm[idx]
    except ValueError:
        return 0

# AJOUT : Score de consonance pour une tonalité (moyenne des consonances des notes diatoniques + meilleure note consonante)
def get_key_consonance_score(key, chroma_norm):
    """Calcule un score de consonance pour une tonalité basée sur ses notes diatoniques et la meilleure note consonante."""
    diat_notes = list(get_diatonic_notes(key))
    if not diat_notes:
        return 0
    note_consonances = [get_note_consonance(chroma_norm, n) for n in diat_notes]
    avg_consonance = np.mean(note_consonances)
    best_note_consonance = max(note_consonances)  # Meilleure note consonante sur la durée
    return 0.6 * avg_consonance + 0.4 * best_note_consonance  # Pondération

def infer_chord_key(chord_name):
    """Infère une clé (root + mode) à partir du nom de l'accord pour mapper à Camelot."""
    if not chord_name or chord_name == "None":
        return "??"
    try:
        root = chord_name[:-3] if 'dim' in chord_name else chord_name[:-3] if 'min' in chord_name else chord_name[:-3] if 'maj' in chord_name else chord_name
        ctype = chord_name[len(root):]
        mode = 'major' if ctype == 'maj' else 'minor' if ctype == 'min' else 'locrian' if ctype == 'dim' else 'major'
        key_str = f"{root} {mode}"
        return CAMELOT_MAP.get(key_str, "??")
    except:
        return "??"

# AJOUT : Fonction pour analyser un segment en parallèle
def analyze_segment(start, y_filt_strict, y_filt_soft, sr, tuning, atonal_thresh, continue_thresh, bass_bonus, bass_low_cutoff):
    idx_start, idx_end = int(start * sr), int((start + 6) * sr)  # Step fixe à 6s
    seg_strict = y_filt_strict[idx_start:idx_end]
    seg_soft = y_filt_soft[idx_start:idx_end]
    if len(seg_strict) < 1000 or np.max(np.abs(seg_strict)) < 0.01:
        return None
    
    c_raw_strict = librosa.feature.chroma_cqt(y=seg_strict, sr=sr, tuning=tuning, n_chroma=36, bins_per_octave=36)
    c_raw_soft = librosa.feature.chroma_cqt(y=seg_soft, sr=sr, tuning=tuning, n_chroma=36, bins_per_octave=36)
    c_avg = 0.7 * np.mean(c_raw_strict, axis=1) + 0.3 * np.mean(c_raw_soft, axis=1)
    c_avg = np.sum(c_avg.reshape(12, 3), axis=1)
    b_seg = get_bass_priority(y_filt_strict[idx_start:idx_end], sr, low_cutoff=bass_low_cutoff)
    res = solve_key_sniper(c_avg, b_seg, atonal_threshold=atonal_thresh, bonus_weight=bass_bonus)
    
    if res['score'] < continue_thresh:
        return None
    return {"Temps": start, "Note": res['key'], "Conf": res['score']}

def process_audio_precision(file_bytes, file_name, _progress_callback=None, retry=False):
    ext = file_name.split('.')[-1].lower()
    try:
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
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=22050, mono=True)
    except Exception as e:
        st.error(f"Erreur de lecture du fichier {file_name}: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)

    # Adjustable params for retry
    strict_margin = 2.0 if retry else 4.0  # Softer harmonic margin on retry
    low_freq = 60 if retry else 80  # Lower low freq cutoff
    high_freq = 8000 if retry else 5000  # Higher high freq
    atonal_thresh = 0.5 if retry else 0.7  # Lower atonal threshold
    continue_thresh = 0.75 if retry else 0.9  # Lower continue threshold
    bass_bonus = 0.30 if retry else 0.20  # Higher bass bonus
    bass_low_cutoff = 120 if retry else 150  # Adjust bass cutoff

    y_filt_strict = apply_sniper_filters(y, sr, strict=True, margin=strict_margin, low_freq=low_freq, high_freq=high_freq)
    y_filt_soft = apply_sniper_filters(y, sr, strict=False)

    # AJOUT : Analyse globale (chroma sur tout le morceau)
    chroma_raw_global = librosa.feature.chroma_cqt(y=y_filt_strict, sr=sr, tuning=tuning, n_chroma=36, bins_per_octave=36)
    chroma_avg_global = np.mean(chroma_raw_global, axis=1)
    chroma_avg_global = np.sum(chroma_avg_global.reshape(12, 3), axis=1)  # Fold to 12
    bass_global = get_bass_priority(y, sr, low_cutoff=bass_low_cutoff)
    global_res = solve_key_sniper(chroma_avg_global, bass_global, atonal_threshold=atonal_thresh, bonus_weight=bass_bonus)
    global_key = global_res['key']
    global_score = global_res['score']

    # Analyse segmentée en parallèle
    timeline = []
    votes = Counter()
    segments = list(range(0, max(1, int(duration) - 6), 2))  # Step de 6s avec overlap de 4s
    total_segments = len(segments)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(analyze_segment, start, y_filt_strict, y_filt_soft, sr, tuning, atonal_thresh, continue_thresh, bass_bonus, bass_low_cutoff) for start in segments]
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            if _progress_callback:
                prog_internal = int((idx / total_segments) * 100)
                _progress_callback(prog_internal, f"Scan : {segments[idx]}s / {int(duration)}s" + (" (retry mode)" if retry else ""))
            result = future.result()
            if result:
                timeline.append(result)
                weight = 1.5 if (result['Temps'] < 15 or result['Temps'] > (duration - 20)) else 1.0
                votes[result['Note']] += int(result['Conf'] * 100 * weight)

    if not votes:
        default_res = {"key": "Atonal", "conf": 0, "tempo": 0, "tuning": 440, "modulation": False, "name": file_name, "diatonic_chords": [], "target_diatonic_chords": [], "validation_score": 0, "key_alternatives": [], "best_chord": "None", "best_chord_consonance": 0, "best_global_chord": "None", "best_global_consonance": 0, "camelot": "??", "target_camelot": None, "mod_target_percentage": 0, "mod_ends_in_target": False, "modulation_time_str": None, "chroma": [0]*12, "timeline": [], "best_verified_key": "Atonal"}
        if not retry:
            st.warning(f"Resultat faible pour {file_name} - Réanalyse automatique avec params relaxés...")
            return process_audio_precision(file_bytes, file_name, _progress_callback, retry=True)
        else:
            return default_res

    most_common = votes.most_common(3)  # Top 3 pour candidats

    segment_key = most_common[0][0]
    segment_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == segment_key]) * 100)
    
    mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / max(1, sum(votes.values()))) > 0.25
    target_key = most_common[1][0] if mod_detected else None

    modulation_time = None
    target_percentage = 0
    ends_in_target = False

    if mod_detected and target_key:
        target_times = np.array([t["Temps"] for t in timeline if t["Note"] == target_key])
        if len(target_times) > 3:
            dist = pdist(target_times.reshape(-1,1), 'euclidean')
            Z = linkage(target_times.reshape(-1,1), method='single')
            clust = fcluster(Z, t=5, criterion='distance')  # Clusters si <5s apart
            max_cluster_size = max(Counter(clust).values()) * 2  # Taille en secondes approx
            if max_cluster_size < 10:  # Seuil minimal pour vraie modulation
                mod_detected = False  # Ignore si pas continu
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

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_avg = chroma_avg_global  # Utilise la globale maintenant
    chroma_norm = chroma_avg / np.max(chroma_avg + 1e-6)

    # --- AJOUT : Comparaison avec top 5 notes dominantes pour décision finale ---
    top_indices = np.argsort(chroma_norm)[-5:]  # Top 5 notes
    top_notes_weights = {NOTES_LIST[i]: chroma_norm[i] for i in top_indices if chroma_norm[i] > 0.1 * np.max(chroma_norm)}

    # Candidats : Top 3 des votes + global key
    candidates = [mc[0] for mc in most_common] + [global_key]
    candidates = list(set(candidates))  # Uniques

    # Score match : Proportion de top notes dans diatoniques (pondéré 0.3 sur conf)
    matches = {}
    for key in candidates:
        diat_notes = get_diatonic_notes(key)
        match_score = sum(top_notes_weights.get(n, 0) for n in diat_notes) / (sum(top_notes_weights.values()) + 1e-6)
        # Bonus si tierce match mode
        try:
            note, mode = key.split()
            root_idx = NOTES_LIST.index(note)
            third_idx = (root_idx + 4 if 'major' in mode or mode in ['ionian', 'mixolydian', 'lydian'] else root_idx + 3) % 12
            if NOTES_LIST[third_idx] in top_notes_weights:
                match_score += 0.2  # Bonus tierce
        except ValueError:
            match_score = 0
        matches[key] = match_score

    # AJOUT : Scores combinés (segmenté + global + consonance)
    combined_scores = {}
    for key in candidates:
        segment_score = votes[key] / sum(votes.values()) if sum(votes.values()) > 0 else 0
        global_match = 1 if key == global_key else 0.5 * np.corrcoef(chroma_avg_global, np.roll(PROFILES["krumhansl"]["major" if "major" in key else "minor"], NOTES_LIST.index(key.split()[0])))[0,1]  # Approx global score pour candidat
        consonance_score = get_key_consonance_score(key, chroma_norm)
        combined = 0.4 * segment_score + 0.4 * global_match + 0.2 * consonance_score
        combined_scores[key] = combined

    # Choisir la meilleure tonalité principale (exacte)
    final_key = max(combined_scores, key=combined_scores.get)
    final_conf = int(combined_scores[final_key] * 100)  # Normalisé en %

    # Auto-correction si modulation faible
    if mod_detected and target_percentage < 20:
        mod_detected = False
        target_key = None
        final_key = global_key if combined_scores[global_key] > combined_scores[final_key] else final_key

    # Additional verification for best key (stabilité)
    all_candidates = [final_key] + [k for k in candidates if k != final_key]
    stability_scores = {}
    for k in all_candidates:
        segments_k = [t for t in timeline if t['Note'] == k]
        if segments_k:
            prop = len(segments_k) / len(timeline)
            avg_conf = np.mean([t['Conf'] for t in segments_k])
            stability = prop * avg_conf
        else:
            stability = 0
        combined = 0.6 * stability + 0.4 * matches.get(k, 0)
        stability_scores[k] = combined

    best_verified_key = max(stability_scores, key=stability_scores.get)

    # --- AJOUT : Test de consonance sur tous les accords possibles ---
    diatonic_chords = get_diatonic_chords(final_key)
    consonance_scores = {}
    for chord in diatonic_chords:
        score = test_chord_consonance(chroma_norm, chord['notes_list'])
        consonance_scores[chord['name']] = score

    # Meilleur accord (celui avec le score le plus haut)
    if consonance_scores:
        best_chord_name = max(consonance_scores, key=consonance_scores.get)
        best_chord_score = consonance_scores[best_chord_name] * 100  # Pour %
    else:
        best_chord_name = "None"
        best_chord_score = 0

    # --- AJOUT : Meilleur accord global (tous les triads possibles, excluant root de main/target si applicable) ---
    all_chords = []
    for root in NOTES_LIST:
        # Major
        third_maj = NOTES_LIST[(NOTES_LIST.index(root) + 4) % 12]
        fifth_maj = NOTES_LIST[(NOTES_LIST.index(root) + 7) % 12]
        all_chords.append({
            'name': f"{root}maj",
            'notes_list': [root, third_maj, fifth_maj]
        })
        # Minor
        third_min = NOTES_LIST[(NOTES_LIST.index(root) + 3) % 12]
        fifth_min = NOTES_LIST[(NOTES_LIST.index(root) + 7) % 12]
        all_chords.append({
            'name': f"{root}min",
            'notes_list': [root, third_min, fifth_min]
        })
        # Dim
        third_dim = NOTES_LIST[(NOTES_LIST.index(root) + 3) % 12]
        fifth_dim = NOTES_LIST[(NOTES_LIST.index(root) + 6) % 12]
        all_chords.append({
            'name': f"{root}dim",
            'notes_list': [root, third_dim, fifth_dim]
        })

    overall_consonance_scores = {}
    for chord in all_chords:
        score = test_chord_consonance(chroma_norm, chord['notes_list'])
        overall_consonance_scores[chord['name']] = score

    # Identifier les chords à exclure (root de main et target)
    exclude_chords = set()
    try:
        note, mode = final_key.split()
        root_chord = f"{note}maj" if 'major' in mode or mode in ['ionian', 'mixolydian', 'lydian'] else f"{note}min"
        exclude_chords.add(root_chord)
    except ValueError:
        pass
    if target_key:
        try:
            t_note, t_mode = target_key.split()
            target_root_chord = f"{t_note}maj" if 'major' in t_mode or t_mode in ['ionian', 'mixolydian', 'lydian'] else f"{t_note}min"
            exclude_chords.add(target_root_chord)
        except ValueError:
            pass

    # Trier et trouver le meilleur non exclu
    sorted_chords = sorted(overall_consonance_scores.items(), key=lambda x: x[1], reverse=True)
    best_global_chord = None
    best_global_score = 0
    for ch, sc in sorted_chords:
        if ch not in exclude_chords:
            best_global_chord = ch
            best_global_score = sc * 100
            break

    if not best_global_chord:
        best_global_chord = "None"
        best_global_score = 0

    # Génération accords pour affichage (sur final_key ajustée)
    diatonic_chords = get_diatonic_chords(final_key)
    target_diatonic_chords = get_diatonic_chords(target_key) if target_key else []

    # Validation score pour affichage (coverage globale)
    validation_score = matches.get(final_key, 0) * 100

    res_obj = {
        "key": final_key,
        "camelot": CAMELOT_MAP.get(final_key, "??"),  # Utilise le mapping étendu
        "conf": min(final_conf, 99),
        "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline,
        "chroma": chroma_avg.tolist(),
        "modulation": mod_detected,
        "target_key": target_key,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "modulation_time_str": seconds_to_mmss(modulation_time) if mod_detected else None,
        "mod_target_percentage": round(target_percentage, 1) if mod_detected else 0,
        "mod_ends_in_target": ends_in_target if mod_detected else False,
        "name": file_name,
        "diatonic_chords": diatonic_chords,
        "target_diatonic_chords": target_diatonic_chords,
        "validation_score": int(validation_score),
        "key_alternatives": [k for k in candidates if k != final_key],
        "best_chord": best_chord_name,  # Best diatonic
        "best_chord_consonance": int(best_chord_score),
        "best_global_chord": best_global_chord,  # Best global excluant roots
        "best_global_consonance": int(best_global_score),
        "best_verified_key": best_verified_key
    }

    # Auto-retry logic if result is poor and not already in retry mode
    if (res_obj["key"] == "Atonal" or res_obj["conf"] < 50) and not retry:
        st.warning(f"Résultat faible pour {file_name} (Atonal ou conf <50%) - Réanalyse automatique avec params relaxés...")
        retry_res = process_audio_precision(file_bytes, file_name, _progress_callback, retry=True)
        # Choose the better result (higher conf)
        if retry_res["conf"] > res_obj["conf"]:
            st.info(f"Retry amélioré le résultat pour {file_name}.")
            res_obj = retry_res

    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            df_tl = pd.DataFrame(timeline)
            fig_tl = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
            img_tl = fig_tl.to_image(format="png", width=1000, height=500)
            
            fig_rd = go.Figure(data=go.Scatterpolar(r=res_obj['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_rd.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            img_rd = fig_rd.to_image(format="png", width=600, height=600)

            mod_line = ""
            if mod_detected:
                perc = res_obj["mod_target_percentage"]
                end_txt = " → **fin en " + target_key.upper() + " (" + res_obj['target_camelot'] + ")**" if res_obj['mod_ends_in_target'] else ""
                mod_line = f"  *MODULATION →* `{target_key.upper()} ({res_obj['target_camelot']})` ≈ **{res_obj['modulation_time_str']}** ({perc}%){end_txt}"

            verified_camelot = CAMELOT_MAP.get(res_obj['best_verified_key'], "??")
            chord_camelot = infer_chord_key(res_obj['best_global_chord'])

            compat_principal = get_compatible_keys(res_obj['best_verified_key'])
            compat_target = get_compatible_keys(target_key) if mod_detected else []

            caption = (f"  *RCDJ228 MUSIC SNIPER - RAPPORT*\n━━━━━━━━━━━━\n"
                       f"  *FICHIER:* `{file_name}`\n"
                       f"  *TONALITÉ PRINCIPALE:* `{final_key.upper()} ({res_obj['camelot']})`\n"
                       f"  *CAMELOT:* `{res_obj['camelot']}`\n"
                       f"  *CONFIANCE:* `{res_obj['conf']}%`\n"
                       f"  *TONALITÉ VÉRIFIÉE:* `{res_obj['best_verified_key'].upper()} ({verified_camelot})`\n"
                       f"  *MEILLEUR ACCORD:* `{res_obj['best_global_chord'].upper()} ({chord_camelot})` ({res_obj['best_global_consonance']}% consonance)\n"
                       f"  *TEMPO:* `{res_obj['tempo']} BPM`\n"
                       f"  *ACCORDAGE:* `{res_obj['tuning']} Hz`\n"
                       f"{mod_line if mod_detected else '  *STABILITÉ TONALE:* OK'}\n━━━━━━━━━━━━"
                       f"*Compat safe principal:* {', '.join(compat_principal[:3])}\n*Compat safe target:* {', '.join(compat_target[:3]) if mod_detected else 'N/A'}")

            # ─── AJOUT : CONSEIL RAPIDE MIX EN VERSION ULTRA-RÉSUMÉE ───
            advice_text = get_mixing_advice(res_obj)
            summary_advice = ""
            if advice_text:
                if "fin en target" in advice_text or "Oui : le morceau termine" in advice_text:
                    summary_advice = f"→ Termine en {res_obj['target_camelot']} → privilégie cette tonalité pour le suivant !"
                elif res_obj.get('mod_target_percentage', 0) > 45:
                    summary_advice = f"→ {res_obj['target_camelot']} très présent → traite presque comme track en {res_obj['target_camelot']}"
                else:
                    summary_advice = "→ Modulation ponctuelle → reste sur tonalité principale"
            
            if summary_advice:
                caption += f"\n\n*Conseil rapide mix :* {summary_advice}"
            else:
                caption += "\n\n*Pas de modulation détectée → mix safe sur la tonalité principale*"

            files = {'p1': ('timeline.png', img_tl, 'image/png'), 'p2': ('radar.png', img_rd, 'image/png')}
            media = [
                {'type': 'photo', 'media': 'attach://p1', 'caption': caption, 'parse_mode': 'Markdown'},
                {'type': 'photo', 'media': 'attach://p2'}
            ]
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup",
                data={'chat_id': CHAT_ID, 'media': json.dumps(media)},
                files=files,
                timeout=20
            )
        except Exception:
            pass

    del y, y_filt_strict, y_filt_soft
    gc.collect()
    return res_obj

def get_chord_js(btn_id, key_str):
    try:
        note, mode = key_str.split()
    except ValueError:
        return ""
    intervals_str = 'minor' if mode in ['minor', 'aeolian', 'dorian', 'phrygian', 'locrian'] else 'major'
    js_code = f"""
document.getElementById('{btn_id}').onclick = function() {{
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
  const intervals = '{intervals_str}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
  intervals.forEach(i => {{
    const o = ctx.createOscillator(); const g = ctx.createGain();
    o.type = 'triangle'; 
    o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
    g.gain.setValueAtTime(0, ctx.currentTime);
    g.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
    o.connect(g); g.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 2.0);
  }});
}};
"""
    return textwrap.dedent(js_code)

# ────────────────────────────────────────────────
#               INTERFACE PRINCIPALE
# ────────────────────────────────────────────────

st.title("🎯 RCDJ228 MUSIC SNIPER")

uploaded_files = st.file_uploader(
    "Déposez vos fichiers audio (mp3, wav, flac, m4a)",
    type=['mp3','wav','flac','m4a'],
    accept_multiple_files=True
)

if uploaded_files:
    global_progress_placeholder = st.empty()
    total_files = len(uploaded_files)
    results_container = st.container()
    
    for i, f in enumerate(reversed(uploaded_files)):
        global_progress_placeholder.markdown(f"""
            <div style="padding:15px; border-radius:15px; background-color:rgba(16,185,129,0.12); border:1px solid #10b981; margin-bottom:20px;">
                <h3 style="margin:0; color:#10b981;">Analyse en cours : {i+1} / {total_files}</h3>
                <p style="margin:6px 0 0 0; opacity:0.85;">{f.name}</p>
            </div>
            """, unsafe_allow_html=True)

        with st.status(f"Analyse → {f.name}", expanded=True) as status:
            inner_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(val, msg):
                inner_bar.progress(val)
                status_text.code(msg)

            audio_bytes = f.getvalue()
            data = process_audio_precision(audio_bytes, f.name, update_progress)
            status.update(label=f"{f.name} terminé", state="complete", expanded=False)

        if data:
            with results_container:
                st.markdown(f"<div class='file-header'> {data['name']}</div>", unsafe_allow_html=True)
                
                key_col, verified_col, chord_col = st.columns(3)

                with key_col:
                    color = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 85 else "linear-gradient(135deg, #1e293b, #0f172a)"

                    mod_alert = ""
                    if data.get('modulation'):
                        perc = data.get('mod_target_percentage', 0)
                        ends_in_target = data.get('mod_ends_in_target', False)
                        time_str = data.get('modulation_time_str', '??:??')
                        
                        if perc < 25:
                            nature = "passage court / ponctuel"
                        elif perc < 50:
                            nature = "section significative"
                        else:
                            nature = "dominante sur une grande partie"

                        end_txt = " → **fin en " + data['target_key'].upper() + "**" if ends_in_target else ""
                        
                        mod_alert = f"""
                            <div class="modulation-alert">
                                MODULATION → {data['target_key'].upper()} ({data['target_camelot']})<br>
                                <span class="detail">≈ {time_str} – {perc}% du morceau{end_txt}</span><br>
                                <span class="nature">({nature})</span>
                            </div>
                        """.strip()

                    st.markdown(f"""
                        <div class="report-card" style="background:{color};">
                            <h1 style="font-size:5.4em; margin:8px 0; font-weight:900;">{data['key'].upper()}</h1>
                            <p style="font-size:1.5em; opacity:0.92;">
                                CAMELOT <b>{data['camelot']}</b>  •  Confiance <b>{data['conf']}%</b>
                            </p>
                            {mod_alert}
                        </div>
                        """, unsafe_allow_html=True)

                with verified_col:
                    verified_color = "linear-gradient(135deg, #065f46, #064e3b)" if data['best_verified_key'] == data['key'] else "linear-gradient(135deg, #4338ca, #3730a3)"
                    verified_camelot = CAMELOT_MAP.get(data['best_verified_key'], "??")
                    st.markdown(f"""
                        <div class="report-card" style="background:{verified_color};">
                            <h1 style="font-size:5.4em; margin:8px 0; font-weight:900;">{data['best_verified_key'].upper()}</h1>
                            <p style="font-size:1.5em; opacity:0.92;">
                                VERIFIED BEST KEY  •  CAMELOT <b>{verified_camelot}</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                with chord_col:
                    color_chord = "linear-gradient(135deg, #4338ca, #3730a3)" if data['best_global_consonance'] > 85 else "linear-gradient(135deg, #1e293b, #0f172a)"
                    chord_camelot = infer_chord_key(data['best_global_chord'])
                    st.markdown(f"""
                        <div class="report-card" style="background:{color_chord};">
                            <h1 style="font-size:5.4em; margin:8px 0; font-weight:900;">{data['best_global_chord'].upper()}</h1>
                            <p style="font-size:1.5em; opacity:0.92;">
                                MEILLEUR ACCORD  •  CAMELOT <b>{chord_camelot}</b>  •  Consonance <b>{data['best_global_consonance']}%</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                advice = get_mixing_advice(data)
                if advice:
                    with st.expander("📋 Checklist MIX – que faire avec ce track ?", expanded=True):
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, rgba(16,185,129,0.10), rgba(16,185,129,0.04));
                                border: 1px solid rgba(16,185,129,0.4);
                                border-radius: 12px;
                                padding: 20px 24px;
                                margin: 16px 0;
                                line-height: 1.65;
                                font-size: 1.02em;
                                white-space: pre-wrap;
                            ">
                            {advice}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                m1, m2, m3 = st.columns(3)
                with m1: 
                    st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.4em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                with m2: 
                    st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2.2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                with m3:
                    btn_id = f"play_{i}_{hash(data['name'])}"
                    js = get_chord_js(btn_id, data['key'])
                    if js:
                        components.html(f"""
                            <button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.1em;">
                                TESTER L'ACCORD
                            </button>
                            <script>{js}</script>
                        """, height=110)

                c1, c2 = st.columns([2, 1])
                with c1: 
                    df_timeline = pd.DataFrame(data['timeline'])
                    if not df_timeline.empty:
                        fig_tl = px.line(
                            df_timeline, 
                            x="Temps", y="Note", 
                            markers=True, 
                            template="plotly_dark", 
                            category_orders={"Note": NOTES_ORDER}
                        )
                    else:
                        fig_tl = go.Figure()
                    fig_tl.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig_tl, use_container_width=True, key=f"tl_{i}_{hash(f.name)}")
                
                with c2: 
                    fig_rd = go.Figure(data=go.Scatterpolar(
                        r=data['chroma'], 
                        theta=NOTES_LIST, 
                        fill='toself', 
                        line_color='#10b981'
                    ))
                    fig_rd.update_layout(
                        template="plotly_dark", 
                        height=320, 
                        polar=dict(radialaxis=dict(visible=False)),
                        margin=dict(l=30,r=30,t=20,b=20)
                    )
                    st.plotly_chart(fig_rd, use_container_width=True, key=f"rd_{i}_{hash(f.name)}")
                
                # --- AJOUT : Affichage des accords et validation ---
                with st.expander("🎹 Accords diatoniques (comme dans Piano Companion) – pour valider la tonalité", expanded=False):
                    chroma_avg = np.array(data['chroma'])
                    chroma_norm = chroma_avg / np.max(chroma_avg + 1e-6)

                    def add_consonance_info(df, chroma_norm):
                        if df.empty:
                            return df
                        df['Consonance (%)'] = df['notes_list'].apply(lambda notes: round(test_chord_consonance(chroma_norm, notes) * 100, 1))
                        df['notes_with_%'] = df['notes_list'].apply(lambda notes: ' '.join([f"{n} ({round(chroma_norm[NOTES_LIST.index(n)] * 100, 1)}%)" for n in notes]))
                        df = df[['roman', 'name', 'notes_with_%', 'Consonance (%)']]
                        df.columns = ['Roman', 'Chord', 'Notes (with % consonance)', 'Chord Consonance (%)']
                        return df

                    # Main key
                    chords = data.get("diatonic_chords", [])
                    if chords:
                        df_chords = pd.DataFrame(chords)
                        df_chords = add_consonance_info(df_chords, chroma_norm)
                        main_camelot = data.get('camelot', '??')
                        st.subheader(f"Accords pour la tonalité principale ({data['key'].upper()} - Camelot {main_camelot})")
                        st.table(df_chords)
                        st.markdown(f"**Score de validation (coverage chroma) :** {data.get('validation_score', 0)}%")
                        if data.get('key_alternatives'):
                            st.warning(f"Alternatives possibles : {', '.join(data['key_alternatives'])} – Verified Best: {data['best_verified_key']}")
                    else:
                        st.info("Pas d'accords générés (tonalité inconnue).")

                    # Verified key
                    verified_key = data.get('best_verified_key', 'Unknown')
                    verified_chords = get_diatonic_chords(verified_key)
                    if verified_chords and verified_key != data['key']:
                        df_verified = pd.DataFrame(verified_chords)
                        df_verified = add_consonance_info(df_verified, chroma_norm)
                        verified_camelot = CAMELOT_MAP.get(verified_key, "??")
                        st.subheader(f"Accords pour la tonalité vérifiée ({verified_key.upper()} - Camelot {verified_camelot})")
                        st.table(df_verified)

                    # Pour la modulation
                    target_chords = data.get("target_diatonic_chords", [])
                    if target_chords:
                        df_target = pd.DataFrame(target_chords)
                        df_target = add_consonance_info(df_target, chroma_norm)
                        target_camelot = data.get('target_camelot', '??')
                        st.subheader(f"Accords pour la tonalité cible (modulation) ({data['target_key'].upper()} - Camelot {target_camelot})")
                        st.table(df_target)
                    
                    # Affichage du meilleur accord
                    st.markdown(f"**Meilleur accord consonant :** {data.get('best_chord', 'None')} (Score: {data.get('best_chord_consonance', 0)}%)")
                
                st.markdown("<hr style='border-color:#30363d; margin:40px 0 30px 0;'>", unsafe_allow_html=True)

    global_progress_placeholder.success(f"Analyse terminée — {total_files} piste(s) traitée(s) avec succès !")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Contrôles Sniper")
    if st.button("🔄 Vider le cache & relancer"):
        st.cache_data.clear()
        st.rerun()
