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
import os
import tempfile
import shutil
import traceback

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# ══════════════════════════════════════════════════════════════════════════
# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODAL_MODES = ['major', 'minor']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in MODAL_MODES]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# --- PROFILS DE RÉFÉRENCE MAJEUR/MINEUR ---
PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4],
    }
}

ALL_MODES = ['major', 'minor']
MODE_THIRD = {'major': 4, 'minor': 3}

# --- PRÉCOMPUTATION DES PROFILS ROULÉS ---
PROFILES_ROLLED = {
    p_name: {
        mode: [np.roll(PROFILES[p_name][mode], i) for i in range(12)]
        for mode in ALL_MODES
    }
    for p_name in PROFILES
}

MODES_PROFILES = {m: PROFILES["krumhansl"][m] for m in ALL_MODES}
MODAL_PROFILES_ROLLED = {
    m_name: [np.roll(MODES_PROFILES[m_name], i) for i in range(12)]
    for m_name in MODES_PROFILES
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    .metric-box:hover { border-color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# --- CORRECTION AUTOMATIQUE DE L'ACCORDAGE ---
def auto_correct_tuning(y, sr):
    """
    Correction automatique de l'accordage à 440 Hz standard.
    """
    tuning_cents = librosa.estimate_tuning(y=y, sr=sr)
    correction_applied = False
    correction_info = ""

    if abs(tuning_cents) > 5.0:
        n_steps = -tuning_cents / 100.0
        y_corrected = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
        correction_applied = True
        correction_info = f"✅ Accordage corrigé automatiquement : {tuning_cents:.1f} cents → 440.0 Hz"
        final_tuning_cents = 0.0
    else:
        y_corrected = y
        correction_applied = False
        correction_info = f"✅ Déjà accordé à 440 Hz (±{abs(tuning_cents):.1f} cents)"
        final_tuning_cents = tuning_cents

    return y_corrected, final_tuning_cents, correction_applied, correction_info


# ══════════════════════════════════════════════════════════════════════════
# --- MOTEURS DE CALCUL ---
def arbitrage_expert_universel(chroma, bass_vec, key_cons, key_dom, cam_map, y_harm=None, sr=None, tuning=0.0):
    cam_c = get_exact_camelot(key_cons)
    cam_d = get_exact_camelot(key_dom)

    if not cam_c or not cam_d:
        return {"key": key_cons, "duel_actif": False, "dist_num": 99, "dist_mode": 99, "type": "NONE"}

    val_c, mode_c = int(cam_c[:-1]), cam_c[-1]
    val_d, mode_d = int(cam_d[:-1]), cam_d[-1]

    dist_num = abs(val_c - val_d)
    if dist_num > 6:
        dist_num = 12 - dist_num

    dist_mode = 0 if mode_c == mode_d else 1

    if dist_num <= 1 and dist_mode <= 1:
        idx_c = NOTES_LIST.index(key_cons.split()[0])
        idx_d = NOTES_LIST.index(key_dom.split()[0])

        def get_advanced_strength(idx, mode, chroma_vec, b_vec):
            quinte = chroma_vec[(idx + 7) % 12]
            tierce = chroma_vec[(idx + 3) % 12] if mode == 'minor' else chroma_vec[(idx + 4) % 12]
            basse  = b_vec[idx]
            return (quinte * 0.5) + (tierce * 0.3) + (basse * 0.2)

        force_c = get_advanced_strength(idx_c, key_cons.split()[1], chroma, bass_vec)
        force_d = get_advanced_strength(idx_d, key_dom.split()[1], chroma, bass_vec)

        if key_cons == "D minor" and key_dom == "C major":
            si_idx = NOTES_LIST.index("B")
            if chroma[si_idx] > 0.15:
                force_d *= 1.25

        if y_harm is not None and sr is not None:
            sub_vec = get_sub_bass_priority(y_harm, sr, tuning=tuning)
            if sub_vec[idx_d] > sub_vec[idx_c] * 1.2:
                return {"key": key_dom, "duel_actif": True, "dist_num": dist_num, "dist_mode": dist_mode, "type": "BASS_DOMINANCE"}

        winner = key_dom if force_d > force_c * 1.15 else key_cons
        return {"key": winner, "duel_actif": True, "dist_num": dist_num, "dist_mode": dist_mode, "type": "SPECTRAL"}

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
    nyq = 0.5 * sr
    b, a = butter(2, [40/nyq, 100/nyq], btype='band')
    y_sub = lfilter(b, a, y)
    chroma_sub = librosa.feature.chroma_cqt(y=y_sub, sr=sr, n_chroma=12, tuning=tuning)
    return np.mean(chroma_sub, axis=1)

def detect_harmonic_sections(y, sr, duration, step=6, min_harm_duration=20, harm_threshold=0.3, perc_threshold=0.5, tuning=0.0):
    hop_length = 512
    chroma_full = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, hop_length=hop_length, tuning=tuning)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)

    harmonic_starts = []
    segments = range(0, int(duration) - step, step // 2)

    for start in segments:
        idx_start = int(start * sr) // hop_length
        idx_end   = int((start + step) * sr) // hop_length

        seg_chroma = chroma_full[:, idx_start:idx_end]
        chroma_var = np.var(seg_chroma)
        seg_flat   = np.mean(flatness[0, idx_start:idx_end])

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
    harm_end   = longest_section[-1] + step
    harm_start = max(harm_start, 5)
    harm_end   = min(harm_end, duration - 5)

    return harm_start, harm_end

def detect_cadence_resolution(timeline, final_key):
    parts = final_key.split()
    note  = parts[0]
    mode  = parts[1] if len(parts) > 1 else 'major'
    root_idx    = NOTES_LIST.index(note)
    dom_idx     = (root_idx + 7) % 12
    subdom_idx  = (root_idx + 5) % 12
    is_minor_family = mode == 'minor'

    resolution_count = 0
    for i in range(1, len(timeline)):
        prev_note = timeline[i-1]["Note"]
        curr_note = timeline[i]["Note"]

        dom_key = f"{NOTES_LIST[dom_idx]} {mode}"
        if is_minor_family:
            if (prev_note == f"{NOTES_LIST[dom_idx]} major" or
                prev_note == dom_key) and curr_note == final_key:
                resolution_count += 1 if 'major' in prev_note else 0.5
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
    if not key_str or "Unknown" in key_str:
        return "??"
    key_str = key_str.replace('ionian', 'major').replace('aeolian', 'minor')
    return CAMELOT_MAP.get(key_str, "??")

def solve_key_sniper(chroma_vector, bass_vector):
    best_overall_score = -1
    best_key = "Unknown"
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector   - bass_vector.min())   / (bass_vector.max()   - bass_vector.min()   + 1e-6)
    for m_name in ALL_MODES:
        for i in range(12):
            profile_scores = []
            third_idx = (i + MODE_THIRD[m_name]) % 12
            fifth_idx = (i + 7) % 12
            for p_name in PROFILES_ROLLED:
                rolled = PROFILES_ROLLED[p_name][m_name][i]
                score  = np.corrcoef(cv, rolled)[0, 1]
                if cv[fifth_idx] > 0.6 and cv[third_idx] > 0.4:
                    score += 0.25
                elif cv[fifth_idx] > 0.6:
                    score += 0.1
                profile_scores.append(score)
            avg_score = np.mean(profile_scores)
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_key = f"{NOTES_LIST[i]} {m_name}"
    return {"key": best_key, "score": best_overall_score}

def solve_key_sniper_modal(chroma_vector, bass_vector):
    best_overall_score = -1
    best_res = {"key": "Unknown", "mode": "major", "score": 0}
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    for m_name, profiles in MODAL_PROFILES_ROLLED.items():
        for i in range(12):
            rolled = profiles[i]
            score  = np.corrcoef(cv, rolled)[0, 1]
            third_idx = (i + MODE_THIRD[m_name]) % 12
            fifth_idx = (i + 7) % 12
            if cv[fifth_idx] > 0.6 and cv[third_idx] > 0.4:
                score += 0.25
            elif cv[fifth_idx] > 0.6:
                score += 0.10
            if score > best_overall_score:
                best_overall_score = score
                best_res = {
                    "key": f"{NOTES_LIST[i]} {m_name}",
                    "raw_mode": m_name,
                    "score": best_overall_score
                }
    return best_res

def get_key_score(key, chroma_vector, bass_vector):
    parts    = key.split()
    note     = parts[0]
    mode     = parts[1] if len(parts) > 1 else 'major'
    root_idx = NOTES_LIST.index(note)
    cv_norm = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    third_idx = (root_idx + MODE_THIRD.get(mode, 4)) % 12
    fifth_idx = (root_idx + 7) % 12
    profile_scores = []
    for p_name in PROFILES_ROLLED:
        if mode in PROFILES_ROLLED[p_name]:
            rolled = PROFILES_ROLLED[p_name][mode][root_idx]
        else:
            fallback = 'major'
            rolled   = PROFILES_ROLLED[p_name][fallback][root_idx]
        score = np.corrcoef(cv_norm, rolled)[0, 1]
        if cv_norm[fifth_idx] > 0.6 and cv_norm[third_idx] > 0.4:
            score += 0.25
        elif cv_norm[fifth_idx] > 0.6:
            score += 0.1
        profile_scores.append(score)
    return np.mean(profile_scores)


# ══════════════════════════════════════════════════════════════════════════
# --- ENVOI TELEGRAM CORRIGÉ ---
def send_telegram_report(caption, fig_radar, fig_timeline, file_name):
    """
    Envoi du rapport Telegram avec gestion robuste des erreurs.
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    
    try:
        # Convertir les figures en bytes
        radar_bytes = fig_radar.to_image(format="png", width=700, height=500)
        
        # Si on a une timeline, on l'ajoute
        has_timeline = fig_timeline is not None
        
        if has_timeline:
            timeline_bytes = fig_timeline.to_image(format="png", width=1000, height=450)
            
            # Méthode 1: Envoyer comme album média
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup"
            
            files = {
                'radar.png': ('radar.png', radar_bytes, 'image/png'),
                'timeline.png': ('timeline.png', timeline_bytes, 'image/png')
            }
            
            media = [
                {
                    'type': 'photo',
                    'media': 'attach://radar.png',
                    'caption': caption,
                    'parse_mode': 'Markdown'
                },
                {
                    'type': 'photo',
                    'media': 'attach://timeline.png'
                }
            ]
            
            response = requests.post(
                url,
                data={
                    'chat_id': CHAT_ID,
                    'media': json.dumps(media)
                },
                files=files,
                timeout=30
            )
            
            # Si l'album échoue, envoyer les images une par une
            if not response.ok:
                # Envoyer d'abord le radar avec la légende
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                    data={
                        'chat_id': CHAT_ID,
                        'caption': caption,
                        'parse_mode': 'Markdown'
                    },
                    files={'photo': ('radar.png', radar_bytes, 'image/png')},
                    timeout=30
                )
                
                # Puis la timeline sans légende
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                    data={'chat_id': CHAT_ID},
                    files={'photo': ('timeline.png', timeline_bytes, 'image/png')},
                    timeout=30
                )
        else:
            # Une seule image (radar uniquement)
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={
                    'chat_id': CHAT_ID,
                    'caption': caption,
                    'parse_mode': 'Markdown'
                },
                files={'photo': ('radar.png', radar_bytes, 'image/png')},
                timeout=30
            )
        
        return True
        
    except Exception as e:
        st.error(f"Erreur d'envoi Telegram : {str(e)}")
        st.error(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════
# --- PROCESS AUDIO ---
def process_audio(audio_file, file_name, progress_placeholder):
    status_text = progress_placeholder.empty()
    progress_bar = progress_placeholder.progress(0)

    def update_prog(value, text):
        progress_bar.progress(value)
        status_text.markdown(f"**{text} | {value}%**")

    try:
        update_prog(10, f"Chargement de {file_name}")
        file_bytes = audio_file.getvalue()
        ext = file_name.split('.')[-1].lower()
        if ext == 'm4a':
            audio   = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            y  = samples / (2**15)
            sr = audio.frame_rate
            if sr != 22050:
                y  = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
        else:
            try:
                y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)
            except Exception as load_err:
                try:
                    audio   = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
                    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2)).mean(axis=1)
                    y  = samples / (2**15)
                    sr = audio.frame_rate
                    if sr != 22050:
                        y  = librosa.resample(y, orig_sr=sr, target_sr=22050)
                        sr = 22050
                except Exception as pydub_err:
                    raise ValueError(
                        f"Impossible de lire le fichier audio '{file_name}'.\n"
                        f"Format '{ext}' non supporté ou fichier corrompu.\n"
                        f"Formats acceptés : mp3, wav, ogg, flac, m4a.\n"
                        f"Détail : {load_err}"
                    )

        update_prog(15, "Correction automatique de l'accordage (440 Hz)")
        y, tuning, correction_applied, correction_info = auto_correct_tuning(y, sr)
        update_prog(18, correction_info)

        update_prog(20, "Détection des sections harmoniques")
        duration   = librosa.get_duration(y=y, sr=sr)
        harm_start, harm_end = detect_harmonic_sections(y, sr, duration, tuning=tuning)
        update_prog(30, f"Section harmonique détectée : {seconds_to_mmss(harm_start)} à {seconds_to_mmss(harm_end)}")

        idx_harm_start = int(harm_start * sr)
        idx_harm_end   = int(harm_end   * sr)
        y_harm         = y[idx_harm_start:idx_harm_end]
        duration_harm  = harm_end - harm_start

        update_prog(40, "Filtrage des fréquences")
        y_filt  = apply_sniper_filters(y_harm, sr)

        chroma_avg   = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
        bass_global  = get_bass_priority(y_harm, sr, tuning=tuning)

        update_prog(50, "Analyse du spectre harmonique")

        hop_length = 512
        full_chroma_raw = librosa.feature.chroma_cqt(
            y=y_filt,
            sr=sr,
            tuning=tuning,
            n_chroma=24,
            bins_per_octave=24,
            hop_length=hop_length
        )

        step = 6
        advance = 2
        timeline = []
        votes = Counter()

        segments = range(0, int(duration_harm) - step + 1, advance) if duration_harm > step else [0]

        for i, start in enumerate(segments):
            start_frame = int(start * sr / hop_length)
            end_frame   = int((start + step) * sr / hop_length) + 1
            c_raw = full_chroma_raw[:, start_frame:end_frame]
            c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)

            idx_start = int(start * sr)
            idx_end   = int((start + step) * sr)
            b_seg = get_bass_priority(y_harm[idx_start:idx_end], sr, tuning=tuning)

            res       = solve_key_sniper(c_avg, b_seg)
            res_modal = solve_key_sniper_modal(c_avg, b_seg)

            segment_camelot = get_exact_camelot(res['key'])
            weight = 3.0 if start > (duration_harm - 15) else 2.0 if start < 10 else 1.0
            votes[res['key']] += int(res['score'] * 100 * weight)
            timeline.append({
                "Temps":  harm_start + start,
                "Note":   res['key'],
                "Camelot": segment_camelot,
                "Conf":   res['score'],
                "Mode":   res_modal.get('key', res['key'])
            })

            p_val = 50 + int((i / len(segments)) * 38)
            update_prog(p_val, "Calcul chirurgical en cours")

        update_prog(90, "Synthèse finale et validation de la tonique")
        most_common   = votes.most_common(10)
        total_votes   = sum(votes.values())

        res_global    = solve_key_sniper(chroma_avg, bass_global)
        final_key     = res_global['key']
        final_conf    = int(res_global['score'] * 100)

        res_modal_global = solve_key_sniper_modal(chroma_avg, bass_global)
        modal_key        = res_modal_global.get('key', final_key)
        modal_camelot    = get_exact_camelot(modal_key)
        modal_raw_mode   = res_modal_global.get('raw_mode', 'major')
        modal_conf       = int(res_modal_global.get('score', 0) * 100)
        modal_count      = sum(1 for t in timeline if t["Mode"] == modal_key)
        modal_presence   = (modal_count / len(timeline) * 100) if len(timeline) > 0 else 0

        cadence_score = detect_cadence_resolution(timeline, final_key)
        if cadence_score < 2 and len(most_common) > 1:
            alt_keys    = [k for k, _ in most_common if k != final_key]
            alt_cadences = {ak: detect_cadence_resolution(timeline, ak) for ak in alt_keys}
            best_alt     = max(alt_cadences, key=alt_cadences.get)
            if alt_cadences[best_alt] > cadence_score + 1:
                final_key  = best_alt
                final_conf = int(get_key_score(final_key, chroma_avg, bass_global) * 100)

        if timeline and timeline[-1]["Note"] == final_key:
            final_conf = min(final_conf + 5, 99)

        dominant_key        = most_common[0][0] if most_common else "Unknown"
        dominant_votes      = most_common[0][1] if most_common else 0
        dominant_percentage = (dominant_votes / total_votes * 100) if total_votes > 0 else 0
        dominant_conf       = int(get_key_score(dominant_key, chroma_avg, bass_global) * 100) if dominant_key != "Unknown" else 0

        final_vote_count = votes.get(final_key, 0)
        final_percentage = (final_vote_count / total_votes * 100) if total_votes > 0 else 0
        dominant_camelot = get_exact_camelot(dominant_key)

        mod_detected   = len(most_common) > 1 and (votes[most_common[1][0]] / sum(votes.values())) > 0.25
        target_key     = most_common[1][0] if mod_detected else None
        target_conf    = min(int(get_key_score(target_key, chroma_avg, bass_global) * 100), 99) if mod_detected else None

        modulation_time  = None
        target_percentage = 0
        ends_in_target   = False

        if mod_detected and target_key:
            target_times = np.array([t["Temps"] for t in timeline if t["Note"] == target_key])
            if len(target_times) > 3:
                Z     = linkage(target_times.reshape(-1, 1), method='single')
                clust = fcluster(Z, t=5, criterion='distance')
                max_cluster_size = max(Counter(clust).values()) * 2
                if max_cluster_size < 10:
                    mod_detected = False
            if mod_detected:
                candidates = [t["Temps"] for t in timeline if t["Note"] == target_key and t["Conf"] >= 0.84]
                if candidates:
                    modulation_time = min(candidates)
                else:
                    sorted_times    = sorted(target_times)
                    modulation_time = sorted_times[max(0, len(sorted_times) // 3)]

                total_valid = len(timeline)
                if total_valid > 0:
                    target_count      = sum(1 for t in timeline if t["Note"] == target_key)
                    target_percentage = (target_count / total_valid) * 100

                if timeline:
                    last_n        = max(5, len(timeline) // 10)
                    last_segments = timeline[-last_n:]
                    last_counter  = Counter(s["Note"] for s in last_segments)
                    last_key      = last_counter.most_common(1)[0][0]
                    ends_in_target = (last_key == target_key)

        top_votes       = [v for k, v in most_common if v > total_votes * 0.1]
        stability_index = 100 / len(top_votes) if top_votes else 0
        is_unstable     = len(top_votes) >= 3

        total_duration    = duration
        dynamic_threshold = min(total_duration * 0.20, 60)

        raw_final_conf    = final_conf
        raw_dominant_conf = dominant_conf
        final_power  = raw_final_conf    * np.sqrt(max(final_percentage, 0))
        dom_power    = raw_dominant_conf * np.sqrt(max(dominant_percentage, 0))
        power_ratio  = dom_power / final_power if final_power > 0 else 0

        est_fantome           = (final_percentage < 5.0 and dominant_percentage > 20.0)
        domination_statistique = (power_ratio > 1.25)

        decision_pivot = None
        arb_dist_num   = 99
        arb_dist_mode  = 99
        arb_type       = "NONE"
        if final_conf >= 70 and dominant_conf >= 70:
            arb_result = arbitrage_expert_universel(
                chroma_avg, bass_global, final_key, dominant_key,
                CAMELOT_MAP, y_harm, sr, tuning=tuning
            )
            if arb_result["duel_actif"]:
                decision_pivot = arb_result["key"]
                arb_dist_num   = arb_result["dist_num"]
                arb_dist_mode  = arb_result["dist_mode"]
                arb_type       = arb_result.get("type", "SPECTRAL")

        mod_threshold   = 20.0 if mod_detected else 25.0
        mod_confirmed   = (mod_detected and target_percentage >= mod_threshold)

        if final_key == dominant_key and final_conf >= 70 and final_percentage >= 30:
            confiance_pure_key = final_key
            avis_expert        = f"💎 VERROUILLAGE STATISTIQUE"
            color_bandeau      = "linear-gradient(135deg, #10b981, #059669)"

        elif power_ratio > 1.25 or (decision_pivot and power_ratio > 1.10):
            confiance_pure_key = dominant_key
            avis_expert        = f"⚡ FORCE SUPRÊME ({round(dominant_percentage, 1)}%)"
            color_bandeau      = "linear-gradient(135deg, #7c3aed, #4c1d95)"

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
            color_bandeau = "linear-gradient(135deg, #0369a1, #0c4a6e)"

        elif (mod_confirmed and ends_in_target and target_key is not None
              and modulation_time is not None and modulation_time <= dynamic_threshold):
            confiance_pure_key = target_key
            avis_expert        = f"🏁 MODULATION VALIDÉE ({round(modulation_time)}s / {round(total_duration)}s)"
            color_bandeau      = "linear-gradient(135deg, #4338ca, #1e1b4b)"

        elif final_key == dominant_key and final_conf >= 85:
            confiance_pure_key = final_key
            avis_expert        = f"💎 ACCORD PARFAIT"
            color_bandeau      = "linear-gradient(135deg, #059669, #064e3b)"

        else:
            if dom_power > (final_power * 1.2):
                confiance_pure_key = dominant_key
                avis_expert        = f"✅ STABILITÉ DOMINANTE ({round(dominant_percentage, 1)}%)"
                color_bandeau      = "linear-gradient(135deg, #065f46, #064e3b)"
            else:
                confiance_pure_key = final_key
                avis_expert        = f"✅ ANALYSE STABLE"
                color_bandeau      = "linear-gradient(135deg, #065f46, #064e3b)"

        # --- CONSTRUCTION DU RAPPORT TELEGRAM ---
        if TELEGRAM_TOKEN and CHAT_ID:
            try:
                mod_line = ""
                if mod_detected:
                    perc    = round(target_percentage, 1)
                    end_txt = " 🏁 *FIN SUR MODULATION*" if ends_in_target else ""
                    mod_line = (
                        f"\n⚠️ *MODULATION →* `{target_key.upper()} ({get_exact_camelot(target_key)})`"
                        f" | ≈ *{seconds_to_mmss(modulation_time)}*"
                        f" | *PRÉSENCE:* `{perc}%`"
                        f" | *CONFIANCE:* `{target_conf}%`"
                        f"{end_txt}"
                    )

                dom_line  = (
                    f"\n🏆 *TONALITÉ DOMINANTE:* `{dominant_key.upper()} ({dominant_camelot})`"
                    f" | *PRÉSENCE:* `{round(dominant_percentage, 1)}%`"
                    f" | *CONFIANCE:* `{dominant_conf}%`"
                )

                consonance_line = (
                    f"\n🎹 *TONALITÉ DE LA CONSONANCE:* `{final_key.upper()} ({get_exact_camelot(final_key)})`"
                    f" | *PRÉSENCE:* `{round(final_percentage, 1)}%`"
                    f" | *CONFIANCE:* `{min(int(raw_final_conf), 100)}%`"
                )

                modal_emojis = {"major": "🟢 (Majeur)", "minor": "🔵 (Mineur)"}
                modal_emoji  = modal_emojis.get(modal_raw_mode, "⚪")
                modal_line   = (
                    f"\n🎼 *MODE DÉTECTÉ :* {modal_emoji}"
                    f"\n└ `{modal_key.upper()} ({modal_camelot})`"
                )

                tuning_display = "440.0 Hz (corrigé automatiquement)" if correction_applied else f"{round(440 * (2 ** (tuning / 1200)), 1)} Hz"
                accordage_line = f"🎸 *ACCORDAGE :* `{tuning_display}` ✅\n"

                # --- DÉTECTION DE CHOC HARMONIQUE ---
                choc_harmonique_alerte = ""
                is_dissonant = False

                if final_key != dominant_key:
                    cam_final = get_exact_camelot(final_key)
                    cam_dom = get_exact_camelot(dominant_key)
                    
                    if cam_final not in ("??", "") and cam_dom not in ("??", ""):
                        try:
                            cam_final_num = int(cam_final[:-1])
                            cam_dom_num = int(cam_dom[:-1])
                            diff = abs(cam_final_num - cam_dom_num)
                            if diff > 1 and diff < 11:
                                is_dissonant = True
                                choc_harmonique_alerte = (
                                    f"\n⚠️ *CHOC HARMONIQUE DÉTECTÉ:*\n"
                                    f"└ Le moteur indique `{cam_final}`, mais l'énergie brute frappe en `{cam_dom}`. "
                                    f"Risque de dissonance lors du mixage."
                                )
                        except (ValueError, IndexError):
                            pass

                # --- TONALITÉ VERROUILLÉE PAR L'ALGORITHME ---
                pure_camelot_tg = get_exact_camelot(confiance_pure_key)
                verrou_emoji = "🔴" if is_dissonant else "🟢"
                verrou_line = (
                    f"\n🔒 *TONALITÉ VERROUILLÉE:* `{confiance_pure_key.upper()} ({pure_camelot_tg})`"
                    f"  {verrou_emoji}"
                    f"\n└ 🤖 *AVIS EXPERT:* _{avis_expert}_"
                )

                # --- CONSENSUS HARMONIQUE (systématique) ---
                cam_moteur   = get_exact_camelot(final_key)
                cam_energie  = get_exact_camelot(dominant_key)
                if final_key == dominant_key:
                    consensus_statut = "✅ ACCORD PARFAIT"
                else:
                    consensus_statut = "⚠️ DIVERGENCE" if is_dissonant else "🔀 LÉGÈRE DIFFÉRENCE"

                consensus_line = (
                    f"\n🧠 *CONSENSUS HARMONIQUE :* {consensus_statut}"
                    f"\n├ Moteur (consonance) : `{final_key.upper()} ({cam_moteur})`"
                    f"  — Confiance : `{min(int(raw_final_conf), 100)}%`"
                    f"\n└ Énergie brute (dominante) : `{dominant_key.upper()} ({cam_energie})`"
                    f"  — Confiance : `{dominant_conf}%`"
                )

                # --- FRÉQUENCE DOMINANTE (systématique) ---
                freq_dominante_line = (
                    f"\n🔊 *FRÉQUENCE DOMINANTE :*"
                    f"\n├ Clé : `{dominant_key.upper()} ({cam_energie})`"
                    f"\n├ Présence : `{round(dominant_percentage, 1)}%`"
                    f"\n└ Force (dom_power) : `{round(dom_power, 1)}`"
                )

                caption = (
                    f"🎯 *RCDJ228 MUSIC SNIPER*\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"📂 *FICHIER:* `{file_name}`\n"
                    + accordage_line
                    + verrou_line
                    + "\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    + consonance_line
                    + dom_line
                    + modal_line
                    + mod_line
                    + choc_harmonique_alerte
                    + "\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    + consensus_line
                    + "\n"
                    + freq_dominante_line
                    + "\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"🛡️ *SECTION HARMONIQUE:* {seconds_to_mmss(harm_start)} → {seconds_to_mmss(harm_end)}"
                )

                # Création des graphiques
                CAMELOT_LABELS_TG = [get_exact_camelot(f"{n} major") for n in NOTES_LIST]
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1),
                    theta=CAMELOT_LABELS_TG, fill='toself', line_color='#10b981'
                ))
                fig_radar.update_layout(
                    template="plotly_dark",
                    title="SPECTRE HARMONIQUE (Camelot)",
                    polar=dict(radialaxis=dict(visible=False))
                )

                # Timeline (optionnelle)
                fig_tl = None
                df_tl = pd.DataFrame(timeline)
                if not df_tl.empty:
                    CAMELOT_ORDER_TG = [f"{i}{m}" for i in range(1, 13) for m in ['A', 'B']]
                    fig_tl = px.line(
                        df_tl, x="Temps", y="Camelot", markers=True,
                        template="plotly_dark",
                        category_orders={"Camelot": CAMELOT_ORDER_TG},
                        hover_data={"Note": True, "Temps": ":.2f"},
                        title="ÉVOLUTION TEMPORELLE (Camelot)"
                    )

                # Envoi du rapport Telegram avec la fonction corrigée
                send_telegram_report(caption, fig_radar, fig_tl, file_name)

            except Exception as e:
                st.error(f"Erreur lors de la construction du rapport Telegram : {e}")
                st.error(traceback.format_exc())

        update_prog(100, "Analyse terminée")
        status_text.empty()
        progress_bar.empty()

        final_tuning_hz = 440.0 if correction_applied else round(440 * (2 ** (tuning / 1200)), 1)

        res_obj = {
            "key": final_key,
            "camelot": get_exact_camelot(final_key),
            "conf": min(int(raw_final_conf), 100),
            "tuning": final_tuning_hz,
            "correction_applied": correction_applied,
            "correction_info": correction_info,
            "modulation": mod_detected,
            "target_key": target_key,
            "target_camelot": get_exact_camelot(target_key) if target_key else None,
            "name": file_name,
            "modulation_time_str": seconds_to_mmss(modulation_time) if mod_detected else None,
            "mod_target_percentage": round(target_percentage, 1) if mod_detected else 0,
            "mod_ends_in_target": ends_in_target if mod_detected else False,
            "harm_start": seconds_to_mmss(harm_start),
            "harm_end":   seconds_to_mmss(harm_end),
            "target_conf": target_conf,
            "dominant_key": dominant_key,
            "dominant_camelot": dominant_camelot,
            "dominant_conf": dominant_conf,
            "dominant_percentage": round(dominant_percentage, 1),
            "key_presence": round(final_percentage, 1),
            "duration_detected": round(total_duration, 1),
            "final_power": round(final_power, 1),
            "dom_power":   round(dom_power, 1),
            "power_ratio": round(power_ratio, 2),
            "confiance_pure": confiance_pure_key,
            "confiance_pure_key": confiance_pure_key,
            "pure_camelot": get_exact_camelot(confiance_pure_key),
            "avis_expert": avis_expert,
            "color_bandeau": color_bandeau,
            "modal_key": modal_key,
            "modal_camelot": modal_camelot,
            "modal_raw_mode": modal_raw_mode,
            "modal_conf": modal_conf,
            "modal_presence": round(modal_presence, 1),
            "stability_score": round(stability_index, 1),
            "is_unstable": is_unstable,
            "timeline_data": timeline,
            "chroma_data": chroma_avg,
        }

        del y, y_filt, full_chroma_raw
        gc.collect()
        return res_obj

    except Exception as e:
        raise e

# ══════════════════════════════════════════════════════════════════════════
# --- UTILITAIRES UI ---
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

# ══════════════════════════════════════════════════════════════════════════
# --- DASHBOARD PRINCIPAL ---
st.title("🎯 RCDJ228 MUSIC SNIPER")
st.markdown("#### Système d'Analyse Harmonique Modale — Majeur / Mineur × 12 Toniques + **Correction Auto Accordage**")

global_status = st.empty()

uploaded_files = st.file_uploader(
    "📥 Déposez vos fichiers (Audio)",
    type=['mp3', 'wav', 'flac', 'm4a'],
    accept_multiple_files=True,
    key="file_uploader"
)

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
                old_data = st.session_state.analyses[oldest_file]
                del st.session_state.analyses[oldest_file]

        if file_name in st.session_state.analyses:
            analysis_data = st.session_state.analyses[file_name]

            timeline = analysis_data['timeline_data']
            chroma = analysis_data['chroma_data']

            with st.container():
                st.markdown(
                    f"<div class='file-header'>📂 ANALYSE : {analysis_data['name']}</div>",
                    unsafe_allow_html=True
                )

                st.subheader("🎯 Résultat du Sniper")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="Tonalité Arbitrée (Finale)",
                        value=analysis_data["confiance_pure_key"],
                        delta=f"{analysis_data['pure_camelot']}"
                    )

                with col2:
                    st.metric(
                        label="Consonance (Moteur)",
                        value=analysis_data["key"],
                        delta=f"{analysis_data['camelot']} | {analysis_data.get('conf', 0)}%"
                    )

                with col3:
                    st.metric(
                        label="Dominante (Moteur)",
                        value=analysis_data["dominant_key"],
                        delta=f"{analysis_data['dominant_camelot']} | {analysis_data.get('dominant_conf', 0)}%"
                    )
                
                with col4:
                    st.metric(
                        label="Fiabilité Globale",
                        value=f"{analysis_data.get('conf', 0)}%"
                    )

                # --- NOUVELLE CASE : TONALITÉ VERROUILLÉE PAR L'ALGORITHME ---
                st.markdown("---")
                st.subheader("🔒 DÉCISION FINALE DE L'ALGORITHME")

                verrou_col1, verrou_col2 = st.columns([3, 2])

                with verrou_col1:
                    # Style pour la case verrouillée
                    border_color = "#10b981"  # Vert par défaut
                    bg_color = "rgba(16, 185, 129, 0.1)"
                    
                    # Si choc harmonique, changer la couleur
                    if analysis_data["key"] != analysis_data["dominant_key"]:
                        _cam_f_str = analysis_data.get("camelot", "??")
                        _cam_d_str = analysis_data.get("dominant_camelot", "??")
                        if _cam_f_str not in ("??", "") and _cam_d_str not in ("??", ""):
                            try:
                                _cam_f = int(_cam_f_str[:-1])
                                _cam_d = int(_cam_d_str[:-1])
                                _diff = abs(_cam_f - _cam_d)
                                if _diff > 1 and _diff < 11:
                                    border_color = "#ef4444"  # Rouge pour choc
                                    bg_color = "rgba(239, 68, 68, 0.1)"
                            except (ValueError, IndexError):
                                pass
                    
                    confiance_pure = analysis_data.get("confiance_pure_key", "Unknown")
                    pure_camelot = analysis_data.get("pure_camelot", "??")
                    
                    st.markdown(
                        f"""
                        <div style="
                            background: {bg_color};
                            border: 3px solid {border_color};
                            border-radius: 15px;
                            padding: 20px;
                            text-align: center;
                            transition: all 0.3s ease;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                        ">
                            <span style="font-size: 0.9em; color: #94a3b8;">🎯 TONALITÉ VERROUILLÉE</span><br>
                            <span style="font-size: 2.5em; font-weight: 900; color: {border_color}; 
                                         font-family: 'JetBrains Mono', monospace;">
                                {confiance_pure.upper()}
                            </span><br>
                            <span style="font-size: 1.8em; font-weight: bold; color: #94a3b8;">
                                🏷️ {pure_camelot}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with verrou_col2:
                    avis = analysis_data.get("avis_expert", "")
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(88, 166, 255, 0.1);
                            border: 2px solid #58a6ff;
                            border-radius: 15px;
                            padding: 20px;
                            height: 100%;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        ">
                            <span style="font-size: 0.9em; color: #94a3b8;">🤖 AVIS EXPERT</span><br>
                            <span style="font-size: 1.2em; font-weight: bold; color: #58a6ff;">
                                {avis}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Affichage détaillé si conflit
                if analysis_data["key"] != analysis_data["dominant_key"]:
                    st.markdown("---")
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        st.markdown(
                            f"""
                            <div style="
                                background: rgba(16, 185, 129, 0.1);
                                border: 1px solid #10b981;
                                border-radius: 10px;
                                padding: 15px;
                                text-align: center;
                            ">
                                <span style="color: #10b981;">✅ CONSENSUS HARMONIQUE</span><br>
                                <span style="font-size: 1.3em; font-weight: bold;">
                                    {analysis_data['key']} ({analysis_data['camelot']})
                                </span><br>
                                <span style="font-size: 0.8em; color: #94a3b8;">
                                    Confiance: {analysis_data.get('conf', 0)}%
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with detail_col2:
                        st.markdown(
                            f"""
                            <div style="
                                background: rgba(124, 58, 237, 0.1);
                                border: 1px solid #7c3aed;
                                border-radius: 10px;
                                padding: 15px;
                                text-align: center;
                            ">
                                <span style="color: #7c3aed;">⚡ FRÉQUENCE DOMINANTE</span><br>
                                <span style="font-size: 1.3em; font-weight: bold;">
                                    {analysis_data['dominant_key']} ({analysis_data['dominant_camelot']})
                                </span><br>
                                <span style="font-size: 0.8em; color: #94a3b8;">
                                    Présence: {analysis_data.get('dominant_percentage', 0)}%
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with detail_col3:
                        verdict = "🔀 CONFLIT DÉTECTÉ"
                        verdict_color = "#f59e0b"
                        
                        if analysis_data.get("confiance_pure_key") == analysis_data["dominant_key"]:
                            verdict = "⚡ DOMINATION FRÉQUENTIELLE"
                            verdict_color = "#7c3aed"
                        elif analysis_data.get("confiance_pure_key") == analysis_data["key"]:
                            verdict = "💎 STABILITÉ HARMONIQUE"
                            verdict_color = "#10b981"
                        
                        # Correction: conversion correcte de l'hex en RGB pour le rgba
                        try:
                            verdict_r = int(verdict_color[1:3], 16)
                            verdict_g = int(verdict_color[3:5], 16)
                            verdict_b = int(verdict_color[5:7], 16)
                            verdict_bg = f"rgba({verdict_r}, {verdict_g}, {verdict_b}, 0.1)"
                        except:
                            verdict_bg = "rgba(245, 158, 11, 0.1)"
                        
                        st.markdown(
                            f"""
                            <div style="
                                background: {verdict_bg};
                                border: 1px solid {verdict_color};
                                border-radius: 10px;
                                padding: 15px;
                                text-align: center;
                            ">
                                <span style="color: {verdict_color};">🎯 VERDICT FINAL</span><br>
                                <span style="font-size: 1em; font-weight: bold; color: {verdict_color};">
                                    {verdict}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Si choc harmonique, afficher l'alerte détaillée
                if analysis_data["key"] != analysis_data["dominant_key"]:
                    _cam_f_str = analysis_data.get("camelot", "??")
                    _cam_d_str = analysis_data.get("dominant_camelot", "??")
                    if _cam_f_str not in ("??", "") and _cam_d_str not in ("??", ""):
                        try:
                            _cam_f = int(_cam_f_str[:-1])
                            _cam_d = int(_cam_d_str[:-1])
                            _diff = abs(_cam_f - _cam_d)
                            if _diff > 1 and _diff < 11:
                                st.markdown(
                                    f"""
                                    <div style="
                                        background: rgba(239, 68, 68, 0.15);
                                        border: 2px solid #ef4444;
                                        border-radius: 15px;
                                        padding: 20px;
                                        margin-top: 15px;
                                    ">
                                        <span style="font-size: 1.5em;">⚠️</span>
                                        <span style="font-size: 1.2em; font-weight: bold; color: #fca5a5;">
                                            CHOC HARMONIQUE DÉTECTÉ
                                        </span><br><br>
                                        <span style="color: #fca5a5; line-height: 1.6;">
                                            L'algorithme a verrouillé <b style="color: #10b981;">{analysis_data['camelot']}</b> 
                                            grâce à la structure mélodique, mais une fréquence parasite massive sature en 
                                            <b style="color: #7c3aed;">{analysis_data['dominant_camelot']}</b>.<br>
                                            <span style="font-size: 0.9em; color: #94a3b8;">
                                                → Soyez prudent sur l'égalisation des basses lors de la transition.
                                            </span>
                                        </span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        except (ValueError, IndexError):
                            pass

                # --- AFFICHAGE SYSTÉMATIQUE DU CONSENSUS ET DE LA FRÉQUENCE ---
                tonalite_moteur = analysis_data.get('camelot', '??')
                note_brute = analysis_data.get('dominant_key', 'Inconnue')
                freq_brute = analysis_data.get('dominant_percentage', 0)
                st.info(
                    f"🔍 **Analyse approfondie** : Le moteur de tonalité indique **{tonalite_moteur}**, "
                    f"et l'énergie brute frappe en **{note_brute}**."
                )
                col_cons, col_freq = st.columns(2)
                with col_cons:
                    st.markdown("**🧠 Consensus Harmonique**")
                    df_consensus = pd.DataFrame({
                        "Moteur (Camelot)": [tonalite_moteur],
                        "Énergie Brute": [note_brute],
                        "Confiance Moteur": [f"{analysis_data.get('conf', 0)}%"],
                        "Confiance Brute": [f"{analysis_data.get('dominant_conf', 0)}%"],
                    })
                    st.dataframe(df_consensus, use_container_width=True)
                with col_freq:
                    st.markdown("**🔊 Fréquence Dominante (Top 1)**")
                    df_freq = pd.DataFrame({
                        "Clé Dominante": [note_brute],
                        "Camelot Dominant": [analysis_data.get('dominant_camelot', '??')],
                        "Présence (%)": [f"{freq_brute}%"],
                    })
                    st.dataframe(df_freq, use_container_width=True)

                if analysis_data.get("correction_applied", False):
                    st.success(f"🎸 **{analysis_data.get('correction_info', 'Accordage corrigé automatiquement à 440 Hz')}**")
                else:
                    st.info(f"🎸 Accordage détecté : {analysis_data.get('tuning', 440.0)} Hz")

                if analysis_data.get('modulation', False):
                    secondary_key = analysis_data.get('target_key')
                    if secondary_key:
                        with st.expander("🔄 Modulation détectée"):
                            camelot_sec = analysis_data.get('target_camelot') or CAMELOT_MAP.get(secondary_key, "??")
                            st.write(f"Le morceau semble changer vers **{secondary_key.upper()}** ({camelot_sec}) vers la fin.")
                            st.info("Conseil : Idéal pour une transition progressive.")

                if analysis_data.get('is_unstable'):
                    st.markdown(
                        f"<div style='background:rgba(245,158,11,0.12); border:1px solid #f59e0b; border-radius:15px;"
                        f"padding:14px 20px; margin-bottom:12px; font-family:JetBrains Mono,monospace; color:#fbbf24;'>"
                        f"⚠️ <b>ALERTE INSTABILITÉ</b> — Indice de stabilité : <b>{analysis_data.get('stability_score',0)}</b>"
                        f" &nbsp;|&nbsp; Ce morceau change fréquemment de structure harmonique."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                m2, m3 = st.columns(2)
                with m2:
                    st.markdown(
                        f"<div class='metric-box'><b>ACCORDAGE</b><br>"
                        f"<span style='font-size:2em; color:#58a6ff;'>{analysis_data['tuning']}</span><br>Hz</div>",
                        unsafe_allow_html=True
                    )
                with m3:
                    btn_id = f"play_{hash(analysis_data['name'])}"
                    components.html(f"""
                        <button id="{btn_id}" style="width:100%; height:95px;
                            background:linear-gradient(45deg,#4F46E5,#7C3AED); color:white;
                            border:none; border-radius:15px; cursor:pointer; font-weight:bold;">
                            🎹 TESTER L'ACCORD
                        </button>
                        <script>{get_chord_js(btn_id, analysis_data['key'])}</script>
                    """, height=110)

                raw_mode = analysis_data.get('modal_raw_mode', 'major')
                modal_colors = {"major": "#10b981", "minor": "#3b82f6"}
                modal_descriptions = {
                    "major": "Majeur classique — lumineux, stable",
                    "minor": "Mineur naturel — mélancolique, profond",
                }
                mc = modal_colors.get(raw_mode, "#6b7280")
                md = modal_descriptions.get(raw_mode, "")
                st.markdown(
                    f"<div class='metric-box' style='border-color:{mc}; margin-bottom:12px;'>"
                    f"<b>🎼 TONALITÉ DÉTECTÉE</b><br>"
                    f"<span style='font-size:1.8em; color:{mc}; font-weight:900;'>"
                    f"{analysis_data.get('modal_key','—').upper()}</span>"
                    f"&nbsp;&nbsp;<span style='font-size:1em; color:#94a3b8;'>"
                    f"Camelot : <b>{analysis_data.get('modal_camelot','??')}</b></span><br>"
                    f"<div style='font-size:0.9em; margin-top:5px; color:#94a3b8;'>"
                    f"🎯 CONFIANCE : <b>{analysis_data.get('modal_conf',0)}%</b>"
                    f" &nbsp;|&nbsp; 📊 PRÉSENCE : <b>{analysis_data.get('modal_presence',0)}%</b>"
                    f"</div>"
                    f"<span style='font-size:0.75em; color:#94a3b8; font-style:italic;'>{md}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                ps1, ps2, ps3 = st.columns(3)
                with ps1:
                    st.markdown(
                        f"<div class='metric-box'><b>💪 FORCE CONSONANCE</b><br>"
                        f"<span style='font-size:1.6em; color:#a78bfa;'>{analysis_data.get('final_power','—')}</span></div>",
                        unsafe_allow_html=True
                    )
                with ps2:
                    st.markdown(
                        f"<div class='metric-box'><b>💪 FORCE DOMINANTE</b><br>"
                        f"<span style='font-size:1.6em; color:#a78bfa;'>{analysis_data.get('dom_power','—')}</span></div>",
                        unsafe_allow_html=True
                    )
                with ps3:
                    ratio_val   = analysis_data.get('power_ratio', 0)
                    ratio_color = "#ef4444" if ratio_val > 1.25 else "#f59e0b" if ratio_val > 1.10 else "#10b981"
                    st.markdown(
                        f"<div class='metric-box'><b>📊 RATIO DE PUISSANCE</b><br>"
                        f"<span style='font-size:1.6em; color:{ratio_color};'>{ratio_val}</span></div>",
                        unsafe_allow_html=True
                    )

                c1, c2 = st.columns([2, 1])
                with c1:
                    df_tl = pd.DataFrame(timeline)
                    if not df_tl.empty:
                        CAMELOT_ORDER = [f"{i}{m}" for i in range(1, 13) for m in ['A', 'B']]
                        fig_tl = px.line(
                            df_tl, x="Temps", y="Camelot", markers=True,
                            template="plotly_dark",
                            category_orders={"Camelot": CAMELOT_ORDER},
                            hover_data={"Note": True, "Temps": ":.2f"}
                        )
                        fig_tl.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=30, b=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_tl, use_container_width=True, key=f"timeline_{analysis_data['name']}_{i}")
                    else:
                        st.warning("⚠️ Pas assez de données harmoniques pour générer la timeline.")
                with c2:
                    CAMELOT_LABELS = [get_exact_camelot(f"{n} major") for n in NOTES_LIST]
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=chroma, theta=CAMELOT_LABELS, fill='toself', line_color='#10b981'
                    ))
                    fig_radar.update_layout(
                        template="plotly_dark", height=300,
                        margin=dict(l=40, r=40, t=30, b=20),
                        polar=dict(radialaxis=dict(visible=False)),
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{analysis_data['name']}_{i}")

                st.markdown("<hr style='border-color:#30363d; margin-bottom:40px;'>", unsafe_allow_html=True)

            del timeline, chroma
            gc.collect()

    st.session_state.analyzing = False
    global_status.success("Tous les fichiers ont été analysés avec succès !")
    gc.collect()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Sniper Control")
    st.markdown("---")
    st.success("✅ **Correction Auto Accordage 440 Hz** activée — zéro erreur de tonalité")
    st.success("✅ **Nettoyage automatique activé**")

    if st.button("🧹 Vider la file d'analyse"):
        st.session_state.analyses = {}
        st.session_state.analyzing = False
        gc.collect()
        st.rerun()
