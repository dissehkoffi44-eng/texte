# RCDJ228 MUSIC SNIPER M5-MAX - VERSION ULTIME (MOTEUR M5 + AGRESSIVITÉ M3)
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
from pydub import AudioSegment

# --- FORCE FFMEG PATH (WINDOWS) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER M5-MAX", page_icon="🎯", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "krumhansl": {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                  "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]},
    "temperley": {"major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
                  "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]},
    "aarden":    {"major": [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 18.6691],
                  "minor": [18.2648, 0.737619, 14.0499, 16.8599, 0.702699, 14.5212, 0.737619, 19.8145, 5.84214, 2.68046, 2.51091, 9.84455]},
    "bellman":   {"major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
                  "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]}
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(16, 185, 129, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 24px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 12px 24px; border-radius: 12px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; border-left: 6px solid #10b981;
    }
    .metric-box {
        background: #161b22; border-radius: 16px; padding: 20px; text-align: center;
        border: 1px solid #30363d; height: 100%;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.18); color: #f87171; padding: 16px; border-radius: 16px;
        border: 1px solid #ef4444; margin: 20px 0; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE FILTRAGE AGRESSIFS (M3) ---
def apply_super_sniper_filters(y, sr):
    # Séparation Harmonique Agressive
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    # Filtre Passe-Bande 80Hz - 5000Hz (Précision M3)
    nyq = 0.5 * sr
    low, high = 80 / nyq, 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority_m3(y, sr):
    nyq = 0.5 * sr
    b, a = butter(2, 150/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    return np.mean(librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12), axis=1)

# --- MOTEUR DE VOTE HYBRIDE (M5) ---
def solve_hybrid_key(chroma_cqt, chroma_cens, bass_vector):
    cv = (chroma_cqt - chroma_cqt.min()) / (chroma_cqt.max() - chroma_cqt.min() + 1e-9)
    cens = (chroma_cens - chroma_cens.min()) / (chroma_cens.max() - chroma_cens.min() + 1e-9)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-9)
    
    scores = {f"{n} {m}": 0.0 for n in NOTES_LIST for m in ["major", "minor"]}
    for p_data in PROFILES.values():
        for mode in ["major", "minor"]:
            for i in range(12):
                # Hybridation CQT/CENS
                corr_cqt = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                corr_cens = np.corrcoef(cens, np.roll(p_data[mode], i))[0, 1]
                combined = 0.65 * corr_cqt + 0.35 * corr_cens
                
                # Bonus Basses + Quinte + Tierce
                bonus = (bv[i] * 0.40) + (cv[(i+7)%12] * 0.15)
                scores[f"{NOTES_LIST[i]} {mode}"] += (combined + bonus) / len(PROFILES)
    return scores

# --- CORE PROCESSOR ---
def process_audio_ultimate(file_bytes, file_name, progress_cb=None, threshold=0.78):
    sr_target = 22050
    try:
        # Support M4A via Pydub
        if file_name.lower().endswith('.m4a'):
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2: samples = samples.reshape((-1, 2)).mean(axis=1)
            y = samples / (1 << (8 * audio.sample_width - 1))
            sr = audio.frame_rate
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                sr = sr_target
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=sr_target, mono=True)
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_super_sniper_filters(y, sr)

    # Scan par segments avec stratégie M3 (Début/Fin prioritaires)
    seg_len, step = 6.0, 2.0
    segments = np.arange(0, max(0.1, duration - seg_len), step)
    votes = Counter()
    timeline = []
    retained_scores = []

    for idx, start in enumerate(segments):
        if progress_cb: progress_cb(int((idx/len(segments))*100), f"Scan {start:.1f}s / {duration:.1f}s")
        
        i_s, i_e = int(start * sr), int((start + seg_len) * sr)
        seg = y_filt[i_s:i_e]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue

        c_cqt = np.mean(librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning), axis=1)
        c_cens = np.mean(librosa.feature.chroma_cens(y=seg, sr=sr, tuning=tuning), axis=1)
        b_seg = get_bass_priority_m3(y[i_s:i_e], sr)
        
        scores = solve_hybrid_key(c_cqt, c_cens, b_seg)
        best_k = max(scores, key=scores.get)
        conf = scores[best_k]

        if conf >= threshold:
            # STRATÉGIE M3 : Boost début (15s) et Fin (20s)
            weight = 2.0 if (start < 15 or start > (duration - 20)) else 1.0
            
            # BOOST M5 : Confiance extrême
            if conf >= 0.88: weight *= 2.0
            elif conf >= 0.84: weight *= 1.5
            
            votes[best_k] += conf * weight
            timeline.append({"time": start, "key": best_k, "score": conf})
            retained_scores.append(conf)

    if not votes: return None

    # Résultats finaux
    final_key = votes.most_common(1)[0][0]
    confidence = min(99, int((np.mean(retained_scores) if retained_scores else 0) * 110))
    
    # Détection Modulation M5
    modulation = None
    if len(timeline) > 6:
        mid = len(timeline) // 2
        f_part = Counter(t["key"] for t in timeline[:mid])
        s_part = Counter(t["key"] for t in timeline[mid:])
        if f_part.most_common(1)[0][0] != s_part.most_common(1)[0][0]:
            modulation = s_part.most_common(1)[0][0]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    res = {
        "name": file_name, "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": confidence, "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
        "modulation": modulation, "target_camelot": CAMELOT_MAP.get(modulation, "??"),
        "chroma": np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1).tolist()
    }

    # Telegram (Optionnel)
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            caption = f"🎯 *SNIPER ULTIME*\nTrack: `{file_name}`\nKey: `{final_key.upper()}` ({res['camelot']})\nConf: `{confidence}%`"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                          data={'chat_id': CHAT_ID, 'text': caption, 'parse_mode': 'Markdown'})
        except: pass

    gc.collect()
    return res

# --- INTERFACE ---
st.title("🎯 RCDJ228 SNIPER M5-MAX")
files = st.file_uploader("Tracks", type=['mp3','wav','m4a','flac'], accept_multiple_files=True)

with st.sidebar:
    st.header("Paramètres Sniper")
    THR = st.slider("Seuil de précision", 0.60, 0.90, 0.78)

if files:
    for f in files:
        with st.status(f"Analyse {f.name}...") as status:
            data = process_audio_ultimate(f.getvalue(), f.name, threshold=THR)
            status.update(label="Analyse terminée", state="complete")
        
        if data:
            st.markdown(f"<div class='file-header'> {data['name']} </div>", unsafe_allow_html=True)
            bg = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 85 else "#161b22"
            st.markdown(f"""
                <div class="report-card" style="background:{bg};">
                    <h1 style="font-size:6em; margin:0;">{data['key'].upper()}</h1>
                    <h3>CAMELOT {data['camelot']} | CONFIANCE {data['conf']}%</h3>
                    {f"<div class='modulation-alert'>MODULATION DETECTÉE : {data['modulation']}</div>" if data['modulation'] else ""}
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("TEMPO", f"{data['tempo']} BPM")
            c2.metric("ACCORDAGE", f"{data['tuning']} Hz")
            with c3:
                # Bouton de test sonore JS
                btn_id = f"play_{hash(f.name)}"
                note, mode = data['key'].split()
                js_code = f"""
                document.getElementById('{btn_id}').onclick = function() {{
                    const ctx = new AudioContext();
                    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
                    const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
                    intervals.forEach(i => {{
                        const o = ctx.createOscillator(); const g = ctx.createGain();
                        o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
                        g.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.1);
                        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
                        o.connect(g); g.connect(ctx.destination); o.start(); o.stop(ctx.currentTime + 1.5);
                    }});
                }};
                """
                components.html(f'<button id="{btn_id}" style="width:100%;height:60px;background:#4F46E5;color:white;border-radius:10px;cursor:pointer;">TESTER L\'ACCORD</button><script>{js_code}</script>')
