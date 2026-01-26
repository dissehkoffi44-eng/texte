import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import os

# ────────────────────────────────────────────────
#                CONFIGURATION & PATHS
# ────────────────────────────────────────────────

# Note : On ne force le path ffmpeg que s'il existe localement
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

st.set_page_config(page_title="DJ's Ear Pro Elite", page_icon="🎼", layout="wide")

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
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# ────────────────────────────────────────────────
#                     STYLES
# ────────────────────────────────────────────────

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
    </style>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────
#              FONCTIONS DE FILTRAGE
# ────────────────────────────────────────────────

def filter_original(y, sr):
    y_harm, _ = librosa.effects.hpss(y, margin=(4.0, 1.0))
    y_harm = librosa.effects.preemphasis(y_harm)
    nyq = 0.5 * sr
    b, a = butter(4, [100/nyq, 3000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def filter_sniper(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    low, high = 80 / nyq, 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

# ────────────────────────────────────────────────
#              DÉTECTION TONALITÉ
# ────────────────────────────────────────────────

def solve_key(chroma_vector, global_dom_root=None):
    best_score, best_key = -np.inf, "Inconnu"
    try:
        cv = np.asarray(chroma_vector, dtype=np.float64).flatten()
        if cv.size != 12: return {"key": "Erreur dim", "score": 0.0}
    except: return {"key": "Erreur conv", "score": 0.0}

    cv_min, cv_max = cv.min(), cv.max()
    if cv_max <= cv_min + 1e-12: return {"key": "Silence", "score": 0.0}
    cv = (cv - cv_min) / (cv_max - cv_min + 1e-10)

    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            profile = np.asarray(p_data[mode], dtype=np.float64)
            for shift in range(12):
                rotated = np.roll(profile, shift)
                corr = np.corrcoef(cv, rotated)[0, 1]
                if not np.isfinite(corr): corr = -1.0

                third_idx = (shift + (3 if mode == "minor" else 4)) % 12
                fifth_idx = (shift + 7) % 12
                
                bonus = 0.18 if (global_dom_root is not None and (shift + 7) % 12 == global_dom_root and cv[global_dom_root] > 0.35) else 0.0
                score = corr + 0.15 * cv[third_idx] + 0.05 * cv[fifth_idx] + bonus

                if score > best_score:
                    best_score, best_key = score, f"{NOTES_LIST[shift]} {mode}"
    return {"key": best_key, "score": float(best_score)}

# ────────────────────────────────────────────────
#               ANALYSE ENGINE
# ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name, filter_type="original"):
    try:
        ext = file_name.rsplit('.', 1)[-1].lower()
        if ext == 'm4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2: samples = samples.reshape((-1, 2)).mean(axis=1)
            y, sr = samples / (1 << 15), audio.frame_rate
            if sr != 22050: 
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=22050, mono=True)
    except Exception: return None

    if len(y) < 22050: return None # Trop court

    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = filter_sniper(y, sr) if filter_type == "sniper" else filter_original(y, sr)
    
    chroma_global = librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning, bins_per_octave=24)
    global_avg = np.mean(chroma_global, axis=1)
    
    top2 = np.argsort(global_avg)[-2:]
    dom_root = top2[1] if (top2[0] + 7) % 12 == top2[1] else (top2[0] if (top2[1] + 7) % 12 == top2[0] else None)

    duration = librosa.get_duration(y=y, sr=sr)
    step, timeline, votes = 2, [], Counter()

    for start_sec in range(0, max(1, int(duration) - step), step):
        seg = y_filt[int(start_sec * sr):int((start_sec + step) * sr)]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue

        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, bins_per_octave=24)
        c12 = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        
        res = solve_key(c12, global_dom_root=dom_root)
        if res['score'] < 0.70: continue # Seuil assoupli pour garantir un résultat

        w = 2.0 if start_sec < 10 or start_sec > (duration - 15) else 1.0
        votes[res['key']] += int(res['score'] * 100 * w)
        timeline.append({"Temps": start_sec, "Note": res['key'], "Conf": res['score']})

    if not votes: return None

    main_key = votes.most_common(1)[0][0]
    main_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == main_key]) * 100)
    
    mod_detected, target_key, target_conf = False, None, None
    if len(votes.most_common(2)) > 1:
        sec_key, sec_val = votes.most_common(2)[1]
        if (sec_val / sum(votes.values())) > 0.25:
            mod_detected, target_key = True, sec_key
            target_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == sec_key]) * 100)

    tempo, _ = librosa.beat.beat_track(y=librosa.effects.percussive(y), sr=sr)

    return {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "conf": min(main_conf, 99), "tempo": int(float(tempo or 0)),
        "tuning_hz": round(440 * (2 ** (tuning / 12)), 1), "pitch_offset": round(tuning, 2),
        "timeline": timeline, "chroma": global_avg.tolist(),
        "modulation": mod_detected, "target_key": target_key, "target_conf": target_conf,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name
    }

# ────────────────────────────────────────────────
#                UTILITAIRES UI
# ────────────────────────────────────────────────

def get_piano_js(btn_id, key_name):
    if not key_name or " " not in key_name: return ""
    root, mode = key_name.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0,3,7,12] : [0,4,7,12];
        intervals.forEach(i => {{
            const base = freqs['{root}'] * Math.pow(2, i/12);
            [1,2].forEach(h => {{
                const osc = ctx.createOscillator(); const gain = ctx.createGain();
                osc.type = h===1 ? 'triangle' : 'sine';
                osc.frequency.setValueAtTime(base * h, ctx.currentTime);
                gain.gain.setValueAtTime(0, ctx.currentTime);
                gain.gain.linearRampToValueAtTime(0.1/h, ctx.currentTime + 0.05);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
                osc.connect(gain); gain.connect(ctx.destination);
                osc.start(); osc.stop(ctx.currentTime + 1.5);
            }});
        }});
    }};
    """

def send_telegram_report(data, fig_line, fig_polar):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        caption = f"🎼 *Report: {data['name']}*\nKey: `{data['key']}` ({data['camelot']})\nTempo: `{data['tempo']} BPM`"
        img_tl = fig_line.to_image(format="png", engine="kaleido")
        img_rd = fig_polar.to_image(format="png", engine="kaleido")
        media = [{'type': 'photo', 'media': 'attach://a.png', 'caption': caption, 'parse_mode': 'Markdown'}, {'type': 'photo', 'media': 'attach://b.png'}]
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup", data={'chat_id': CHAT_ID, 'media': json.dumps(media)}, files={'a.png': img_tl, 'b.png': img_rd}, timeout=15)
    except: pass

# ────────────────────────────────────────────────
#                INTERFACE PRINCIPALE
# ────────────────────────────────────────────────

st.title("🎧 DJ's Ear Pro Elite")

with st.sidebar:
    filter_mode = st.radio("Style de filtrage", ["Original", "Sniper (Recommandé)"], index=1)
    filter_type = "sniper" if "Sniper" in filter_mode else "original"
    if st.button("🧹 Vider le cache"):
        st.cache_data.clear()
        st.rerun()

uploaded_files = st.file_uploader("Audio (MP3, WAV, M4A)", type=['mp3', 'wav', 'm4a', 'flac'], accept_multiple_files=True)

if uploaded_files:
    for idx, file in enumerate(reversed(uploaded_files)):
        container = st.container()
        with container:
            with st.status(f"Analyse de {file.name}...", expanded=True) as status:
                data = analyze_full_engine(file.getvalue(), file.name, filter_type)
                if data:
                    status.update(label=f"✅ {file.name} analysé", state="complete", expanded=False)
                else:
                    status.update(label=f"❌ Erreur sur {file.name}", state="error")
                    st.error("Données harmoniques introuvables. Vérifiez le volume du fichier.")
                    continue

            # Affichage UI
            bg = "linear-gradient(135deg, #0f172a, #1e3a8a)" if not data['modulation'] else "linear-gradient(135deg, #1e1b4b, #7f1d1d)"
            st.markdown(f"""
                <div class="report-card" style="background:{bg};">
                    <p style="opacity:0.7;">TONALITÉ PRINCIPALE</p>
                    <h1 style="font-size:5em; margin:10px 0;">{data['key'].upper()}</h1>
                    <h3>CAMELOT {data['camelot']}  •  Confiance {data['conf']}%</h3>
                    {f"<div class='modulation-alert'>⚠️ MODULATION VERS {data['target_key']} ({data['target_camelot']})</div>" if data['modulation'] else ""}
                </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><b>TUNING</b><br><span style='font-size:2em;'>{data['tuning_hz']}</span><br>Hz</div>", unsafe_allow_html=True)
            with c3:
                bid = f"btn_{idx}"
                components.html(f'<button id="{bid}" style="width:100%; height:90px; background:#4F46E5; color:white; border-radius:12px; border:none; cursor:pointer; font-weight:bold;">🎹 TESTER L\'ACCORD</button><script>{get_piano_js(bid, data["key"])}</script>', height=100)

            gl, gr = st.columns([2, 1])
            df = pd.DataFrame(data['timeline'])
            fig_l = px.line(df, x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
            gl.plotly_chart(fig_l, use_container_width=True)

            fig_p = go.Figure(go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself'))
            fig_p.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            gr.plotly_chart(fig_p, use_container_width=True)

            send_telegram_report(data, fig_l, fig_p)
            st.divider()
else:
    st.info("En attente de fichiers...")
