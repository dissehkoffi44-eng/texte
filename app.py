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
import os

# --- FORCE FFMPEG PATH (pour Windows / certains hébergements) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="DJ's Ear Pro Music Elite", page_icon="🎼", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
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
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE TRAITEMENT ---

def apply_filters(y, sr):
    y_harm, _ = librosa.effects.hpss(y, margin=(4.0, 1.0))
    y_harm = librosa.effects.preemphasis(y_harm)
    nyq = 0.5 * sr
    b, a = butter(4, [100/nyq, 3000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def solve_key(chroma_vector, global_dom_root=None):
    best_score = -1
    best_key = "Inconnu"
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for profile_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                rotated_profile = np.roll(p_data[mode], i)
                corr_score = np.corrcoef(cv, rotated_profile)[0, 1]
                
                third_idx = (i + 3) % 12 if mode == "minor" else (i + 4) % 12
                fifth_idx = (i + 7) % 12
                
                dom_bonus = 0
                if global_dom_root is not None:
                    if (i + 7) % 12 == global_dom_root and cv[global_dom_root] > 0.35:
                        dom_bonus = 0.18

                final_score = corr_score + (0.15 * cv[third_idx]) + (0.05 * cv[fifth_idx]) + dom_bonus

                if final_score > best_score:
                    best_score = final_score
                    best_key = f"{NOTES_LIST[i]} {mode}"
    
    return {"key": best_key, "score": best_score}

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name, _progress_callback=None):
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
            with io.BytesIO(file_bytes) as b:
                y, sr = librosa.load(b, sr=22050, mono=True)
    except Exception as e:
        st.error(f"Erreur de lecture {file_name}: {e}")
        return None

    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_filters(y, sr)
    
    # Chroma global (on garde la version 24 bins classique pour la signature visuelle)
    chroma_complex = librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning, bins_per_octave=24, hop_length=512)
    global_chroma_avg = np.mean(chroma_complex, axis=1)
    
    top_2_idx = np.argsort(global_chroma_avg)[-2:]
    n_p, n_s = top_2_idx[1], top_2_idx[0]
    global_dom_root = n_s if (n_p + 7) % 12 == n_s else (n_p if (n_s + 7) % 12 == n_p else None)

    duration = librosa.get_duration(y=y, sr=sr)
    step = 2
    timeline = []
    votes = Counter()
    
    segments = list(range(0, max(1, int(duration) - step), step))
    total_segments = len(segments)
    
    for idx, start in enumerate(segments):
        if _progress_callback:
            prog = int((idx / total_segments) * 100)
            _progress_callback(prog, f"Analyse segment {start}s / {int(duration)}s")
        
        idx_start = int(start * sr)
        idx_end   = int((start + step) * sr)
        seg = y_filt[idx_start:idx_end]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: 
            continue
        
        # === AJOUT demandé : conversion 24→12 bins moyenne ===
        c_raw = librosa.feature.chroma_cqt(
            y=seg, sr=sr, tuning=tuning,
            bins_per_octave=24, hop_length=512
        )
        c_avg_12 = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)   # ← ici
        
        res = solve_key(c_avg_12, global_dom_root=global_dom_root)
        
        if res['score'] < 0.85:
            continue
        
        weight = 2.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not votes:
        return None

    most_common = votes.most_common(2)
    main_key = most_common[0][0]
    main_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == main_key]) * 100)
    
    modulation_detected = False
    target_key = None
    target_conf = 0
    if len(most_common) > 1:
        second_key = most_common[1][0]
        vote_ratio = votes[second_key] / sum(votes.values())
        if vote_ratio > 0.25:
            modulation_detected = True
            target_key = second_key
            target_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == second_key]) * 100)

    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    
    output = {
        "key": main_key, 
        "camelot": CAMELOT_MAP.get(main_key, "??"),
        "conf": min(main_conf, 99),
        "tempo": int(float(tempo)),
        "tuning_hz": round(440 * (2**(tuning/12)), 1),
        "pitch_offset": round(tuning, 2),
        "timeline": timeline, 
        "chroma": global_chroma_avg.tolist(),
        "modulation": modulation_detected, 
        "target_key": target_key,
        "target_conf": target_conf,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name
    }
    
    del y, y_filt, y_perc; gc.collect()
    return output

def get_piano_js(button_id, key_name):
    if not key_name or " " not in key_name: return ""
    n, mode = key_name.split()
    return f"""
    document.getElementById('{button_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const chord = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        chord.forEach((interval) => {{
            const baseFreq = freqs['{n}'] * Math.pow(2, interval/12);
            [1, 2].forEach((h) => {{
                const osc = ctx.createOscillator(); const g = ctx.createGain();
                osc.type = h === 1 ? 'triangle' : 'sine';
                osc.frequency.setValueAtTime(baseFreq * h, ctx.currentTime);
                g.gain.setValueAtTime(0, ctx.currentTime);
                g.gain.linearRampToValueAtTime(0.1/h, ctx.currentTime + 0.05);
                g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
                osc.connect(g); g.connect(ctx.destination);
                osc.start(); osc.stop(ctx.currentTime + 1.5);
            }});
        }});
    }};
    """

def send_telegram_expert(data, fig_timeline, fig_radar):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    
    mod_text = ""
    if data['modulation']:
        mod_text = f"⚠️ *MODULATION →* `{data['target_key']}` ({data['target_camelot']}) — Conf: `{data['target_conf']}%`\n\n"

    caption = (
        f"🎼 *DJ'S EAR PRO ELITE*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📂 *{data['name']}*\n\n"
        f"**TONALITÉ** : `{data['key'].upper()}`  ({data['camelot']})\n"
        f"**Confiance** : `{data['conf']}%`\n"
        f"**Tempo** : `{data['tempo']} BPM`\n"
        f"**Tuning** : `{data['tuning_hz']} Hz`  ({data['pitch_offset']}c)\n"
        f"{mod_text}"
        f"━━━━━━━━━━━━━━━━━━━"
    )

    try:
        img_tl = fig_timeline.to_image(format="png", width=1000, height=500, engine="kaleido")
        img_rd = fig_radar.to_image(format="png", width=600, height=600, engine="kaleido")
        
        media = [
            {'type': 'photo', 'media': 'attach://timeline.png', 'caption': caption, 'parse_mode': 'Markdown'},
            {'type': 'photo', 'media': 'attach://radar.png'}
        ]
        
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup",
            data={'chat_id': CHAT_ID, 'media': json.dumps(media)},
            files={'timeline.png': img_tl, 'radar.png': img_rd},
            timeout=20
        )
    except Exception as e:
        st.warning(f"Échec envoi Telegram : {e}")

# ────────────────────────────────────────────────
#                  INTERFACE
# ────────────────────────────────────────────────

st.title("🎧 DJ's Ear Pro Elite  •  Haute Précision")
st.markdown("Analyse multi-profils • Scan 2s • 24→12 bins moyenné • .m4a support • Telegram groupé")

files = st.file_uploader("Déposez vos morceaux (MP3, WAV, FLAC, M4A)", 
                         type=['mp3','wav','flac','m4a'], 
                         accept_multiple_files=True)

if files:
    files_to_process = list(reversed(files))
    total_files = len(files_to_process)
    
    progress_container = st.empty()
    results_container = st.container()
    
    for i, file in enumerate(files_to_process):
        progress_container.markdown(f"""
            <div style="padding:12px; background:rgba(16,185,129,0.12); border:1px solid #10b981; border-radius:12px; margin:12px 0;">
                <strong>Analyse {i+1}/{total_files}</strong> — {file.name}
            </div>
            """, unsafe_allow_html=True)
        
        with results_container:
            with st.status(f"Analyse → {file.name}", expanded=True) as status:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_prog(val, msg):
                    progress_bar.progress(val)
                    status_text.code(msg)
                
                file_bytes = file.getvalue()
                data = analyze_full_engine(file_bytes, file.name, update_prog)
                
                status.update(label=f"Terminé : {file.name}", state="complete", expanded=False)
        
        if data:
            with results_container:
                st.markdown(f"<div class='file-header'>RÉSULTAT — {data['name']}</div>", unsafe_allow_html=True)
                
                bg = "linear-gradient(135deg, #0f172a, #1e3a8a)" if not data['modulation'] else "linear-gradient(135deg, #1e1b4b, #7f1d1d)"
                st.markdown(f"""
                    <div class="report-card" style="background:{bg};">
                        <p style="opacity:0.7; letter-spacing:1.5px;">TONALITÉ PRINCIPALE</p>
                        <h1 style="font-size:5.8em; margin:8px 0;">{data['key'].upper()}</h1>
                        <p style="font-size:1.9em;">CAMELOT <b>{data['camelot']}</b>  •  Confiance <b>{data['conf']}%</b></p>
                        {f"<div class='modulation-alert'>⚠️ MODULATION VERS {data['target_key'].upper()} ({data['target_camelot']}) — Conf: {data['target_conf']}%</div>" if data['modulation'] else ""}
                    </div>
                    """, unsafe_allow_html=True)

                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.4em;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<div class='metric-box'><b>TUNING</b><br><span style='font-size:2.4em;'>{data['tuning_hz']}</span><br>Hz ({data['pitch_offset']}c)</div>", unsafe_allow_html=True)
                with cols[2]:
                    btn_id = f"play_{i}_{hash(file.name)}"
                    components.html(f"""
                        <button id="{btn_id}" style="width:100%; height:100px; background:linear-gradient(90deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:12px; font-weight:bold; font-size:1.15em; cursor:pointer;">
                            🎹 JOUER ACCORD
                        </button>
                        <script>{get_piano_js(btn_id, data['key'])}</script>
                        """, height=120)

                c1, c2 = st.columns([2.2, 1])
                with c1:
                    df_tl = pd.DataFrame(data['timeline'])
                    fig_t = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark",
                                    category_orders={"Note": NOTES_ORDER},
                                    title="Évolution harmonique")
                    fig_t.update_layout(height=340, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_t, use_container_width=True)
                
                with c2:
                    fig_r = go.Figure(data=go.Scatterpolar(
                        r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#818cf8'))
                    fig_r.update_layout(template="plotly_dark", height=340,
                                        title="Signature chromatique",
                                        margin=dict(l=20,r=20,t=40,b=20),
                                        polar=dict(radialaxis=dict(visible=False)))
                    st.plotly_chart(fig_r, use_container_width=True)

                # Envoi Telegram groupé
                send_telegram_expert(data, fig_t, fig_r)
                st.toast(f"Rapport Telegram envoyé pour {file.name}", icon="✅")

    progress_container.success(f"✓ {total_files} fichier(s) analysé(s) avec succès !")

    if st.sidebar.button("🧹 Vider cache & mémoire"):
        st.cache_data.clear()
        st.rerun()

else:
    st.info("Déposez vos fichiers audio pour lancer l’analyse (multi-profils + 24→12 bins moyenné sur segments)")
