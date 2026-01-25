# RCDJ228 MUSIC SNIPER M5 - HYBRIDE 2026 (version corrigée avec cache + session_state)
# Moteur M4 + Timeline fine M3 + UI/Telegram premium M4

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

# Force FFMPEG (Windows)
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# ────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ────────────────────────────────────────────────
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER M5", page_icon="🔫🎵", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID        = st.secrets.get("CHAT_ID")

NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "krumhansl": {"major": [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],
                  "minor": [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]},
    "temperley": {"major": [5.0,2.0,3.5,2.0,4.5,4.0,2.0,4.5,2.0,3.5,1.5,4.0],
                  "minor": [5.0,2.0,3.5,4.5,2.0,4.0,2.0,4.5,3.5,2.0,1.5,4.0]},
    "aarden":    {"major": [17.7661,0.145624,14.9265,0.160186,19.8049,11.3587,0.291248,22.062,0.145624,8.15494,0.232998,18.6691],
                  "minor": [18.2648,0.737619,14.0499,16.8599,0.702699,14.5212,0.737619,19.8145,5.84214,2.68046,2.51091,9.84455]},
    "bellman":   {"major": [16.8,0.86,12.95,1.41,13.49,11.93,1.25,16.74,1.56,12.81,1.89,12.44],
                  "minor": [18.16,0.69,12.99,13.34,1.07,11.15,1.38,17.2,13.62,1.27,12.79,2.4]}
}

WEIGHTS = {"global": 0.55, "segments": 0.35, "bass_bonus": 0.10}

# ────────────────────────────────────────────────
# INITIALISATION SESSION STATE
# ────────────────────────────────────────────────
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}      # {filename: data}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "global_progress" not in st.session_state:
    st.session_state.global_progress = 0.0

# ────────────────────────────────────────────────
# STYLES CSS
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99,102,241,0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 24px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 12px 24px; border-radius: 12px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 16px;
        border-left: 6px solid #10b981;
    }
    .modulation-alert {
        background: rgba(239,68,68,0.18); color: #f87171; padding: 16px; border-radius: 16px;
        border: 1px solid #ef4444; margin: 20px 0; font-weight: bold; font-family: 'JetBrains Mono';
    }
    .metric-box {
        background: #161b22; border-radius: 16px; padding: 20px; text-align: center;
        border: 1px solid #30363d; height: 100%; transition: all 0.25s;
    }
    .metric-box:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
    .stats-box {
        background: #1a2332; border-radius: 12px; padding: 16px; margin: 16px 0;
        border: 1px solid #2d3748; font-family: 'JetBrains Mono'; font-size: 0.95em;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# FONCTIONS TECHNIQUES
# ────────────────────────────────────────────────

def butter_lowpass(y, sr, cutoff=160):
    nyq = 0.5 * sr
    b, a = butter(4, cutoff / nyq, btype='low')
    return lfilter(b, a, y)

def apply_precision_filters(y, sr):
    y_harm, _ = librosa.effects.hpss(y, margin=(1.3, 4.8))
    nyq = 0.5 * sr
    b, a = butter(4, [50/nyq, 5200/nyq], btype='band')
    return lfilter(b, a, y_harm)

def vote_profiles(chroma, chroma_cens, bass_chroma):
    cv   = (chroma       - chroma.min())       / (chroma.max()       - chroma.min()       + 1e-9)
    cens = (chroma_cens  - chroma_cens.min())  / (chroma_cens.max()  - chroma_cens.min()  + 1e-9)
    bv   = (bass_chroma  - bass_chroma.min())  / (bass_chroma.max()  - bass_chroma.min()  + 1e-9)

    scores = {f"{n} {m}": 0.0 for n in NOTES_LIST for m in ["major", "minor"]}

    for profile in PROFILES.values():
        for mode in ["major", "minor"]:
            for i in range(12):
                corr_cqt  = np.corrcoef(cv,   np.roll(profile[mode], i))[0,1]
                corr_cens = np.corrcoef(cens, np.roll(profile[mode], i))[0,1]
                combined  = 0.68 * corr_cqt + 0.32 * corr_cens

                bonus = (
                    bv[i]              * 0.42 +
                    cv[(i+7)%12]       * 0.19 +
                    (cv[i] + bv[i])/2  * 0.14
                )
                scores[f"{NOTES_LIST[i]} {mode}"] += (combined + bonus) / len(PROFILES)
    return scores

def process_audio_m5(file_bytes, file_name, progress_cb=None, threshold=0.78):
    ext = file_name.lower().split('.')[-1]
    sr_target = 22050

    try:
        if ext == 'm4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            y = samples / (1 << (8 * audio.sample_width - 1))
            sr = audio.frame_rate
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                sr = sr_target
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=sr_target, mono=True)
    except Exception as e:
        st.error(f"Erreur lecture {file_name}: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_precision_filters(y, sr)

    # Analyse globale
    chroma_cqt_glob  = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning_offset), axis=1)
    chroma_cens_glob = np.mean(librosa.feature.chroma_cens(y=y_filt, sr=sr, tuning=tuning_offset), axis=1)
    bass_glob        = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y, sr), sr=sr), axis=1)
    global_scores    = vote_profiles(chroma_cqt_glob, chroma_cens_glob, bass_glob)

    # Analyse segments
    seg_duration = 8.0
    step         = 3.0
    segments_starts = np.arange(0, max(0.1, duration - seg_duration), step)

    segment_votes = Counter()
    timeline = []
    valid_segments = 0
    retained_scores = []

    for idx, start_s in enumerate(segments_starts):
        if progress_cb:
            prog = int((idx + 1) / len(segments_starts) * 100)
            progress_cb(prog, f"Segment {idx+1}/{len(segments_starts)} — {start_s:.1f}s")

        seg_start_idx = int(start_s * sr)
        seg_end_idx   = int((start_s + seg_duration) * sr)
        y_seg = y_filt[seg_start_idx:seg_end_idx]

        if len(y_seg) < 1200 or np.max(np.abs(y_seg)) < 0.012:
            continue

        cqt_seg  = np.mean(librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning_offset), axis=1)
        cens_seg = np.mean(librosa.feature.chroma_cens(y=y_seg, sr=sr, tuning=tuning_offset), axis=1)
        bass_seg = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y_seg, sr), sr=sr), axis=1)

        seg_scores = vote_profiles(cqt_seg, cens_seg, bass_seg)
        best_key = max(seg_scores, key=seg_scores.get)

        if seg_scores[best_key] >= threshold:
            weight = 1.45 if 0.18 < (start_s / duration) < 0.82 else 1.0
            
            conf = seg_scores[best_key]
            if conf >= 0.88:
                weight *= 2.2
            elif conf >= 0.84:
                weight *= 1.7
            elif conf >= 0.80:
                weight *= 1.3
            
            segment_votes[best_key] += conf * weight
            timeline.append({
                "time": start_s + seg_duration/2,
                "key": best_key,
                "score": conf
            })
            valid_segments += 1
            retained_scores.append(conf)

    if not segment_votes and not global_scores:
        return None

    total_seg = sum(segment_votes.values()) or 1
    seg_norm = {k: v / total_seg for k,v in segment_votes.items()}

    final_scores = Counter()
    for k in set(global_scores) | set(seg_norm):
        final_scores[k] = (
            global_scores.get(k, 0) * WEIGHTS["global"] +
            seg_norm.get(k, 0)      * WEIGHTS["segments"]
        )

    best_key, best_raw = final_scores.most_common(1)[0]
    max_raw = max(final_scores.values()) if final_scores else 1
    confidence = min(99, int(100 * best_raw / max_raw * 1.18))

    modulation = None
    if len(timeline) >= 8:
        mid = len(timeline) // 2
        first = Counter(t["key"] for t in timeline[:mid])
        second = Counter(t["key"] for t in timeline[mid:])
        if first and second:
            top1 = first.most_common(1)[0][0]
            top2 = second.most_common(1)[0][0]
            if top1 != top2 and second[top2] > first[top2] * 1.45:
                modulation = top2

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    result = {
        "name": file_name,
        "key": best_key,
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "conf": confidence,
        "tempo": int(round(float(tempo))),
        "tuning_hz": round(440 * (2 ** (tuning_offset / 12)), 1),
        "tuning_cents": round(tuning_offset * 100, 1),
        "modulation": modulation,
        "target_camelot": CAMELOT_MAP.get(modulation, "??") if modulation else None,
        "timeline": timeline,
        "chroma": chroma_cqt_glob.tolist(),
        "valid_segments": valid_segments,
        "duration": round(duration, 1),
        "retention_pct": (valid_segments / len(segments_starts) * 100) if len(segments_starts) > 0 else 0,
        "avg_retained_score": np.mean(retained_scores) if retained_scores else 0
    }

    # Telegram Report (inchangé)
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            df_tl = pd.DataFrame(timeline)
            fig_tl = px.line(df_tl, x="time", y="key", markers=True, template="plotly_dark",
                             category_orders={"key": NOTES_ORDER})
            fig_tl.update_layout(height=480, margin=dict(l=20,r=20,t=30,b=20))
            img_tl = fig_tl.to_image(format="png", width=1100, height=520)

            fig_rd = go.Figure(go.Scatterpolar(r=result["chroma"], theta=NOTES_LIST,
                                               fill='toself', line_color='#10b981'))
            fig_rd.update_layout(template="plotly_dark", height=520,
                                 polar=dict(radialaxis=dict(visible=False)),
                                 margin=dict(l=40,r=40,t=30,b=30))
            img_rd = fig_rd.to_image(format="png", width=620, height=620)

            mod_text = f"**MODULATION →** {modulation.upper()} ({result['target_camelot']})" if modulation else "**STABLE**"
            caption = (
                f"**RCDJ228 SNIPER M5 RAPPORT – 2026**\n━━━━━━━━━━━━━━━━━\n"
                f"**Track**  `{file_name}`\n"
                f"**Tonalité**  `{best_key.upper()}`\n"
                f"**Camelot**  `{result['camelot']}`\n"
                f"**Confiance**  `{confidence}%`\n"
                f"**Tempo**  `{result['tempo']} BPM`\n"
                f"**Accordage**  `{result['tuning_hz']} Hz  ({result['tuning_cents']:+.1f}¢)`\n"
                f"**Segments valides**  `{valid_segments}`\n"
                f"{mod_text}\n━━━━━━━━━━━━━━━━━"
            )

            files = {'p1': ('timeline.png', img_tl, 'image/png'), 'p2': ('radar.png', img_rd, 'image/png')}
            media = [
                {'type': 'photo', 'media': 'attach://p1', 'caption': caption, 'parse_mode': 'Markdown'},
                {'type': 'photo', 'media': 'attach://p2'}
            ]
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup",
                data={'chat_id': CHAT_ID, 'media': json.dumps(media)},
                files=files, timeout=25
            )
        except:
            pass

    del y, y_filt
    gc.collect()
    return result

# ────────────────────────────────────────────────
# VERSION CACHÉE (clé du gain de performance)
# ────────────────────────────────────────────────
@st.cache_data(
    show_spinner="Analyse M5 en cours...",
    hash_funcs={bytes: id},
    persist="disk",
    ttl=None
)
def cached_process_audio(file_bytes, file_name, threshold):
    # On passe progress_cb=None car on ne peut pas sérialiser les callbacks
    return process_audio_m5(file_bytes, file_name, None, threshold)

def get_chord_test_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const base = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,
                       'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}}['{note}'];
        const intervals = '{mode}' === 'minor' ? [0,3,7,12] : [0,4,7,12];
        intervals.forEach(i => {{
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(base * Math.pow(2, i/12), ctx.currentTime);
            gain.gain.setValueAtTime(0, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.13, ctx.currentTime + 0.09);
            gain.gain.exponentialRampToValueAtTime(0.0008, ctx.currentTime + 2.4);
            osc.connect(gain); gain.connect(ctx.destination);
            osc.start(); osc.stop(ctx.currentTime + 2.4);
        }});
    }};
    """

# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────
header_col1, header_col2 = st.columns([6, 2])

with header_col1:
    st.title("🔫 RCDJ228 MUSIC SNIPER M5 — Précision & Granularité 2026")

with header_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.analysis_results:
        st.caption(f"{len(st.session_state.analysis_results)} track(s) analysée(s)")

# Barre de progression globale (placeholder)
prog_container = st.empty()
prog_text = st.empty()

if "last_total" not in st.session_state:
    st.session_state.last_total = 0

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=100)
    st.header("Sniper M5")
    st.caption("Hybride 2026 • Cache + Timeline fine + boost conf ≥80%")

    st.subheader("Réglages fins")
    SEGMENT_THRESHOLD = st.slider(
        "Seuil confiance segment",
        min_value=0.55,
        max_value=0.90,
        value=0.78,
        step=0.01,
        format="%.2f",
        help="Plus haut = plus fiable mais timeline plus vide"
    )

    st.markdown("---")
    if st.button("🗑️ Tout effacer et recommencer", type="primary"):
        st.session_state.analysis_results = {}
        st.session_state.processed_files = set()
        st.session_state.global_progress = 0.0
        st.session_state.last_total = 0
        st.rerun()

uploaded_files = st.file_uploader("Déposez vos tracks (mp3, wav, flac, m4a)", 
                                 type=['mp3','wav','flac','m4a'], 
                                 accept_multiple_files=True)

# ────────────────────────────────────────────────
# TRAITEMENT
# ────────────────────────────────────────────────
if uploaded_files:
    total = len(uploaded_files)
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

    # Mise à jour barre globale seulement si nouveau total ou nouveaux fichiers
    if total != st.session_state.last_total or new_files:
        st.session_state.global_progress = len(st.session_state.processed_files) / total if total > 0 else 0
        prog_container.progress(st.session_state.global_progress)
        prog_text.markdown(f"**Prêt — {len(st.session_state.processed_files)}/{total} déjà analysé(s)**")

    results_container = st.container()

    current_idx = len(st.session_state.processed_files)

    for file in uploaded_files:
        fname = file.name

        if fname in st.session_state.processed_files:
            data = st.session_state.analysis_results.get(fname)
            # On affiche quand même le rapport (déjà calculé)
        else:
            current_idx += 1
            percent = current_idx / total

            prog_container.progress(percent)
            prog_text.markdown(f"**Analyse {current_idx}/{total}** — {fname}")

            with st.status(f"Scan M5 → {fname}", expanded=True) as status:
                inner_prog = st.progress(0)
                inner_text = st.empty()

                def upd_prog(p, msg):
                    inner_prog.progress(p/100)
                    inner_text.code(msg, language="text")

                data = cached_process_audio(file.getvalue(), fname, SEGMENT_THRESHOLD)

                if data:
                    st.session_state.analysis_results[fname] = data
                    st.session_state.processed_files.add(fname)
                    status.update(label=f"Terminé — {fname}", state="complete", expanded=False)
                else:
                    status.update(label=f"Échec — {fname}", state="error")

        # Affichage du rapport (pour tous les fichiers)
        if data and fname in st.session_state.analysis_results:
            with results_container:
                st.markdown(f"<div class='file-header'>RAPPORT M5 → {data['name']}</div>", unsafe_allow_html=True)
                
                bg = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 87 else "linear-gradient(135deg, #1e293b, #0f172a)"
                st.markdown(f"""
                <div class="report-card" style="background:{bg};">
                    <h1 style="font-size:6.2em; margin:0; font-weight:900;">{data['key'].upper()}</h1>
                    <p style="font-size:1.9em; margin:14px 0;">
                        CAMELOT <b>{data['camelot']}</b>  •  CONFIANCE <b>{data['conf']}%</b>
                    </p>
                    {f"<div class='modulation-alert'>MODULATION → {data['modulation'].upper()} ({data['target_camelot']})</div>" if data['modulation'] else ""}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="stats-box">
                    Seuil : <b>{SEGMENT_THRESHOLD:.2f}</b>  
                      Segments retenus : <b>{data['valid_segments']}</b> / ~{len(np.arange(0, max(0.1, data['duration'] - 8), 3)):.0f} 
                    ({data['retention_pct']:.1f} %)  
                      Score moyen retenus : <b>{data['avg_retained_score']:.3f}</b>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2.6em;color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2.6em;color:#58a6ff;'>{data['tuning_hz']}</span><br>Hz ({data['tuning_cents']:+.1f}¢)</div>", unsafe_allow_html=True)
                with col3:
                    btn_id = f"chord_{hash(fname)}"
                    components.html(f"""
                    <button id="{btn_id}" style="width:100%; height:110px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:18px; font-size:1.3em; cursor:pointer; font-weight:bold;">TESTER L'ACCORD</button>
                    <script>{get_chord_test_js(btn_id, data['key'])}</script>
                    """, height=130)

                c1, c2 = st.columns([2.3, 1])
                with c1:
                    if data["timeline"]:
                        df = pd.DataFrame(data["timeline"])
                        fig = px.line(df, x="time", y="key", markers=True, template="plotly_dark",
                                      category_orders={"key": NOTES_ORDER})
                        fig.update_layout(height=360, margin=dict(l=12,r=12,t=12,b=12))
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig_r = go.Figure(go.Scatterpolar(r=data["chroma"], theta=NOTES_LIST,
                                                      fill='toself', line_color='#22c55e'))
                    fig_r.update_layout(template="plotly_dark", height=360,
                                        polar=dict(radialaxis=dict(visible=False)),
                                        margin=dict(l=24,r=24,t=12,b=12))
                    st.plotly_chart(fig_r, use_container_width=True)

                st.markdown("---")

    # Fin du traitement → on met à jour la progression finale
    st.session_state.global_progress = 1.0
    prog_container.progress(1.0)
    prog_text.success(f"**Mission terminée — {total} track(s) traitée(s)**")
    st.session_state.last_total = total

else:
    prog_container.empty()
    prog_text.empty()
