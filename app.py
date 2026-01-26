import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import json
import os
from pathlib import Path

from scipy.signal import butter, lfilter
from pydub import AudioSegment
import streamlit.components.v1 as components

# ────────────────────────────────────────────────
#                CONFIGURATION
# ────────────────────────────────────────────────

st.set_page_config(
    page_title="DJ's Ear Pro Elite",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajout ffmpeg si présent localement (Windows principalement)
if os.name == 'nt' and (ffmpeg_path := Path(r"C:\ffmpeg\bin")).exists():
    os.environ["PATH"] += os.pathsep + str(ffmpeg_path)

# ────────────────────────────────────────────────
#                CONSTANTES
# ────────────────────────────────────────────────

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# Profils de tonalité (Krumhansl le plus utilisé et fiable)
PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    # Les autres profils sont moins utilisés → on peut les garder en option avancée plus tard
}

# ────────────────────────────────────────────────
#                     STYLES
# ────────────────────────────────────────────────

st.markdown("""
    <style>
    .stApp { background-color: #0a0e17; }
    .report-card {
        padding: 2.2rem; border-radius: 1.6rem; text-align: center; color: white;
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        border: 1px solid rgba(99,102,241,0.25); box-shadow: 0 10px 40px rgba(0,0,0,0.65);
        margin: 1.5rem 0;
    }
    .modulation-alert {
        background: rgba(220,38,38,0.18); color: #fca5a5;
        padding: 1rem; border-radius: 1rem; border: 1px solid #dc2626;
        margin: 1.5rem 0; font-weight: 600;
    }
    .metric-box {
        background: #111827; border-radius: 1rem; padding: 1.4rem;
        border: 1px solid #374151; text-align: center;
    }
    .file-title {
        background: #1f2937; color: #34d399; padding: 0.6rem 1.2rem;
        border-radius: 0.8rem; font-family: 'Consolas', monospace; font-weight: bold;
        border-left: 5px solid #10b981; margin: 1rem 0 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────
#              FONCTIONS UTILITAIRES AUDIO
# ────────────────────────────────────────────────

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def preprocess_audio(y, sr, mode="sniper"):
    if mode == "sniper":
        # Plus agressif → on garde surtout les harmoniques médiums
        y_harm = librosa.effects.harmonic(y, margin=3.5)
        b, a = butter_bandpass(90, 4800, sr)
    else:
        y_harm, _ = librosa.effects.hpss(y, margin=(3.8, 1.2))
        y_harm = librosa.effects.preemphasis(y_harm)
        b, a = butter_bandpass(110, 3200, sr)

    return lfilter(b, a, y_harm)


# ────────────────────────────────────────────────
#              DÉTECTION TONALE
# ────────────────────────────────────────────────

def correlation_key_score(chroma: np.ndarray, profile: np.ndarray) -> float:
    if chroma.size != 12 or profile.size != 12:
        return -1.0
    chroma = (chroma - chroma.min()) / (chroma.ptp() + 1e-10)
    return np.corrcoef(chroma, profile)[0, 1]


def solve_key(chroma12: np.ndarray, profiles=PROFILES, global_dom_root=None) -> dict:
    best_score = -np.inf
    best_key = None

    chroma12 = np.asarray(chroma12).flatten()
    if chroma12.size != 12:
        return {"key": "Erreur", "score": 0.0}

    if chroma12.max() < 1e-6:
        return {"key": "Silence", "score": 0.0}

    for profile_name, modes in profiles.items():
        for mode, profile in modes.items():
            profile = np.asarray(profile)
            for root in range(12):
                rotated = np.roll(profile, root)
                corr = correlation_key_score(chroma12, rotated)

                # Bonus harmoniques fortes (tierce, quinte)
                third = (root + (3 if mode == "minor" else 4)) % 12
                fifth  = (root + 7) % 12
                bonus = 0.20 * chroma12[third] + 0.12 * chroma12[fifth]

                # Bonus si dominante globale détectée
                if global_dom_root is not None and fifth == global_dom_root:
                    bonus += 0.25 * chroma12[global_dom_root]

                score = corr + bonus

                if score > best_score:
                    best_score = score
                    best_key = f"{NOTES[root]} {mode}"

    return {"key": best_key or "Inconnu", "score": float(best_score)}


# ────────────────────────────────────────────────
#              MOTEUR D'ANALYSE
# ────────────────────────────────────────────────

@st.cache_data(ttl="10min", show_spinner="Analyse en cours...")
def analyze_track(audio_bytes: bytes, filename: str, filter_mode: str = "sniper") -> dict | None:
    try:
        ext = Path(filename).suffix.lower().lstrip('.')

        if ext in ['m4a', 'mp4', 'aac']:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            y = samples / (1 << (8 * audio.sample_width - 1))
            sr = audio.frame_rate
        else:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)

        if len(y) < sr * 5:  # moins de 5 secondes
            return None

        # Resampling systématique pour cohérence
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        y_clean = preprocess_audio(y, sr, filter_mode)

        # Estimation du diapason global
        tuning_offset = librosa.estimate_tuning(y=y_clean, sr=sr)

        # Chroma global (plus haute résolution)
        chroma_cqt = librosa.feature.chroma_cqt(
            y=y_clean, sr=sr,
            tuning=tuning_offset,
            bins_per_octave=36,       # ← plus précis
            hop_length=512*2
        )
        global_chroma = np.mean(chroma_cqt, axis=1)

        # Détection dominante globale (souvent quinte)
        top_idx = np.argsort(global_chroma)[-2:]
        dom_root = None
        if (top_idx[0] + 7) % 12 in top_idx:
            dom_root = (top_idx[0] + 7) % 12

        duration = librosa.get_duration(y=y, sr=sr)
        window = 4          # secondes par segment
        step = 1.5          # chevauchement important

        timeline = []
        votes = Counter()

        for start in np.arange(0, duration - window + 0.1, step):
            seg = y_clean[int(start * sr):int((start + window) * sr)]
            if len(seg) < sr * 0.8 or np.max(np.abs(seg)) < 0.015:
                continue

            cqt = librosa.feature.chroma_cqt(
                y=seg, sr=sr, tuning=tuning_offset,
                bins_per_octave=36, hop_length=512
            )
            c12 = np.mean(cqt, axis=1)

            result = solve_key(c12, global_dom_root=dom_root)

            if result["score"] < 0.68:
                continue

            weight = 1.8 if start < 12 or start > duration - 20 else 1.0
            votes[result["key"]] += result["score"] * 100 * weight

            timeline.append({
                "start": round(start, 1),
                "key": result["key"],
                "confidence": round(result["score"] * 100, 1)
            })

        if not votes:
            return None

        main_key = votes.most_common(1)[0][0]
        main_conf = np.mean([t["confidence"] for t in timeline if t["key"] == main_key])

        # Détection modulation
        modulation = False
        target_key = target_camelot = None
        if len(votes) >= 2:
            _, second_score = votes.most_common(2)[1]
            if second_score / sum(votes.values()) > 0.28:
                modulation = True
                target_key = votes.most_common(2)[1][0]
                target_camelot = CAMELOT_MAP.get(target_key)

        tempo, _ = librosa.beat.beat_track(y=librosa.effects.percussive(y), sr=sr, units="tempo")

        return {
            "filename": filename,
            "key": main_key,
            "camelot": CAMELOT_MAP.get(main_key, "??"),
            "confidence": min(int(main_conf), 99),
            "tempo": int(round(tempo)) if tempo is not None else "—",
            "tuning_offset_cents": round(tuning_offset * 100, 1),
            "tuning_hz": round(440 * (2 ** (tuning_offset / 12)), 1),
            "timeline": timeline,
            "global_chroma": global_chroma.tolist(),
            "modulation": modulation,
            "target_key": target_key,
            "target_camelot": target_camelot
        }

    except Exception as e:
        st.error(f"Erreur analyse {filename} : {str(e)}")
        return None


# ────────────────────────────────────────────────
#                   INTERFACE
# ────────────────────────────────────────────────

st.title("🎧 DJ's Ear Pro Elite  · Analyse tonale avancée")

with st.sidebar:
    st.header("Paramètres")
    filter_style = st.radio(
        "Style de pré-filtrage",
        ["Original (polyvalent)", "Sniper (médiums clairs)"],
        index=1
    )
    filter_mode = "sniper" if "Sniper" in filter_style else "original"

    if st.button("🧹 Vider le cache", type="primary"):
        st.cache_data.clear()
        st.success("Cache vidé", icon="✅")
        st.rerun()

    st.caption("v2025 – amélioré par Grok")


uploaded_files = st.file_uploader(
    "Déposez vos fichiers audio (mp3, wav, m4a, flac...)",
    type=["mp3", "wav", "m4a", "flac", "aac"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Déposez un ou plusieurs fichiers audio pour commencer l'analyse", icon="🎵")
    st.stop()

for file in reversed(uploaded_files):
    with st.container():
        with st.status(f"Analyse → {file.name}", expanded=True) as status:
            result = analyze_track(file.getvalue(), file.name, filter_mode)

            if result is None:
                status.update(label="Échec de l'analyse", state="error")
                st.error("Impossible d'extraire des informations harmoniques exploitables.")
                continue

            status.update(label="Analyse terminée", state="complete", expanded=False)

        # ─── Affichage principal ────────────────────────────────
        bg = "linear-gradient(135deg, #1e1b4b, #7f1d1d)" if result["modulation"] else "linear-gradient(135deg, #0f172a, #1e3a8a)"

        st.markdown(f"""
        <div class="report-card" style="background:{bg};">
            <div style="font-size:1.1rem; opacity:0.8; margin-bottom:0.6rem;">TONALITÉ PRINCIPALE</div>
            <h1 style="font-size:5.8rem; margin:0.2rem 0;">{result['key']}</h1>
            <h3 style="margin:0.4rem 0;">Camelot {result['camelot']}  •  Confiance {result['confidence']}%</h3>
            {f"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE → {result['target_key']} ({result['target_camelot']})</div>" if result['modulation'] else ""}
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns([1,1,1.4])
        cols[0].markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:3.2rem'>{result['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div class='metric-box'><b>TUNING</b><br><span style='font-size:3.2rem'>{result['tuning_hz']}</span><br>Hz  ({result['tuning_offset_cents']:+}¢)</div>", unsafe_allow_html=True)

        with cols[2]:
            btn_id = f"play_{hash(file.name)}"
            components.html(
                f"""
                <button id="{btn_id}" style="width:100%; height:88px; background:#6366f1; color:white; border:none; border-radius:12px; font-size:1.3rem; font-weight:bold; cursor:pointer;">
                    ▶ Tester {result['key']}
                </button>
                <script>
                document.getElementById('{btn_id}').onclick = function() {{
                    const ctx = new AudioContext();
                    const root = '{result['key'].split()[0]}';
                    const mode = '{result['key'].split()[1]}';
                    const freqs = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
                    const base = freqs[root];
                    const intervals = mode === 'minor' ? [0,3,7,12] : [0,4,7,12];
                    intervals.forEach(i => {{
                        const f = base * Math.pow(2, i/12);
                        [1, 1.5].forEach(gainMult => {{
                            const osc = ctx.createOscillator();
                            const gain = ctx.createGain();
                            osc.type = gainMult === 1 ? 'triangle' : 'sine';
                            osc.frequency.setValueAtTime(f * gainMult, ctx.currentTime);
                            gain.gain.setValueAtTime(0, ctx.currentTime);
                            gain.gain.linearRampToValueAtTime(0.12 / gainMult, ctx.currentTime + 0.06);
                            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.2);
                            osc.connect(gain).connect(ctx.destination);
                            osc.start();
                            osc.stop(ctx.currentTime + 2.2);
                        }});
                    }});
                }};
                </script>
                """,
                height=100
            )

        st.markdown(f"<div class='file-title'>Évolution dans le temps – {file.name}</div>", unsafe_allow_html=True)

        left, right = st.columns([3, 1.3])

        with left:
            if result["timeline"]:
                df = pd.DataFrame(result["timeline"])
                fig = px.line(
                    df, x="start", y="key",
                    markers=True, template="plotly_dark",
                    category_orders={"key": NOTES_ORDER},
                    labels={"start": "Temps (s)", "key": "Tonalité"}
                )
                fig.update_traces(line=dict(width=3.5), marker=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)

        with right:
            fig_polar = go.Figure(go.Scatterpolar(
                r=result["global_chroma"],
                theta=NOTES,
                fill='toself',
                fillcolor="rgba(99,102,241,0.4)",
                line=dict(color="#6366f1")
            ))
            fig_polar.update_layout(
                template="plotly_dark",
                polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_polar, use_container_width=True)

        st.divider()
