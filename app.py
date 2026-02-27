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

# --- RÉFÉRENTIELS HARMONIQUES --- (tout ton code original reste identique jusqu'ici)
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODAL_MODES = ['ionian', 'major', 'lydian', 'mixolydian', 'dorian', 'aeolian', 'minor', 'phrygian', 'locrian']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in MODAL_MODES]

CAMELOT_MAP = { ... }  # (ton mapping original)

CAMELOT_TABLE = { ... }  # (ton tableau original)

PROFILES = { ... }  # (tous tes profils)

# ... (tout le reste de tes constantes, PROFILES_ROLLED, MODAL_TO_CAMELOT_TYPE, etc. reste EXACTEMENT comme avant)

# --- MOTEURS DE CALCUL --- (toutes tes fonctions restent identiques)
# arbitrage_expert_universel, seconds_to_mmss, apply_sniper_filters, get_bass_priority, get_sub_bass_priority,
# detect_harmonic_sections, detect_cadence_resolution, get_exact_camelot, solve_key_sniper, solve_key_sniper_modal,
# get_safe_camelot, get_camelot_modal, get_key_score, process_audio, get_chord_js
# (je ne recopie pas les 800 lignes pour ne pas alourdir, mais elles restent inchangées)

# ====================== NOUVELLE FONCTION EXPORT ======================
def get_export_content(analysis_data, chosen_key):
    chosen_camelot = get_exact_camelot(chosen_key)
    chosen_conf = (analysis_data.get('conf') if chosen_key == analysis_data.get('key')
                   else analysis_data.get('dominant_conf') if chosen_key == analysis_data.get('dominant_key')
                   else analysis_data.get('modal_conf', 0))

    content = f"""RCDJ228 MUSIC SNIPER - EXPORT TONALITÉ PRINCIPALE
══════════════════════════════════════
Fichier          : {analysis_data['name']}
Date analyse     : {datetime.now().strftime('%d/%m/%Y %H:%M')}

TONALITÉ PRINCIPALE CHOISIE
───────────────────────────
Clé              : {chosen_key}
Camelot          : {chosen_camelot}
Confiance        : {chosen_conf}%

Détails complets
────────────────
Consonance       : {analysis_data['key']} ({analysis_data['camelot']}) — {analysis_data.get('key_presence', 0)}% — {analysis_data['conf']}%
Dominante        : {analysis_data['dominant_key']} ({analysis_data['dominant_camelot']}) — {analysis_data['dominant_percentage']}% — {analysis_data['dominant_conf']}%
Décision Sniper  : {analysis_data['confiance_pure']} ({analysis_data['pure_camelot']}) — {analysis_data['avis_expert']}
Mode détecté     : {analysis_data.get('modal_key', '—')} ({analysis_data.get('modal_camelot', '??')})

Accordage        : {analysis_data['tuning']} Hz
Section harmonique : {analysis_data['harm_start']} → {analysis_data['harm_end']}
"""

    if analysis_data.get('modulation'):
        content += f"""
MODULATION DÉTECTÉE
───────────────────
→ {analysis_data['target_key']} ({analysis_data['target_camelot']}) à {analysis_data['modulation_time_str']}
Présence         : {analysis_data['mod_target_percentage']}%
"""

    content += "\n══════════════════════════════════════\nMerci d'utiliser RCDJ228 MUSIC SNIPER 🎯"
    return content
# =====================================================================

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
                    mod_alert = f"<div class='modulation-alert'>⚠️ MODULATION : <b>{analysis_data['target_key'].upper()}</b> ({analysis_data['target_camelot']}) &nbsp;|&nbsp; PRÉSENCE : <b>{analysis_data['mod_target_percentage']}%</b> &nbsp;|&nbsp; CONFIANCE : <b>{analysis_data['target_conf']}%</b>{ends_badge}</div>"

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
                    st.markdown(f"<div style='background:rgba(245,158,11,0.12); border:1px solid #f59e0b; border-radius:15px; padding:14px 20px; margin-bottom:12px; font-family:JetBrains Mono,monospace; color:#fbbf24;'>⚠️ <b>ALERTE INSTABILITÉ</b> — Indice de stabilité : <b>{stability_val}</b> &nbsp;|&nbsp; Ce morceau change fréquemment de structure harmonique.</div>", unsafe_allow_html=True)

                # ====================== CHOIX TONALITÉ PRINCIPALE (TA CONSONANCE PAR DÉFAUT) ======================
                st.markdown("### 🎯 **Tonalité principale à enregistrer**")
                
                col_choice, col_export = st.columns([3, 1])
                
                with col_choice:
                    choice = st.radio(
                        label="Quelle clé veux-tu utiliser comme tonalité principale ?",
                        options=["Consonance (recommandé)", "Décision Sniper Finale", "Dominante"],
                        horizontal=True,
                        key=f"radio_{file_name}"
                    )
                
                if choice == "Consonance (recommandé)":
                    chosen_key = analysis_data['key']
                elif choice == "Décision Sniper Finale":
                    chosen_key = analysis_data['confiance_pure']
                else:
                    chosen_key = analysis_data['dominant_key']
                
                chosen_camelot = get_exact_camelot(chosen_key)
                
                with col_export:
                    export_content = get_export_content(analysis_data, chosen_key)
                    st.download_button(
                        label="📥 Exporter .txt",
                        data=export_content,
                        file_name=f"{file_name.split('.')[0]}_TONALITE.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                st.markdown(f"""
                    <div style="background:#1f2937; padding:15px; border-radius:12px; text-align:center; margin:15px 0;">
                        <span style="font-size:2.8em; font-weight:900; color:#10b981;">{chosen_camelot}</span><br>
                        <span style="font-size:1.4em; color:white;">{chosen_key.upper()}</span>
                    </div>
                """, unsafe_allow_html=True)
                # ======================================================================================================

                # ====================== CONSEILS DE MIX HARMONIQUE ======================
                st.markdown("### 🎛️ **Conseils de mix harmonique** — basée sur ta tonalité principale")

                cam = chosen_camelot
                num = int(cam[:-1])
                letter = cam[-1]
                opposite = 'B' if letter == 'A' else 'A'

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                        <div style="background:#1f2937; padding:18px; border-radius:12px; text-align:center; border:2px solid #10b981;">
                            <span style="font-size:1.3em; color:#10b981;">🔄 SAME KEY</span><br>
                            <span style="font-size:2.4em; font-weight:900;">{cam}</span><br>
                            <span style="font-size:0.85em; opacity:0.7;">Mix ultra fluide • parfait pour long mix</span>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div style="background:#1f2937; padding:18px; border-radius:12px; text-align:center; border:2px solid #3b82f6;">
                            <span style="font-size:1.3em; color:#3b82f6;">📈 VOISIN ±1</span><br>
                            <span style="font-size:2.1em; font-weight:900;">{num-1}{letter} &nbsp; {num+1}{letter}</span><br>
                            <span style="font-size:0.85em; opacity:0.7;">Transition douce • montée légère</span>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div style="background:#1f2937; padding:18px; border-radius:12px; text-align:center; border:2px solid #8b5cf6;">
                            <span style="font-size:1.3em; color:#8b5cf6;">⚡ RELATIVE (le plus puissant)</span><br>
                            <span style="font-size:2.4em; font-weight:900;">{num}{opposite}</span><br>
                            <span style="font-size:0.85em; opacity:0.7;">Boost d’énergie garanti</span>
                        </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                        <div style="background:#1f2937; padding:18px; border-radius:12px; text-align:center; border:2px solid #f59e0b;">
                            <span style="font-size:1.3em; color:#f59e0b;">🔥 DIAGONAL</span><br>
                            <span style="font-size:2.1em; font-weight:900;">{num-1}{opposite} &nbsp; {num+1}{opposite}</span><br>
                            <span style="font-size:0.85em; opacity:0.7;">Climax / Drop très fort</span>
                        </div>
                    """, unsafe_allow_html=True)
                # ======================================================================================================

                # Le reste de ton affichage original (tuning, play button, mode grec, power scores, timeline, radar)
                m2, m3 = st.columns(2)
                with m2: 
                    st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{analysis_data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                with m3:
                    btn_id = f"play_{hash(analysis_data['name'])}"
                    components.html(f"""
                        <button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🎹 TESTER L'ACCORD</button>
                        <script>{get_chord_js(btn_id, analysis_data['key'])}</script>
                    """, height=110)

                # Mode grec, power scores, timeline + radar... (tout le reste de ton code original reste ici)
                raw_mode = analysis_data.get('modal_raw_mode', 'ionian')
                # ... (le reste du code original pour mode, power, colonnes c1 c2, hr, etc.)

                st.markdown("<hr style='border-color: #30363d; margin-bottom:40px;'>", unsafe_allow_html=True)

            del timeline, chroma
            gc.collect()

    st.session_state.analyzing = False
    global_status.success("Tous les fichiers ont été analysés avec succès !")
    gc.collect()

# ====================== SIDEBAR AVEC EXPORT JSON GLOBAL ======================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Sniper Control")

    if st.button("💾 Exporter TOUTES les tonalités (JSON)"):
        all_data = {}
        for fname, data in st.session_state.analyses.items():
            all_data[fname] = {
                "tonalite_principale": data['key'],           # ta consonance par défaut
                "camelot": data['camelot'],
                "confiance": data['conf'],
                "decisions": {
                    "consonance": data['key'],
                    "sniper_final": data['confiance_pure'],
                    "dominante": data['dominant_key']
                },
                "mode": data.get('modal_key'),
                "accordage": data['tuning']
            }
        
        json_str = json.dumps(all_data, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Télécharger JSON complet",
            data=json_str,
            file_name="RCDJ228_TONALITES_COMPLETES.json",
            mime="application/json"
        )

    if st.button("🧹 Vider la file d'analyse"):
        for data in list(st.session_state.analyses.values()):
            if 'temp_dir' in data and os.path.exists(data['temp_dir']):
                shutil.rmtree(data['temp_dir'])
        st.session_state.analyses = {}
        st.session_state.analyzing = False
        gc.collect()
        st.rerun()
# ============================================================================
