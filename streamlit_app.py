import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import re
import io
import plotly.graph_objects as go # Nová knihovna pro interaktivní grafy

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="SERS Plotter v3", layout="wide")

st.title("Generátor SERS Spekter pro Publikace 🧪")
st.markdown("""
**v3.0**: Interaktivní náhled pro snadné odečítání polohy píků.
""")

# --- FUNKCE ---

def get_voltage_from_filename(filename):
    matches = re.findall(r'([-\d]+)mV', filename)
    if matches:
        return int(matches[-1])
    return None

def load_data(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, engine='python')
        df = df.iloc[:, :2]
        df.columns = ['x', 'y']
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna()
        df = df.sort_values(by='x')
        return df['x'].values, df['y'].values
    except Exception as e:
        return None, None

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# --- HLAVNÍ LOGIKA ---

with st.container():
    uploaded_files = st.file_uploader("1. Nahrajte .txt soubory", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    # --- PŘED-ZPRACOVÁNÍ ---
    all_spectra = []
    for f in uploaded_files:
        volts = get_voltage_from_filename(f.name)
        if volts is not None:
            all_spectra.append({
                'file': f, 'volts': volts, 'label': f"{volts} mV ({f.name})"
            })
    
    all_spectra.sort(key=lambda x: x['volts'], reverse=True)
    options = [s['label'] for s in all_spectra]

    # --- BOČNÍ PANEL ---
    st.sidebar.header("Nastavení grafu")
    
    st.sidebar.subheader("🔎 Výběr spekter")
    auto_step = st.sidebar.number_input("Předvybrat kros (mV)", value=100, step=10)
    default_selection = [s['label'] for s in all_spectra if abs(s['volts']) % auto_step == 0]
    
    selected_labels = st.sidebar.multiselect("Zvolte spektra:", options=options, default=default_selection)
    final_data_list = [s for s in all_spectra if s['label'] in selected_labels]

    st.sidebar.divider()
    offset_val = st.sidebar.number_input("Offset (posun Y)", value=2000, step=100)
    x_range = st.sidebar.slider("Rozsah osy X", 0, 4000, (300, 1800))
    
    st.sidebar.divider()
    # Zde uživatel zadá číslo, které zjistil z interaktivního grafu
    st.sidebar.markdown("### 📍 Manuální píky")
    st.sidebar.info("Najeďte myší na graf nahoře, zjistěte polohu píku (hodnota X) a napište ji sem.")
    manual_peaks_input = st.sidebar.text_input("Polohy píků (např. 1000, 1580):", "")
    
    manual_peaks = []
    if manual_peaks_input:
        try:
            manual_peaks = [int(float(x.strip())) for x in manual_peaks_input.split(',') if x.strip()]
        except:
            pass

    # Design statického grafu
    st.sidebar.divider()
    line_width = st.sidebar.slider("Tloušťka čáry", 0.5, 3.0, 1.5)
    font_size = st.sidebar.slider("Velikost písma", 8, 20, 12)
    show_auto_peaks = st.sidebar.checkbox("Automatické popisky", value=True)
    peak_prominence = st.sidebar.slider("Citlivost automatu", 10, 1000, 100)

    if not final_data_list:
        st.warning("Vyberte alespoň jedno spektrum.")
    else:
        # ==========================================
        # 1. INTERAKTIVNÍ GRAF (PLOTLY) - PRO PRÁCI
        # ==========================================
        st.subheader("🔍 Interaktivní náhled (pro zjištění polohy píků)")
        
        fig_interactive = go.Figure()
        
        # Použijeme barvy z Plotly (Spectrall nebo Jet)
        colors_plotly = [f"hsl({h},80%,50%)" for h in np.linspace(0, 240, len(final_data_list))]
        
        for i, item in enumerate(final_data_list):
            x, y = load_data(item['file'])
            if x is None: continue
            
            # Ořez a offset
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_crop = x[mask]
            y_crop = y[mask]
            if len(y_crop) > 11:
                y_crop = savgol_filter(y_crop, window_length=11, polyorder=3)
            
            y_shifted = y_crop + (i * offset_val)
            
            # Přidání čáry do interaktivního grafu
            fig_interactive.add_trace(go.Scatter(
                x=x_crop, 
                y=y_shifted, 
                mode='lines',
                name=f"{item['volts']} mV",
                line=dict(width=2, color=colors_plotly[i]),
                hovertemplate='<b>%{x:.1f} cm⁻¹</b><br>Intenzita: %{y:.1f}'
            ))

        fig_interactive.update_layout(
            height=600,
            xaxis_title="Ramanův posun (cm⁻¹)",
            yaxis_title="Intenzita (a.u.)",
            hovermode="x unified", # Ukazuje hodnoty všech spekter na daném X
            template="plotly_dark",
            showlegend=True
        )
        
        # Vykreslení interaktivního grafu
        st.plotly_chart(fig_interactive, use_container_width=True)


        # ==========================================
        # 2. STATICKÝ GRAF (MATPLOTLIB) - PRO EXPORT
        # ==========================================
        st.subheader("📄 Finální náhled pro Illustrator")
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = 1.5
        
        fig, ax = plt.subplots(figsize=(10, 8 + (len(final_data_list)*0.5)))
        colors = plt.cm.jet(np.linspace(0, 1, len(final_data_list)))
        
        for i, item in enumerate(final_data_list):
            x, y = load_data(item['file'])
            if x is None: continue
            
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_crop = x[mask]
            y_crop = y[mask]
            if len(y_crop) > 11:
                y_smooth = savgol_filter(y_crop, window_length=11, polyorder=3)
            else:
                y_smooth = y_crop
            
            y_shifted = y_smooth + (i * offset_val)
            
            ax.plot(x_crop, y_shifted, color=colors[i], linewidth=line_width, label=f"{item['volts']} mV")
            
            # Popisek napětí
            ax.text(x_crop[-1] + 20, y_shifted[-1], f"{item['volts']} mV", 
                    color=colors[i], va='center', fontsize=font_size, fontweight='bold')
            
            # Automatické píky
            if show_auto_peaks and (i == 0 or i == len(final_data_list)-1):
                peaks, _ = find_peaks(y_shifted, prominence=peak_prominence, distance=50)
                for p in peaks:
                    px = x_crop[p]
                    py = y_shifted[p]
                    ax.plot([px, px], [py + 50, py + (offset_val*0.15)], color='black', lw=0.5, alpha=0.5)
                    ax.text(px, py + (offset_val*0.2), f"{int(px)}", rotation=90, ha='center', va='bottom', fontsize=font_size-2)

            # Manuální píky (Vykreslíme je červeně)
            if manual_peaks:
                for target_x in manual_peaks:
                    # Najdeme lokální maximum v okolí zadaného bodu
                    idx_approx = find_nearest_idx(x_crop, target_x)
                    window = 30 # Hledáme +/- 30 bodů okolo
                    start = max(0, idx_approx - window)
                    end = min(len(x_crop), idx_approx + window)
                    
                    if start < end:
                        local_max_idx = start + np.argmax(y_shifted[start:end])
                        px = x_crop[local_max_idx]
                        py = y_shifted[local_max_idx]
                        
                        # Červená značka
                        ax.plot([px, px], [py + 50, py + (offset_val*0.2)], color='red', lw=1.0, linestyle='--')
                        
                        # Popisek jen nahoře, ať se to nepřekrývá
                        if i == len(final_data_list)-1:
                             ax.text(px, py + (offset_val*0.25), f"{int(px)}", rotation=90, ha='center', va='bottom', fontsize=font_size-2, color='red', fontweight='bold')

        ax.set_xlabel("Ramanův posun (cm$^{-1}$)")
        ax.set_ylabel("Intenzita (a.u.)")
        ax.set_xlim(x_range)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        
        st.pyplot(fig)
        
        fn = "SERS_output.svg"
        img = io.BytesIO()
        plt.savefig(img, format='svg', bbox_inches='tight')
        st.download_button(label="📥 Stáhnout SVG pro Illustrator", data=img, file_name=fn, mime="image/svg+xml")
