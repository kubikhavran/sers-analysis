import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import re
import io
import plotly.graph_objects as go

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="SERS Plotter v5", layout="wide")

st.title("Generátor SERS Spekter pro Publikace 🧪")
st.markdown("""
**v5.0**: Výběr barevných palet, vypínání vodících čar a větší rozsah pro pozici popisků.
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
    
    # Seřadíme sestupně podle napětí
    all_spectra.sort(key=lambda x: x['volts'], reverse=True)
    options = [s['label'] for s in all_spectra]

    # --- BOČNÍ PANEL ---
    st.sidebar.header("Nastavení grafu")
    
    # VÝBĚR SPEKTER
    with st.sidebar.expander("🔎 Výběr spekter", expanded=True):
        auto_step = st.number_input("Předvybrat krok (mV)", value=100, step=10)
        default_selection = [s['label'] for s in all_spectra if abs(s['volts']) % auto_step == 0]
        selected_labels = st.multiselect("Zvolte spektra:", options=options, default=default_selection)
    
    final_data_list = [s for s in all_spectra if s['label'] in selected_labels]

    # VZHLED A BARVY
    with st.sidebar.expander("🎨 Vzhled a Barvy", expanded=True):
        # Výběr palety
        palette_name = st.selectbox(
            "Barevná paleta", 
            ["jet", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "bwr", "rainbow", "nipy_spectral"],
            index=0
        )
        offset_val = st.number_input("Offset (posun Y)", value=2000, step=100)
        x_range = st.slider("Rozsah osy X", 0, 4000, (300, 1800))
        line_width = st.slider("Tloušťka čáry", 0.5, 3.0, 1.5)
        font_size = st.slider("Velikost písma os", 8, 20, 14)

    # POPISKY PÍKŮ
    with st.sidebar.expander("📍 Popisky píků", expanded=True):
        peak_label_size = st.slider("Velikost písma popisků", 8, 30, 14)
        # ZVĚTŠENÝ ROZSAH SLIDERU (až 5000)
        label_height_offset = st.slider("Výška popisků nad píkem", 50, 5000, 500, step=50)
        show_peak_lines = st.checkbox("Zobrazit vodící čáry k píkům", value=True)
        
        st.divider()
        show_auto_peaks = st.checkbox("Automatické popisky (jen horní)", value=True)
        peak_prominence = st.slider("Citlivost automatu", 10, 2000, 100)
        
        st.markdown("**Manuální píky**")
        manual_peaks_input = st.text_input("Polohy píků (např. 1000, 1580):", "")

    manual_peaks = []
    if manual_peaks_input:
        try:
            manual_peaks = [int(float(x.strip())) for x in manual_peaks_input.split(',') if x.strip()]
        except:
            pass

    if not final_data_list:
        st.warning("Vyberte alespoň jedno spektrum.")
    else:
        # Generování barev pro obě knihovny (Matplotlib i Plotly)
        # Získáme objekt colormap z matplotlibu
        cmap = plt.get_cmap(palette_name)
        # Vygenerujeme N barev rovnoměrně rozložených
        mpl_colors = cmap(np.linspace(0, 1, len(final_data_list)))
        # Pro Plotly potřebujeme hex stringy
        plotly_colors = [mcolors.to_hex(c) for c in mpl_colors]

        # ==========================================
        # 1. INTERAKTIVNÍ GRAF (PLOTLY)
        # ==========================================
        with st.expander("🔍 Interaktivní náhled (klikni pro rozbalení)", expanded=True):
            fig_interactive = go.Figure()
            
            for i, item in enumerate(final_data_list):
                x, y = load_data(item['file'])
                if x is None: continue
                mask = (x >= x_range[0]) & (x <= x_range[1])
                x_crop = x[mask]
                y_crop = y[mask]
                if len(y_crop) > 11:
                    y_crop = savgol_filter(y_crop, window_length=11, polyorder=3)
                y_shifted = y_crop + (i * offset_val)
                
                fig_interactive.add_trace(go.Scatter(
                    x=x_crop, y=y_shifted, mode='lines', name=f"{item['volts']} mV",
                    line=dict(width=2, color=plotly_colors[i]),
                    hovertemplate='<b>%{x:.1f} cm⁻¹</b><br>Intenzita: %{y:.1f}'
                ))

            fig_interactive.update_layout(
                height=500, xaxis_title="Ramanův posun (cm⁻¹)", yaxis_title="Intenzita (a.u.)",
                hovermode="x unified", template="plotly_dark", showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_interactive, use_container_width=True)


        # ==========================================
        # 2. STATICKÝ GRAF (MATPLOTLIB)
        # ==========================================
        st.subheader("📄 Finální náhled pro Illustrator")
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = 1.5
        
        # Dynamická výška grafu
        fig, ax = plt.subplots(figsize=(10, 8 + (len(final_data_list)*0.5)))
        
        top_spectrum_index = len(final_data_list) - 1

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
            
            # Použití barvy z vybrané palety
            ax.plot(x_crop, y_shifted, color=mpl_colors[i], linewidth=line_width, label=f"{item['volts']} mV")
            
            # Popisek napětí vpravo
            ax.text(x_crop[-1] + 20, y_shifted[-1], f"{item['volts']} mV", 
                    color=mpl_colors[i], va='center', fontsize=font_size, fontweight='bold')
            
            # === POPISKY PÍKŮ (JEN PRO HORNÍ SPEKTRUM) ===
            if i == top_spectrum_index:
                
                # Funkce pro vykreslení jednoho popisku
                def draw_peak_label(pos_x, pos_y, color_line='black', color_text='black', weight='normal'):
                    # Vodící čára (pokud je zapnutá)
                    if show_peak_lines:
                        ax.plot([pos_x, pos_x], [pos_y + 50, pos_y + label_height_offset - 50], 
                                color=color_line, lw=0.5, alpha=0.8)
                    
                    # Text
                    ax.text(pos_x, pos_y + label_height_offset, f"{int(pos_x)}", 
                            rotation=90, ha='center', va='bottom', fontsize=peak_label_size, 
                            color=color_text, fontweight=weight)

                # 1. Automatické
                if show_auto_peaks:
                    peaks, _ = find_peaks(y_shifted, prominence=peak_prominence, distance=50)
                    for p in peaks:
                        px = x_crop[p]
                        py = y_shifted[p]
                        draw_peak_label(px, py)

                # 2. Manuální
                if manual_peaks:
                    for target_x in manual_peaks:
                        idx_approx = find_nearest_idx(x_crop, target_x)
                        window = 30
                        start = max(0, idx_approx - window)
                        end = min(len(x_crop), idx_approx + window)
                        
                        if start < end:
                            local_max_idx = start + np.argmax(y_shifted[start:end])
                            px = x_crop[local_max_idx]
                            py = y_shifted[local_max_idx]
                            draw_peak_label(px, py, color_line='red', color_text='red', weight='bold')

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
