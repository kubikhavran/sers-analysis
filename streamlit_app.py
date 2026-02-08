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
st.set_page_config(page_title="SERS Plotter v8", layout="wide")

st.title("Generátor SERS Spekter pro Publikace 🧪")
st.markdown("""
**v8.0**: Fixace popisků spekter mimo graf (zamezení překryvu) a editace názvů os.
""")

# --- POMOCNÉ FUNKCE ---

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
        df = df.sort_values(by='x') # Data jsou vždy seřazena vzestupně podle X
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
    
    # --- 1. PŘÍPRAVA DAT A POPISKŮ ---
    st.sidebar.header("1. Data a Popisky")
    
    with st.sidebar.expander("📂 Zdroje popisků a Řazení", expanded=True):
        label_mode = st.radio("Způsob generování popisků:", ["Automaticky (z názvu)", "Manuálně (sekvence)"])
        
        all_spectra = []
        
        if label_mode == "Automaticky (z názvu)":
            for f in uploaded_files:
                volts = get_voltage_from_filename(f.name)
                val = volts if volts is not None else 0
                label = f"{val} mV" if volts is not None else f"??? ({f.name})"
                all_spectra.append({'file': f, 'volts': val, 'label': label, 'filename': f.name})
            all_spectra.sort(key=lambda x: x['volts'], reverse=True)
            
            auto_step = st.number_input("Filtr kroku (mV)", value=100, step=10)
            default_selection = [s['label'] for s in all_spectra if abs(s['volts']) % auto_step == 0]
            
        else:
            sort_type = st.selectbox("Seřadit soubory podle:", ["Jména (A-Z)", "Jména (Z-A)"])
            sorted_files = sorted(uploaded_files, key=lambda x: x.name)
            if sort_type == "Jména (Z-A)": sorted_files.reverse()
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                start_val = st.number_input("Start", value=0)
                step_val = st.number_input("Krok", value=-100)
            with col2:
                unit_val = st.text_input("Jednotka", value="mV")
            
            for i, f in enumerate(sorted_files):
                calc_val = start_val + (i * step_val)
                label = f"{calc_val} {unit_val}"
                all_spectra.append({'file': f, 'volts': calc_val, 'label': label, 'filename': f.name})
            
            default_selection = [s['label'] for s in all_spectra]

    # Výběr spekter
    options = [s['label'] for s in all_spectra]
    if not options:
        st.error("Žádná data.")
        final_data_list = []
    else:
        selected_labels = st.sidebar.multiselect("Vyberte spektra k zobrazení:", options=options, default=default_selection)
        final_data_list = [s for s in all_spectra if s['label'] in selected_labels]

    # --- 2. VZHLED ---
    st.sidebar.header("2. Vzhled")
    with st.sidebar.expander("🎨 Grafika a Osy", expanded=True):
        palette_name = st.selectbox("Paleta", ["jet", "viridis", "plasma", "inferno", "coolwarm", "bwr", "rainbow"], index=0)
        offset_val = st.number_input("Offset (posun Y)", value=2000, step=100)
        
        st.divider()
        # EDITACE NÁZVŮ OS
        xlabel_text = st.text_input("Popis osy X", "Ramanův posun (cm⁻¹)")
        ylabel_text = st.text_input("Popis osy Y", "Intenzita (a.u.)")
        
        x_range = st.slider("Rozsah osy X", 0, 4000, (300, 1800))
        invert_x = st.checkbox("Invertovat osu X (zprava doleva)", value=False) # Defaultně False
        
        line_width = st.slider("Tloušťka čáry", 0.5, 3.0, 1.5)
        font_size = st.slider("Velikost písma os", 8, 20, 14)

    # --- 3. PÍKY ---
    st.sidebar.header("3. Správa Píků")
    with st.sidebar.expander("📍 Editace píků", expanded=False):
        peak_label_size = st.slider("Velikost písma popisků", 8, 30, 14)
        label_height_offset = st.slider("Výška popisků nad píkem", 50, 5000, 500, step=50)
        show_peak_lines = st.checkbox("Zobrazit vodící čáry", value=True)
        st.divider()
        use_auto = st.checkbox("Použít automatickou detekci", value=True)
        prominence = st.slider("Citlivost automatu", 10, 1000, 100)
        manual_add_str = st.text_input("➕ Přidat píky (např. 1001, 1580):", "")
        manual_remove_str = st.text_input("➖ Smazat píky (např. 220):", "")

    manual_adds = [int(float(x.strip())) for x in manual_add_str.split(',') if x.strip()] if manual_add_str else []
    manual_removes = [int(float(x.strip())) for x in manual_remove_str.split(',') if x.strip()] if manual_remove_str else []

    # --- VYKRESLOVÁNÍ ---
    if not final_data_list:
        st.warning("Vyberte alespoň jedno spektrum.")
    else:
        cmap = plt.get_cmap(palette_name)
        mpl_colors = cmap(np.linspace(0, 1, len(final_data_list)))
        plotly_colors = [mcolors.to_hex(c) for c in mpl_colors]

        # 1. INTERAKTIVNÍ NÁHLED
        with st.expander("🔍 Interaktivní náhled", expanded=False):
            fig_int = go.Figure()
            for i, item in enumerate(final_data_list):
                x, y = load_data(item['file'])
                if x is None: continue
                mask = (x >= x_range[0]) & (x <= x_range[1])
                x_c, y_c = x[mask], y[mask]
                if len(y_c) > 11: y_c = savgol_filter(y_c, 11, 3)
                y_s = y_c + (i * offset_val)
                fig_int.add_trace(go.Scatter(x=x_c, y=y_s, mode='lines', name=item['label'], line=dict(color=plotly_colors[i])))
            
            fig_int.update_layout(
                height=500, xaxis_title=xlabel_text, yaxis_title=ylabel_text, 
                hovermode="x unified", template="plotly_dark",
                xaxis=dict(autorange="reversed" if invert_x else True)
            )
            st.plotly_chart(fig_int, use_container_width=True)

        # 2. STATICKÝ GRAF
        st.subheader("📄 Finální výstup")
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = 1.5
        
        fig, ax = plt.subplots(figsize=(10, 8 + (len(final_data_list)*0.5)))
        top_idx = len(final_data_list) - 1

        for i, item in enumerate(final_data_list):
            x, y = load_data(item['file'])
            if x is None: continue
            
            # Filtrace a vyhlazení
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_c, y_c = x[mask], y[mask]
            if len(y_c) > 11: y_c = savgol_filter(y_c, 11, 3)
            y_s = y_c + (i * offset_val)
            
            # Plot
            ax.plot(x_c, y_s, color=mpl_colors[i], lw=line_width, label=item['label'])
            
            # === CHYTRÉ POPISKY SPEKTER (Mimo graf) ===
            # Použijeme smíšenou transformaci: X souřadnice je relativní k ose (1.0 = pravý okraj), Y je data
            trans = ax.get_yaxis_transform() 
            
            # Určíme Y pozici pro popisek. 
            # Pokud je graf invertovaný (zprava doleva), vizuální konec vpravo je MINIMUM x (index 0).
            # Pokud je graf normální, vizuální konec vpravo je MAXIMUM x (index -1).
            if invert_x:
                y_label_pos = y_s[0] # Data jsou seřazena vzestupně, takže 0 je nejmenší X (vpravo při inverzi)
            else:
                y_label_pos = y_s[-1] # -1 je největší X (vpravo při normálu)

            # Vykreslíme text na pozici x=1.02 (2% za pravým okrajem grafu)
            ax.text(1.02, y_label_pos, item['label'], 
                    color=mpl_colors[i], va='center', ha='left', 
                    fontsize=font_size, fontweight='bold', transform=trans, clip_on=False)
            
            # === PÍKY ===
            if i == top_idx:
                final_peaks_indices = []
                if use_auto:
                    peaks, _ = find_peaks(y_s, prominence=prominence, distance=30)
                    final_peaks_indices.extend(peaks)
                
                for user_x in manual_adds:
                    idx_approx = find_nearest_idx(x_c, user_x)
                    window = 10 
                    start, end = max(0, idx_approx - window), min(len(x_c), idx_approx + window)
                    if start < end:
                        local_max_rel = np.argmax(y_s[start:end])
                        best_idx = start + local_max_rel
                        if not any(abs(existing - best_idx) < 5 for existing in final_peaks_indices):
                            final_peaks_indices.append(best_idx)

                valid_indices = []
                for p_idx in final_peaks_indices:
                    p_x = x_c[p_idx]
                    if not any(abs(p_x - rem_val) < 15 for rem_val in manual_removes):
                        valid_indices.append(p_idx)
                
                for p_idx in valid_indices:
                    px = x_c[p_idx]
                    py = y_s[p_idx]
                    if show_peak_lines:
                        ax.plot([px, px], [py + 50, py + label_height_offset - 50], color='black', lw=0.5, alpha=0.8)
                    ax.text(px, py + label_height_offset, f"{int(px)}", 
                            rotation=90, ha='center', va='bottom', fontsize=peak_label_size)

        # Design os
        ax.set_xlabel(xlabel_text)
        ax.set_ylabel(ylabel_text)
        
        if invert_x:
            ax.set_xlim(x_range[1], x_range[0])
        else:
            ax.set_xlim(x_range[0], x_range[1])
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) # Pravá osa je pryč, ale text tam bude držet
        ax.set_yticks([])
        
        st.pyplot(fig)
        
        fn = "SERS_v8_final.svg"
        img = io.BytesIO()
        plt.savefig(img, format='svg', bbox_inches='tight') # bbox_inches='tight' zajistí, že popisky mimo graf se neoříznou
        st.download_button("📥 Stáhnout SVG", img, fn, "image/svg+xml")
