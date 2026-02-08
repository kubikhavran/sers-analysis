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
st.set_page_config(page_title="SERS Plotter v7", layout="wide")

st.title("Generátor SERS Spekter pro Publikace 🧪")
st.markdown("""
**v7.0**: Pokročilá správa píků (přidávání/mazání), inverze osy X a kontrola řazení souborů.
""")

# --- POMOCNÉ FUNKCE ---

def get_voltage_from_filename(filename):
    """Zkusí najít číslo před 'mV'."""
    matches = re.findall(r'([-\d]+)mV', filename)
    if matches:
        return int(matches[-1])
    return None

def natural_keys(text):
    """Funkce pro přirozené řazení (aby 2.txt bylo před 10.txt)."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

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
    
    # --- 1. PŘÍPRAVA DAT A POPISKŮ ---
    st.sidebar.header("1. Data a Popisky")
    
    with st.sidebar.expander("📂 Zdroje popisků a Řazení", expanded=True):
        label_mode = st.radio("Způsob generování popisků:", ["Automaticky (z názvu)", "Manuálně (sekvence)"])
        
        all_spectra = []
        
        if label_mode == "Automaticky (z názvu)":
            # Auto režim
            for f in uploaded_files:
                volts = get_voltage_from_filename(f.name)
                # Pokud nenajde mV, dá tam 0, aby to nespadlo, ale upozorní
                val = volts if volts is not None else 0
                label = f"{val} mV" if volts is not None else f"??? ({f.name})"
                
                all_spectra.append({
                    'file': f, 'volts': val, 'label': label, 'filename': f.name
                })
            # Seřadit podle hodnoty napětí
            all_spectra.sort(key=lambda x: x['volts'], reverse=True)
            
            # Filtr
            auto_step = st.number_input("Filtr kroku (mV)", value=100, step=10)
            default_selection = [s['label'] for s in all_spectra if abs(s['volts']) % auto_step == 0]
            
        else:
            # Manuální režim
            sort_type = st.selectbox("Seřadit soubory podle:", ["Jména (A-Z)", "Jména (Z-A)"])
            
            # Seřazení souborů
            sorted_files = sorted(uploaded_files, key=lambda x: x.name)
            if sort_type == "Jména (Z-A)":
                sorted_files.reverse()
                
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                start_val = st.number_input("Start", value=0)
                step_val = st.number_input("Krok", value=-100)
            with col2:
                unit_val = st.text_input("Jednotka", value="mV")
            
            # Přiřazení
            for i, f in enumerate(sorted_files):
                calc_val = start_val + (i * step_val)
                label = f"{calc_val} {unit_val}"
                all_spectra.append({
                    'file': f, 'volts': calc_val, 'label': label, 'filename': f.name
                })
            
            default_selection = [s['label'] for s in all_spectra]
            
            # KONTROLNÍ TABULKA
            st.info("Zkontrolujte, zda popisky sedí k souborům:")
            df_preview = pd.DataFrame(all_spectra)[['filename', 'label']]
            st.dataframe(df_preview, height=150, hide_index=True)

    # VÝBĚR KONEČNÝCH SPEKTER
    options = [s['label'] for s in all_spectra]
    if not options:
        st.error("Žádná data.")
        final_data_list = []
    else:
        # Multiselect
        selected_labels = st.sidebar.multiselect("Vyberte spektra k zobrazení:", options=options, default=default_selection)
        final_data_list = [s for s in all_spectra if s['label'] in selected_labels]

    # --- 2. VZHLED ---
    st.sidebar.header("2. Vzhled")
    with st.sidebar.expander("🎨 Grafika", expanded=False):
        palette_name = st.selectbox("Paleta", ["jet", "viridis", "plasma", "inferno", "coolwarm", "bwr", "rainbow"], index=0)
        offset_val = st.number_input("Offset (posun Y)", value=2000, step=100)
        
        # Range slider
        x_range = st.slider("Rozsah osy X", 0, 4000, (300, 1800))
        invert_x = st.checkbox("Invertovat osu X (zprava doleva)", value=False)
        
        line_width = st.slider("Tloušťka čáry", 0.5, 3.0, 1.5)
        font_size = st.slider("Velikost písma os", 8, 20, 14)

    # --- 3. PÍKY (Unified System) ---
    st.sidebar.header("3. Správa Píků")
    with st.sidebar.expander("📍 Editace píků", expanded=True):
        st.markdown("Tato nastavení se aplikují na **horní spektrum**.")
        
        # Nastavení zobrazení
        peak_label_size = st.slider("Velikost písma popisků", 8, 30, 14)
        label_height_offset = st.slider("Výška popisků nad píkem", 50, 5000, 500, step=50)
        show_peak_lines = st.checkbox("Zobrazit vodící čáry", value=True)
        
        st.markdown("---")
        # 1. Automatika
        use_auto = st.checkbox("Použít automatickou detekci", value=True)
        prominence = st.slider("Citlivost automatu", 10, 1000, 100)
        
        # 2. Manuální přidání
        manual_add_str = st.text_input("➕ Přidat píky (např. 1001, 1580):", help="Najde nejvyšší bod v blízkém okolí zadané hodnoty.")
        
        # 3. Manuální smazání
        manual_remove_str = st.text_input("➖ Smazat píky (např. 220, 1023):", help="Smaže píky v blízkosti těchto hodnot.")

    # Zpracování vstupů píků
    manual_adds = []
    if manual_add_str:
        manual_adds = [int(float(x.strip())) for x in manual_add_str.split(',') if x.strip()]
        
    manual_removes = []
    if manual_remove_str:
        manual_removes = [int(float(x.strip())) for x in manual_remove_str.split(',') if x.strip()]


    # --- VYKRESLOVÁNÍ ---
    if not final_data_list:
        st.warning("Vyberte alespoň jedno spektrum.")
    else:
        # Barvy
        cmap = plt.get_cmap(palette_name)
        mpl_colors = cmap(np.linspace(0, 1, len(final_data_list)))
        plotly_colors = [mcolors.to_hex(c) for c in mpl_colors]

        # ----------------------------------------------
        # INTERAKTIVNÍ NÁHLED
        # ----------------------------------------------
        with st.expander("🔍 Interaktivní náhled (pro zjištění polohy)", expanded=True):
            fig_int = go.Figure()
            for i, item in enumerate(final_data_list):
                x, y = load_data(item['file'])
                if x is None: continue
                # Filtrujeme data podle rozsahu
                mask = (x >= x_range[0]) & (x <= x_range[1])
                x_crop, y_crop = x[mask], y[mask]
                if len(y_crop) > 11: y_crop = savgol_filter(y_crop, 11, 3)
                
                y_shift = y_crop + (i * offset_val)
                fig_int.add_trace(go.Scatter(x=x_crop, y=y_shift, mode='lines', name=item['label'], line=dict(color=plotly_colors[i])))
            
            fig_int.update_layout(
                height=500, xaxis_title="cm⁻¹", yaxis_title="Intenzita", 
                hovermode="x unified", template="plotly_dark",
                xaxis=dict(autorange="reversed" if invert_x else True)
            )
            st.plotly_chart(fig_int, use_container_width=True)

        # ----------------------------------------------
        # STATICKÝ GRAF PRO EXPORT
        # ----------------------------------------------
        st.subheader("📄 Finální výstup")
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = 1.5
        
        fig, ax = plt.subplots(figsize=(10, 8 + (len(final_data_list)*0.5)))
        
        top_idx = len(final_data_list) - 1 # Index horního spektra

        for i, item in enumerate(final_data_list):
            x, y = load_data(item['file'])
            if x is None: continue
            
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_c, y_c = x[mask], y[mask]
            if len(y_c) > 11: y_c = savgol_filter(y_c, 11, 3)
            y_s = y_c + (i * offset_val)
            
            # Vykreslení spektra
            ax.plot(x_c, y_s, color=mpl_colors[i], lw=line_width, label=item['label'])
            
            # Popisek napětí vpravo
            lbl_pos_x = x_c[0] if invert_x else x_c[-1]
            # Offset textu trochu doleva nebo doprava podle inverze
            txt_offset = -20 if invert_x else 20
            ha_align = 'right' if invert_x else 'left'
            
            ax.text(lbl_pos_x + txt_offset, y_s[0 if invert_x else -1], item['label'], 
                    color=mpl_colors[i], va='center', ha=ha_align, fontsize=font_size, fontweight='bold')
            
            # === LOGIKA PÍKŮ (Jen horní spektrum) ===
            if i == top_idx:
                # 1. Získat kandidáty z automatiky
                final_peaks_indices = []
                if use_auto:
                    peaks, _ = find_peaks(y_s, prominence=prominence, distance=30)
                    final_peaks_indices.extend(peaks)
                
                # 2. Zpracovat MANUÁLNÍ PŘIDÁNÍ (Snap to max)
                for user_x in manual_adds:
                    # Najdi index v datech, který je blízko zadanému X
                    idx_approx = find_nearest_idx(x_c, user_x)
                    
                    # Hledáme lokální maximum ve velmi úzkém okně (+/- 10 bodů), 
                    # abychom trefili přesně vrchol čáry
                    window = 10 
                    start = max(0, idx_approx - window)
                    end = min(len(x_c), idx_approx + window)
                    
                    if start < end:
                        # Najdi relativní index maxima v okně a přičti start
                        local_max_rel = np.argmax(y_s[start:end])
                        best_idx = start + local_max_rel
                        
                        # Přidáme, pokud tam ještě není (s tolerancí)
                        if not any(abs(existing - best_idx) < 5 for existing in final_peaks_indices):
                            final_peaks_indices.append(best_idx)

                # 3. Zpracovat MANUÁLNÍ SMAZÁNÍ
                # Filtrujeme ty, jejichž X pozice je blízko něčemu v manual_removes
                valid_indices = []
                for p_idx in final_peaks_indices:
                    p_x = x_c[p_idx]
                    # Pokud je pík blízko (do 15 cm-1) nějaké hodnotě v remove listu, smazat ho
                    is_removed = any(abs(p_x - rem_val) < 15 for rem_val in manual_removes)
                    if not is_removed:
                        valid_indices.append(p_idx)
                
                # 4. VYKRESLENÍ VŠECH PÍKŮ (Jednotný styl)
                for p_idx in valid_indices:
                    px = x_c[p_idx]
                    py = y_s[p_idx]
                    
                    # Čára
                    if show_peak_lines:
                        ax.plot([px, px], [py + 50, py + label_height_offset - 50], 
                                color='black', lw=0.5, alpha=0.8)
                    # Text
                    ax.text(px, py + label_height_offset, f"{int(px)}", 
                            rotation=90, ha='center', va='bottom', fontsize=peak_label_size)

        ax.set_xlabel("Ramanův posun (cm$^{-1}$)")
        ax.set_ylabel("Intenzita (a.u.)")
        
        if invert_x:
            ax.set_xlim(x_range[1], x_range[0]) # Od max k min
        else:
            ax.set_xlim(x_range[0], x_range[1])
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        
        st.pyplot(fig)
        
        # Download
        fn = "SERS_v7_final.svg"
        img = io.BytesIO()
        plt.savefig(img, format='svg', bbox_inches='tight')
        st.download_button("📥 Stáhnout SVG", img, fn, "image/svg+xml")
