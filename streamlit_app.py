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
st.set_page_config(page_title="SERS Plotter v11", layout="wide")

st.title("Generátor SERS Spekter pro Publikace 🧪")
st.markdown("""
**v11.0**: Řízení směru řazení (Stacking), přidávání záporných znamének a čisté popisky bez 'Reverse'.
""")

# --- POMOCNÉ FUNKCE ---

def get_voltage_from_filename(filename):
    """Vytáhne poslední číslo před 'mV'."""
    matches = re.findall(r'([-\d]+)mV', filename)
    if matches:
        return int(matches[-1])
    return None

def detect_scan_direction(filename):
    """Rozpozná směr skenu podle klíčového slova v názvu."""
    filename_lower = filename.lower()
    if "reverse" in filename_lower or "zp" in filename_lower or "back" in filename_lower:
        return "reverse"
    return "forward"

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
    uploaded_files = st.file_uploader("1. Nahrajte VŠECHNY .txt soubory", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    
    # --- 1. DATABÁZE SOUBORŮ ---
    # Nejdřív načteme metadata o všech souborech
    all_files_meta = []
    for f in uploaded_files:
        raw_volts = get_voltage_from_filename(f.name)
        direction = detect_scan_direction(f.name)
        val = raw_volts if raw_volts is not None else 0
        
        all_files_meta.append({
            'file': f, 
            'raw_volts': val, 
            'direction': direction,
            'filename': f.name
        })

    # --- 2. BOČNÍ PANEL: FILTRACE A ÚPRAVA ---
    st.sidebar.header("1. Výběr a Úprava Dat")
    
    with st.sidebar.expander("📂 Filtr Skenů a Znamínka", expanded=True):
        # A) FILTR SMĚRU
        scan_filter = st.radio("Vyberte typ skenu:", ["Dopředný (Forward)", "Zpětný (Reverse)", "Všechny"], index=0)
        
        # Filtrace seznamu
        if scan_filter == "Dopředný (Forward)":
            current_batch = [x for x in all_files_meta if x['direction'] == 'forward']
        elif scan_filter == "Zpětný (Reverse)":
            current_batch = [x for x in all_files_meta if x['direction'] == 'reverse']
        else:
            current_batch = all_files_meta
            
        st.caption(f"Vybráno {len(current_batch)} souborů.")

        st.divider()

        # B) ÚPRAVA HODNOT (ZNAMÍNKO)
        force_minus = st.checkbox("Vynutit záporné znamínko (-)", value=True, help="Přidá mínus před hodnoty (např. 50 -> -50). Nula zůstane 0.")
        
        # C) ŘAZENÍ (STACKING)
        # Logika: Uživatel chce ovládat, co je nahoře a co dole.
        stack_order = st.radio("Pořadí spekter (Stacking):", 
                               ["Od 0 (dole) do Max (nahoře)", 
                                "Od Max (dole) do 0 (nahoře)"])
        
        # Aplikace logiky na data
        processed_batch = []
        for item in current_batch:
            # 1. Aplikace znamínka
            final_volts = item['raw_volts']
            if force_minus and final_volts > 0:
                final_volts = -final_volts
            
            # 2. Vytvoření čistého popisku (bez 'Reverse')
            label = f"{final_volts} mV"
            
            # Uložíme si upravená data
            new_item = item.copy()
            new_item['volts'] = final_volts
            new_item['label'] = label
            processed_batch.append(new_item)

        # 3. Seřazení seznamu podle voltů
        # Nejdřív seřadíme matematicky
        processed_batch.sort(key=lambda x: x['volts']) # -1000, -900 ... 0
        
        # Pak otočíme podle přání uživatele
        # Graf vykresluje index 0 dole, index N nahoře.
        if stack_order == "Od Max (dole) do 0 (nahoře)":
            # Chceme -1000 dole, 0 nahoře (pokud jsou záporné)
            # Matematické řazení (-1000 -> 0) je přesně to, co chceme pro plot (0-tý prvek je dole)
            pass 
        else:
            # Chceme 0 dole, -1000 nahoře
            processed_batch.reverse()

        # D) RYCHLÝ VÝBĚR (KROK)
        auto_step = st.number_input("Rychlý výběr - krok (mV)", value=100, step=10)
        # Filtrujeme podle absolutní hodnoty původního napětí, aby to fungovalo i po přidání mínusu
        default_selection = [s['label'] for s in processed_batch if abs(s['raw_volts']) % auto_step == 0]

    # MULTISELECT
    options = [s['label'] for s in processed_batch]
    
    if not options:
        st.warning("Žádná data k zobrazení.")
        final_data_list = []
    else:
        st.write(f"### Výběr spekter ({scan_filter})")
        # Multiselect musí zachovat pořadí z 'options', které už je seřazené podle 'stack_order'
        selected_labels = st.multiselect("Zahrnout do grafu:", options=options, default=default_selection)
        
        # Finální filtrování, zachováváme pořadí z processed_batch
        final_data_list = [s for s in processed_batch if s['label'] in selected_labels]


    # --- 3. VZHLED A EXPORT ---
    st.sidebar.header("2. Vzhled a Export")
    with st.sidebar.expander("📏 Rozměry obrázku", expanded=False):
        col_w, col_h = st.columns(2)
        with col_w: img_width_px = st.number_input("Šířka (px)", value=1200, step=100)
        with col_h: img_height_px = st.number_input("Výška (px)", value=1000, step=100)
        img_dpi = st.number_input("DPI", value=150, step=50)
        figsize_w = img_width_px / img_dpi
        figsize_h = img_height_px / img_dpi

    with st.sidebar.expander("🎨 Grafika a Osy", expanded=False):
        palette_name = st.selectbox("Paleta", ["jet", "viridis", "plasma", "inferno", "coolwarm", "bwr", "rainbow"], index=0)
        offset_val = st.number_input("Offset (posun Y)", value=2000, step=100)
        xlabel_text = st.text_input("Popis osy X", "Ramanův posun (cm⁻¹)")
        ylabel_text = st.text_input("Popis osy Y", "Intenzita (a.u.)")
        x_range = st.slider("Rozsah osy X", 0, 4000, (300, 1800))
        invert_x = st.checkbox("Invertovat osu X", value=False)
        line_width = st.slider("Tloušťka čáry", 0.5, 3.0, 1.5)
        font_size = st.slider("Velikost písma", 8, 30, 14)

    # --- 4. PÍKY ---
    st.sidebar.header("3. Správa Píků")
    with st.sidebar.expander("📍 Editace píků", expanded=False):
        st.info("Píky se zobrazují u nejvyššího spektra v grafu.")
        peak_label_size = st.slider("Velikost popisků píků", 8, 30, 14)
        label_height_offset = st.slider("Výška popisků nad píkem", 50, 5000, 500, step=50)
        show_peak_lines = st.checkbox("Zobrazit vodící čáry", value=True)
        st.divider()
        use_auto = st.checkbox("Automatická detekce", value=True)
        prominence = st.slider("Citlivost automatu", 10, 2000, 100)
        manual_add_str = st.text_input("➕ Přidat píky (např. 1001):", "")
        manual_remove_str = st.text_input("➖ Smazat píky (např. 220):", "")

    manual_adds = [int(float(x.strip())) for x in manual_add_str.split(',') if x.strip()] if manual_add_str else []
    manual_removes = [int(float(x.strip())) for x in manual_remove_str.split(',') if x.strip()] if manual_remove_str else []

    # --- VYKRESLOVÁNÍ ---
    if final_data_list:
        cmap = plt.get_cmap(palette_name)
        mpl_colors = cmap(np.linspace(0, 1, len(final_data_list)))
        plotly_colors = [mcolors.to_hex(c) for c in mpl_colors]

        # 1. INTERAKTIVNÍ
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
            
            fig_int.update_layout(height=500, xaxis_title=xlabel_text, yaxis_title=ylabel_text, 
                                  hovermode="x unified", template="plotly_dark",
                                  xaxis=dict(autorange="reversed" if invert_x else True))
            st.plotly_chart(fig_int, use_container_width=True)

        # 2. STATICKÝ
        st.subheader(f"📄 Finální výstup ({scan_filter})")
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = 1.5
        
        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=img_dpi)
        
        # Určení horního spektra pro píky (vždy to poslední vykreslené, tedy index -1)
        top_idx = len(final_data_list) - 1

        for i, item in enumerate(final_data_list):
            x, y = load_data(item['file'])
            if x is None: continue
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_c, y_c = x[mask], y[mask]
            if len(y_c) > 11: y_c = savgol_filter(y_c, 11, 3)
            y_s = y_c + (i * offset_val)
            
            ax.plot(x_c, y_s, color=mpl_colors[i], lw=line_width, label=item['label'])
            
            # Popisky spekter (Vpravo)
            trans = ax.get_yaxis_transform()
            y_lbl = y_s[0] if invert_x else y_s[-1]
            ax.text(1.02, y_lbl, item['label'], color=mpl_colors[i], va='center', ha='left', 
                    fontsize=font_size, fontweight='bold', transform=trans, clip_on=False)
            
            # Píky (jen horní spektrum)
            if i == top_idx:
                final_peaks = []
                if use_auto:
                    p, _ = find_peaks(y_s, prominence=prominence, distance=30)
                    final_peaks.extend(p)
                for ux in manual_adds:
                    idx = find_nearest_idx(x_c, ux)
                    w = 10
                    s, e = max(0, idx-w), min(len(x_c), idx+w)
                    if s<e:
                        best = s + np.argmax(y_s[s:e])
                        if not any(abs(existing-best)<5 for existing in final_peaks): final_peaks.append(best)
                
                valid = [p for p in final_peaks if not any(abs(x_c[p]-r)<15 for r in manual_removes)]
                
                for p in valid:
                    px, py = x_c[p], y_s[p]
                    if show_peak_lines:
                        ax.plot([px, px], [py + 50, py + label_height_offset - 50], color='black', lw=0.5, alpha=0.8)
                    ax.text(px, py + label_height_offset, f"{int(px)}", rotation=90, ha='center', va='bottom', fontsize=peak_label_size)

        ax.set_xlabel(xlabel_text)
        ax.set_ylabel(ylabel_text)
        ax.set_xlim(x_range[1], x_range[0]) if invert_x else ax.set_xlim(x_range[0], x_range[1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        
        st.pyplot(fig)
        
        c1, c2 = st.columns(2)
        svg_io = io.BytesIO()
        plt.savefig(svg_io, format='svg', bbox_inches='tight', dpi=img_dpi)
        c1.download_button("📥 Stáhnout SVG", svg_io, "SERS_output.svg", "image/svg+xml")
        
        png_io = io.BytesIO()
        plt.savefig(png_io, format='png', bbox_inches='tight', dpi=img_dpi)
        c2.download_button("📥 Stáhnout PNG", png_io, "SERS_output.png", "image/png")
