import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import re

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="SERS Plotter", layout="wide")

st.title("Generátor SERS Spekter pro Publikace 🧪")
st.markdown("""
Tato aplikace převede vaše `.txt` data z Ramanova mikroskopu na **offsetové (waterfall) grafy**.
Výstup je vektorový (`.svg`), připravený pro finální úpravy v Adobe Illustratoru.
""")

# --- NASTAVENÍ V BOČNÍM PANELU ---
st.sidebar.header("Nastavení grafu")

# Slidery pro interaktivní úpravu
offset_val = st.sidebar.number_input("Offset (posun na ose Y)", value=2000, step=100)
voltage_step = st.sidebar.number_input("Krok napětí (např. po 100 mV)", value=100, step=50)
x_min, x_max = st.sidebar.slider("Rozsah osy X (cm-1)", 0, 4000, (300, 1800))
line_width = st.sidebar.slider("Tloušťka čáry", 0.5, 3.0, 1.5)
font_size = st.sidebar.slider("Velikost písma", 8, 20, 12)

st.sidebar.subheader("Detekce píků")
show_peaks = st.sidebar.checkbox("Ukázat píky", value=True)
peak_prominence = st.sidebar.slider("Citlivost píků (Prominence)", 10, 500, 100)

# --- FUNKCE PRO ZPRACOVÁNÍ ---

def get_voltage_from_filename(filename):
    """Vytáhne napětí z názvu souboru (hledá číslo před 'mV')"""
    match = re.search(r'([-\d]+)mV', filename)
    if match:
        return int(match.group(1))
    return None

def load_data(uploaded_file):
    """Načte data z nahraného souboru"""
    try:
        # Přečteme soubor jako CSV, oddělovač je mezera/tabulátor
        # skiprows=1 často pomáhá, pokud je tam hlavička '[source...]'
        # Ale bezpečnější je načíst vše a zahodit nečísla
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, engine='python')
        
        # Vezmeme první dva sloupce
        df = df.iloc[:, :2]
        df.columns = ['x', 'y']
        
        # Převedeme na čísla, texty se změní na NaN
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna()
        
        # Seřadíme podle X
        df = df.sort_values(by='x')
        
        return df['x'].values, df['y'].values
    except Exception as e:
        st.error(f"Chyba u souboru {uploaded_file.name}: {e}")
        return None, None

# --- HLAVNÍ LOGIKA ---

uploaded_files = st.file_uploader("Nahrajte .txt soubory (můžete vybrat více najednou)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    # 1. Zpracování souborů a seřazení
    data_list = []
    
    for uploaded_file in uploaded_files:
        volts = get_voltage_from_filename(uploaded_file.name)
        
        # Pokud se nepodařilo najít napětí, zeptáme se uživatele nebo to přeskočíme
        # Pro zjednodušení teď bereme jen ty, co mají v názvu "mV"
        if volts is not None:
            if abs(volts) % voltage_step == 0:
                # Musíme resetovat pointer souboru, aby šel přečíst
                uploaded_file.seek(0)
                data_list.append({'file': uploaded_file, 'volts': volts, 'name': uploaded_file.name})
    
    # Seřadit podle napětí (sestupně: 0, -100, -200...)
    data_list.sort(key=lambda x: x['volts'], reverse=True)
    
    if not data_list:
        st.warning("Žádné soubory neodpovídají filtru napětí (zkontrolujte 'Krok napětí' v menu).")
    else:
        st.success(f"Zpracovávám {len(data_list)} spekter.")

        # 2. Vykreslení grafu
        # Nastavení pro Illustrator
        plt.rcParams['font.family'] = 'Arial' # Pozor: Streamlit běží na Linuxu, Arial tam nemusí být. Matplotlib použije náhradu (DejaVu Sans), ale v SVG bude definováno jako sans-serif.
        plt.rcParams['svg.fonttype'] = 'none' # Text zůstane textem
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = 1.5
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Barvy
        colors = plt.cm.jet(np.linspace(0, 1, len(data_list)))
        
        for i, item in enumerate(data_list):
            x, y = load_data(item['file'])
            
            if x is None: continue
            
            # Ořez dat
            mask = (x >= x_min) & (x <= x_max)
            x_crop = x[mask]
            y_crop = y[mask]
            
            # Vyhlazení (volitelné)
            if len(y_crop) > 11:
                y_smooth = savgol_filter(y_crop, window_length=11, polyorder=3)
            else:
                y_smooth = y_crop

            # Offset
            y_shifted = y_smooth + (i * offset_val)
            
            # Plot
            ax.plot(x_crop, y_shifted, color=colors[i], linewidth=line_width, label=f"{item['volts']} mV")
            
            # Popisek napětí vpravo
            ax.text(x_crop[-1] + 20, y_shifted[-1], f"{item['volts']} mV", 
                    color=colors[i], va='center', fontsize=font_size, fontweight='bold')
            
            # Píky (pouze pro první a poslední, aby nebyl chaos)
            if show_peaks and (i == 0 or i == len(data_list)-1):
                peaks, _ = find_peaks(y_shifted, prominence=peak_prominence, distance=50)
                for p in peaks:
                    px = x_crop[p]
                    py = y_shifted[p]
                    # Kreslení značky píku
                    ax.plot([px, px], [py + 50, py + (offset_val*0.15)], color='black', lw=0.5)
                    ax.text(px, py + (offset_val*0.2), f"{int(px)}", rotation=90, ha='center', va='bottom', fontsize=font_size-2)

        # Design os
        ax.set_xlabel("Ramanův posun (cm$^{-1}$)")
        ax.set_ylabel("Intenzita (a.u.)")
        ax.set_xlim(x_min, x_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([]) # Skrýt čísla na ose Y
        
        st.pyplot(fig)
        
        # 3. Stažení
        import io
        fn = "SERS_spektra.svg"
        img = io.BytesIO()
        plt.savefig(img, format='svg', bbox_inches='tight')
        
        st.download_button(
            label="📥 Stáhnout jako SVG (pro Illustrator)",
            data=img,
            file_name=fn,
            mime="image/svg+xml"
        )
