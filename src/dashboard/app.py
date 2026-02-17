"""
UrbanShield AI - Dashboard Interativo
Visualização de riscos climáticos em tempo real
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Configurar path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))
sys.path.append(str(src_dir / 'vision'))
sys.path.append(str(src_dir / 'forecasting'))
sys.path.append(str(src_dir / 'inference'))

from inference.risk_engine import RiskEngine

# Configuração da Página
st.set_page_config(
    page_title="UrbanShield AI",
    page_icon="🌍",
    layout="wide"
)

# Estilo CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Inicializar Engine (Cache para não recarregar modelos)
@st.cache_resource
def load_engine():
    return RiskEngine()

engine = load_engine()

# ==========================
# SIDEBAR
# ==========================
st.sidebar.image("https://img.icons8.com/color/96/000000/satellite-in-orbit.png", width=80)
st.sidebar.title("UrbanShield AI")
st.sidebar.markdown("Monitoramento de Riscos Climáticos Urbanos")

# Seletor de Data
forecast = engine.get_weather_forecast()
dates = forecast['date'].dt.strftime('%d/%m/%Y').tolist()
selected_date = st.sidebar.selectbox("Selecione a Data:", dates)
day_index = dates.index(selected_date)

# Seletor de Imagem (Simulação)
st.sidebar.markdown("---")
st.sidebar.subheader("Área de Análise")
image_options = {
    "Centro Cívico": "sat_000.png",
    "Parque Barigui": "sat_001.png",  # Exemplo (usará a mesma se não tiver outras)
    "Boqueirão": "sat_002.png"
}
selected_area = st.sidebar.selectbox("Região:", list(image_options.keys()))

# Upload de Imagem Personalizada
uploaded_file = st.sidebar.file_uploader("Ou carregue uma imagem:", type=["png", "jpg", "tif"])

# ==========================
# MAIN CONTENT
# ==========================
st.title(f"🌍 Análise de Risco: {selected_area}")
st.markdown(f"**Data da Previsão:** {selected_date}")

# 1. Métricas Climáticas (LSTM)
day_forecast = forecast.iloc[day_index]
temp = day_forecast['temp_pred']
rain = day_forecast['precip_pred']

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Temperatura", f"{temp:.1f}°C", delta=f"{temp-20:.1f}°C vs média")

with col2:
    st.metric("Precipitação", f"{rain:.1f} mm", delta_color="inverse")

with col3:
    if temp > 30:
        st.error("🔥 ALERTA DE CALOR")
    elif rain > 20:
        st.info("💧 ALERTA DE CHUVA")
    else:
        st.success("✅ CLIMA ESTÁVEL")

# 2. Processamento da Imagem
if uploaded_file:
    # Salvar temporariamente
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_path = "temp_image.png"
else:
    # Usar imagem do dataset (garantir que existe)
    img_name = image_options[selected_area]
    # Se não tiver outras imagens, usa sempre a 000
    if not (Path("data/raw/deepglobe/train/images") / img_name).exists():
        img_name = "sat_000.png"
        
    img_path = str(Path("data/raw/deepglobe/train/images") / img_name)

# Analisar Risco
risk_map, original_img, risk_type, _, _ = engine.analyze_risk(img_path, day_index)

# Calcular Estatísticas
total_pixels = risk_map.size
risk_pixels = np.sum(risk_map > 0)
risk_pct = (risk_pixels / total_pixels) * 100

with col4:
    st.metric("Área em Risco", f"{risk_pct:.1f}%", help="Porcentagem da área urbana afetada")

# 3. Visualização dos Mapas
st.markdown("---")
st.subheader("🗺️ Mapas de Monitoramento")

col_left, col_right = st.columns(2)

with col_left:
    st.image(original_img, caption="Imagem de Satélite (Sentinel-2)", use_container_width=True)

with col_right:
    # Gerar Overlay para Streamlit
    original_resized = np.array(Image.fromarray(original_img).resize((256, 256)))
    overlay = original_resized.copy()
    
    if np.any(risk_map == 1): overlay[risk_map == 1] = [255, 0, 0]  # Vermelho
    elif np.any(risk_map == 2): overlay[risk_map == 2] = [0, 0, 255]  # Azul
    
    blended = (0.6 * original_resized + 0.4 * overlay).astype(np.uint8)
    
    st.image(blended, caption=f"Mapa de Risco: {risk_type}", use_container_width=True)

# 4. Gráfico de Previsão 7 Dias
st.markdown("---")
st.subheader("📅 Previsão para os Próximos 7 Dias")

chart_data = forecast.set_index('date')[['temp_pred', 'precip_pred']]
chart_data.columns = ['Temperatura (°C)', 'Chuva (mm)']

st.line_chart(chart_data)
