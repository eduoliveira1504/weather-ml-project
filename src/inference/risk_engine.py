"""
UrbanShield AI - Motor de Inferência de Risco
"""
import sys
import os
from pathlib import Path

# =========================================================
# CORREÇÃO DE PATH (Adiciona pasta src ao caminho do Python)
# =========================================================
# Obtém diretório atual do script
current_dir = Path(__file__).resolve().parent
# Adiciona src ao path (subindo um nível: inference -> src)
src_dir = current_dir.parent
sys.path.append(str(src_dir))

# Adiciona especificamente a pasta vision e forecasting
sys.path.append(str(src_dir / 'vision'))
sys.path.append(str(src_dir / 'forecasting'))

# AGORA PODEMOS IMPORTAR
import numpy as np
import tensorflow as tf
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Imports corrigidos (sem prefixos, pois adicionamos as pastas ao path)
from unet_model import UNet
from lstm_model import create_lstm_model
from predict import preprocess_image, predict as predict_segmentation
from preprocess import load_data, feature_engineering

# Configurações (Caminhos relativos à raiz do projeto)
PROJECT_ROOT = src_dir.parent
UNET_PATH = str(PROJECT_ROOT / "models/unet/best_model.pth.tar")
LSTM_PATH = str(PROJECT_ROOT / "models/lstm/best_lstm_model.keras")
DATA_DIR = PROJECT_ROOT / "data/processed"
DATA_RAW = PROJECT_ROOT / "data/raw"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RiskEngine:
    def __init__(self):
        self.unet = None
        self.lstm = None
        self.scalers = {}
        
        # Limiares de Risco
        self.TEMP_THRESHOLD = 30.0  # °C
        self.RAIN_THRESHOLD = 20.0  # mm
        
        self._load_models()
    
    def _load_models(self):
        """Carrega modelos treinados"""
        print("⚙️  Carregando modelos...")
        
        # 1. Carregar U-Net (PyTorch)
        # Importante: models/unet/best_model.pth.tar deve existir
        if not Path(UNET_PATH).exists():
            raise FileNotFoundError(f"❌ Modelo U-Net não encontrado: {UNET_PATH}")
            
        self.unet = UNet(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = torch.load(UNET_PATH, map_location=DEVICE)
        self.unet.load_state_dict(checkpoint["state_dict"])
        self.unet.eval()
        print("✅ U-Net carregada")
        
        # 2. Carregar LSTM (Keras)
        if not Path(LSTM_PATH).exists():
             raise FileNotFoundError(f"❌ Modelo LSTM não encontrado: {LSTM_PATH}")
             
        self.lstm = tf.keras.models.load_model(LSTM_PATH)
        print("✅ LSTM carregada")
        
        # 3. Carregar Scalers
        import pickle
        with open(DATA_DIR / 'scaler_features.pkl', 'rb') as f:
            self.scalers['features'] = pickle.load(f)
        
        with open(DATA_DIR / 'scaler_targets.pkl', 'rb') as f:
            self.scalers['targets'] = pickle.load(f)
            
        print("✅ Scalers carregados")

    def get_weather_forecast(self):
        """
        Gera previsão para os próximos 7 dias
        """
        # Carregar dados recentes
        csv_path = DATA_RAW / "weather/curitiba_historical_10years.csv"
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Feature Engineering
        df = feature_engineering(df)
        df = df.dropna()
        
        # Pegar últimos 30 dias
        last_30_days = df.tail(30)
        
        # Selecionar features (na ordem correta do treino)
        feature_cols = [
            'temp_max', 'temp_min', 'temp_mean', 'precipitation', 'rain',
            'windspeed_max', 'humidity_mean', 'pressure_mean',
            'month_sin', 'month_cos', 'temp_mean_7d', 'precip_sum_7d'
        ]
        
        input_data = last_30_days[feature_cols].values
        
        # Normalizar
        input_scaled = self.scalers['features'].transform(input_data)
        input_scaled = input_scaled.reshape(1, 30, len(feature_cols))
        
        # Prever
        prediction_scaled = self.lstm.predict(input_scaled, verbose=0)
        
        # Desnormalizar
        prediction_reshaped = prediction_scaled.reshape(7, 2)
        forecast_real = self.scalers['targets'].inverse_transform(prediction_reshaped)
        
        # Criar DataFrame com previsão
        last_date = df['date'].iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(7)]
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'temp_pred': forecast_real[:, 0],
            'precip_pred': forecast_real[:, 1]
        })
        
        return forecast_df

    def analyze_risk(self, image_path, day_index=0):
        """Analisa risco (mantido igual)"""
        # 1. Segmentação
        img_tensor, original_img = preprocess_image(image_path)
        mask = predict_segmentation(self.unet, img_tensor, DEVICE)
        
        # 2. Previsão Climática
        forecast = self.get_weather_forecast()
        day_forecast = forecast.iloc[day_index]
        
        temp = day_forecast['temp_pred']
        rain = day_forecast['precip_pred']
        date = day_forecast['date'].strftime('%d/%m/%Y')
        
        print(f"\n📅 Previsão para {date} (D+{day_index+1}):")
        print(f"   🌡️ Temperatura: {temp:.1f}°C")
        print(f"   💧 Chuva: {rain:.1f} mm")
        
        # 3. Cruzamento
        risk_map = np.zeros_like(mask)
        risk_type = "Nenhum"
        
        is_concrete = (mask == 1)
        
        if temp > self.TEMP_THRESHOLD:
            risk_map[is_concrete] = 1
            risk_type = "🔥 Ilha de Calor"
            
        elif rain > self.RAIN_THRESHOLD:
            risk_map[is_concrete] = 2
            risk_type = "💧 Alagamento"
            
        return risk_map, original_img, risk_type, temp, rain

    def visualize_risk(self, original_img, risk_map, risk_type, temp, rain, save_path=None):
        """Visualiza risco (mantido igual)"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Satélite (Original)")
        axes[0].axis('off')
        
        original_resized = np.array(Image.fromarray(original_img).resize((256, 256)))
        overlay = original_resized.copy()
        
        if np.any(risk_map == 1): overlay[risk_map == 1] = [255, 0, 0]  # Vermelho
        elif np.any(risk_map == 2): overlay[risk_map == 2] = [0, 0, 255]  # Azul
            
        blended = (0.6 * original_resized + 0.4 * overlay).astype(np.uint8)
        
        axes[1].imshow(blended)
        title = f"Risco: {risk_type}\nTemp: {temp:.1f}°C | Chuva: {rain:.1f}mm"
        axes[1].set_title(title, fontsize=14, color='red' if 'Calor' in risk_type else 'blue')
        axes[1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"💾 Mapa salvo em: {save_path}")
        plt.show()

if __name__ == "__main__":
    engine = RiskEngine()
    img_path = str(DATA_RAW / "deepglobe/train/images/sat_000.png")
    
    # 1. Teste Normal (D+1)
    print("\n--- TESTE 1: PREVISÃO REAL ---")
    risk_map, img, r_type, temp, rain = engine.analyze_risk(img_path, day_index=0)
    
    # 2. Teste Simulado (FORÇAR RISCO)
    print("\n--- TESTE 2: SIMULAÇÃO DE RISCO ---")
    
    # Forçar temperatura alta (Ilha de Calor)
    print("🔥 Simulando Onda de Calor (35°C)...")
    engine.TEMP_THRESHOLD = 30.0
    fake_temp = 35.0
    fake_rain = 0.0
    
    # Recalcular risco manualmente para teste
    mask = predict_segmentation(engine.unet, preprocess_image(img_path)[0], DEVICE)
    risk_map_sim = np.zeros_like(mask)
    
    if fake_temp > engine.TEMP_THRESHOLD:
        risk_map_sim[mask == 1] = 1 # Risco Calor
        
    output_path = str(PROJECT_ROOT / "outputs/risk_simulation.png")
    engine.visualize_risk(img, risk_map_sim, "🔥 Ilha de Calor (Simulado)", fake_temp, fake_rain, output_path)

