import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configurar cliente com cache e retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Coordenadas de Curitiba
LATITUDE = -25.4284
LONGITUDE = -49.2733

def test_open_meteo_connection():
    """Testa conexão com Open-Meteo e baixa temperatura dos últimos 7 dias"""
    
    print("🌤️  Conectando à API Open-Meteo...")
    print(f"📍 Localização: Curitiba ({LATITUDE}, {LONGITUDE})")
    print("="*60)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m",
        "past_days": 7,
        "forecast_days": 0,
        "timezone": "America/Sao_Paulo"
    }
    
    try:
        # Fazer requisição
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Informações da localização
        print(f"✅ Conexão bem-sucedida!")
        print(f"Coordenadas: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevação: {response.Elevation()} m")
        print(f"Timezone: {response.Timezone()}")
        print("="*60)
        
        # Processar dados horários
        hourly = response.Hourly()
        hourly_temperature = hourly.Variables(0).ValuesAsNumpy()
        
        # Criar DataFrame
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature": hourly_temperature
        }
        
        df = pd.DataFrame(data=hourly_data)
        
        # Estatísticas básicas
        print(f"📊 Dados coletados: {len(df)} registros horários")
        print(f"🌡️  Temperatura média: {df['temperature'].mean():.1f}°C")
        print(f"🔥 Temperatura máxima: {df['temperature'].max():.1f}°C")
        print(f"❄️  Temperatura mínima: {df['temperature'].min():.1f}°C")
        print("="*60)
        
        # Criar gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['temperature'], linewidth=2, color='#FF6B35')
        plt.title('Temperatura em Curitiba - Últimos 7 Dias', fontsize=16, weight='bold')
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Temperatura (°C)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Salvar gráfico
        output_path = 'outputs/temperature_curitiba_7days.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📈 Gráfico salvo em: {output_path}")
        
        # Salvar dados brutos
        csv_path = 'data/raw/temperature_7days.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Dados salvos em: {csv_path}")
        
        print("\n✅ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        raise

if __name__ == "__main__":
    # Criar diretórios necessários
    import os
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    
    test_open_meteo_connection()
