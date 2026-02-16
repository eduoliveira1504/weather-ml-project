"""
UrbanShield AI - Download de dados climáticos históricos
API: Open-Meteo (100% gratuita, sem chave necessária)
Dados: 2014-2024 (10 anos) para Curitiba-PR
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# Coordenadas de Curitiba-PR
LATITUDE = -25.4284
LONGITUDE = -49.2733

def fetch_historical_weather(start_date, end_date, output_path):
    """
    Baixa dados históricos via Open-Meteo Archive API
    
    Args:
        start_date (str): Data inicial (formato: 'YYYY-MM-DD')
        end_date (str): Data final (formato: 'YYYY-MM-DD')
        output_path (str): Caminho para salvar o CSV
    
    Returns:
        pd.DataFrame: Dados meteorológicos diários
    """
    print(f"🌦️  Baixando dados: {start_date} a {end_date}")
    print(f"📍 Local: Curitiba-PR ({LATITUDE}, {LONGITUDE})")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parâmetros da requisição
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",      # Temperatura máxima (°C)
            "temperature_2m_min",      # Temperatura mínima (°C)
            "temperature_2m_mean",     # Temperatura média (°C)
            "precipitation_sum",       # Precipitação total (mm)
            "rain_sum",                # Chuva total (mm)
            "precipitation_hours",     # Horas de precipitação
            "windspeed_10m_max",       # Velocidade máx vento (km/h)
            "relative_humidity_2m_mean", # Umidade relativa (%)
            "pressure_msl_mean"        # Pressão atmosférica (hPa)
        ],
        "timezone": "America/Sao_Paulo"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Converter JSON para DataFrame
        daily = data['daily']
        df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            'temp_max': daily['temperature_2m_max'],
            'temp_min': daily['temperature_2m_min'],
            'temp_mean': daily['temperature_2m_mean'],
            'precipitation': daily['precipitation_sum'],
            'rain': daily['rain_sum'],
            'precip_hours': daily['precipitation_hours'],
            'windspeed_max': daily['windspeed_10m_max'],
            'humidity_mean': daily['relative_humidity_2m_mean'],
            'pressure_mean': daily['pressure_msl_mean']
        })
        
        # Preencher valores nulos com 0 (comum em precipitação)
        df['precipitation'] = df['precipitation'].fillna(0)
        df['rain'] = df['rain'].fillna(0)
        
        # Salvar CSV
        df.to_csv(output_path, index=False)
        print(f"✅ Dados salvos: {output_path}")
        print(f"📊 Total de dias: {len(df)}")
        print(f"📅 Período: {df['date'].min().date()} a {df['date'].max().date()}")
        
        # Estatísticas descritivas
        print("\n📈 ESTATÍSTICAS CLIMATOLÓGICAS:")
        print(f"  🌡️  Temp média anual: {df['temp_mean'].mean():.1f}°C")
        print(f"  🔥 Temp máxima (recorde): {df['temp_max'].max():.1f}°C em {df.loc[df['temp_max'].idxmax(), 'date'].date()}")
        print(f"  ❄️  Temp mínima (recorde): {df['temp_min'].min():.1f}°C em {df.loc[df['temp_min'].idxmin(), 'date'].date()}")
        print(f"  💧 Precipitação anual média: {df['precipitation'].sum() / 10:.0f} mm/ano")
        print(f"  ⚠️  Dias com chuva >50mm: {(df['rain'] > 50).sum()} dias (risco de alagamento)")
        print(f"  ⚠️  Dias com temp >35°C: {(df['temp_max'] > 35).sum()} dias (ilha de calor)")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro na requisição HTTP: {e}")
        return None
    except KeyError as e:
        print(f"❌ Erro ao processar dados: campo ausente {e}")
        return None
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return None


def main():
    """Função principal de execução"""
    
    # Configuração: 10 anos de dados (2014-2024)
    START_DATE = "2014-01-01"
    END_DATE = "2024-12-31"
    
    # Criar diretório de saída
    output_dir = Path("data/raw/weather")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "curitiba_historical_10years.csv"
    
    # Executar download
    print("=" * 60)
    print("🌍 URBANSHIELD AI - COLETA DE DADOS CLIMÁTICOS")
    print("=" * 60)
    
    df = fetch_historical_weather(START_DATE, END_DATE, output_file)
    
    if df is not None:
        # Verificar dados faltantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  ATENÇÃO: Dados faltantes detectados:")
            print(missing[missing > 0])
        else:
            print("\n✅ Dataset completo! Nenhum dado faltante.")
        
        print("\n" + "=" * 60)
        print("✅ COLETA CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print(f"\n📂 Arquivo salvo em: {output_file}")
        print("📌 Próximo passo: python src/data/download_sentinel.py")
    else:
        print("\n❌ Falha na coleta de dados. Verifique a conexão com a internet.")


if __name__ == "__main__":
    main()
