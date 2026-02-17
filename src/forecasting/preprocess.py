"""
UrbanShield AI - Pré-processamento de Dados Climáticos para LSTM

Prepara séries temporais para treinar o modelo LSTM:
- Normalização (0-1)
- Criação de sequências (janelas deslizantes)
- Separação treino/validação
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pickle

# Configurações
DATA_PATH = "data/raw/weather/curitiba_historical_10years.csv"
OUTPUT_DIR = Path("data/processed")
SEQUENCE_LENGTH = 30  # Usar 30 dias para prever próximos 7
FORECAST_HORIZON = 7  # Prever 7 dias à frente


def load_data(file_path):
    """
    Carrega dados climáticos históricos
    
    Returns:
        pd.DataFrame: Dados carregados
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"📂 Dados carregados: {len(df)} dias")
    print(f"📅 Período: {df['date'].min()} a {df['date'].max()}")
    
    return df


def feature_engineering(df):
    """
    Criar features adicionais (sazonalidade, lags, etc)
    
    Args:
        df: DataFrame com dados brutos
    
    Returns:
        pd.DataFrame: DataFrame com features adicionais
    """
    df = df.copy()
    
    # 1. Features temporais (sazonalidade)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    
    # 2. Features cíclicas (seno/cosseno para sazonalidade)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. Rolling statistics (médias móveis)
    df['temp_mean_7d'] = df['temp_mean'].rolling(window=7, min_periods=1).mean()
    df['precip_sum_7d'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
    
    print(f"✅ Features criadas: {len(df.columns)} colunas")
    
    return df


def select_features(df):
    """
    Seleciona features relevantes para previsão
    
    Returns:
        pd.DataFrame: Apenas features selecionadas
    """
    # Features para LSTM
    feature_cols = [
        'temp_max',
        'temp_min',
        'temp_mean',
        'precipitation',
        'rain',
        'windspeed_max',
        'humidity_mean',
        'pressure_mean',
        'month_sin',
        'month_cos',
        'temp_mean_7d',
        'precip_sum_7d'
    ]
    
    # Targets (o que queremos prever)
    target_cols = [
        'temp_mean',
        'precipitation'
    ]
    
    print(f"📊 Features selecionadas: {len(feature_cols)}")
    print(f"🎯 Targets: {target_cols}")
    
    return df[feature_cols].values, df[target_cols].values, feature_cols, target_cols


def normalize_data(features, targets):
    """
    Normaliza dados para [0, 1] usando MinMaxScaler
    
    Args:
        features: Array com features
        targets: Array com targets
    
    Returns:
        tuple: (features_scaled, targets_scaled, scaler_features, scaler_targets)
    """
    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()
    
    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets)
    
    print(f"✅ Dados normalizados para [0, 1]")
    
    return features_scaled, targets_scaled, scaler_features, scaler_targets


def create_sequences(features, targets, seq_length, forecast_horizon):
    """
    Cria sequências de janelas deslizantes para LSTM
    
    Exemplo:
        Input: [dia 1, dia 2, ..., dia 30]
        Output: [dia 31, dia 32, ..., dia 37]
    
    Args:
        features: Features normalizadas
        targets: Targets normalizados
        seq_length: Tamanho da janela (30 dias)
        forecast_horizon: Quantos dias prever (7 dias)
    
    Returns:
        tuple: (X, y) onde X tem shape (samples, 30, n_features)
                           y tem shape (samples, 7, 2)
    """
    X, y = [], []
    
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        # Pegar 30 dias de features
        X.append(features[i:i+seq_length])
        
        # Pegar 7 dias futuros de targets
        y.append(targets[i+seq_length:i+seq_length+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Sequências criadas:")
    print(f"   X shape: {X.shape} (samples, seq_length, features)")
    print(f"   y shape: {y.shape} (samples, forecast_horizon, targets)")
    
    return X, y


def split_data(X, y, train_ratio=0.8):
    """
    Divide dados em treino e validação
    
    IMPORTANTE: Para séries temporais, não embaralhamos!
    Treino = primeiros 80%, Validação = últimos 20%
    
    Args:
        X: Sequências de entrada
        y: Sequências de saída
        train_ratio: Proporção de treino (0.8 = 80%)
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    print(f"✅ Dados divididos:")
    print(f"   Treino: {len(X_train)} sequências ({train_ratio*100:.0f}%)")
    print(f"   Validação: {len(X_val)} sequências ({(1-train_ratio)*100:.0f}%)")
    
    return X_train, X_val, y_train, y_val


def save_preprocessed_data(X_train, X_val, y_train, y_val, 
                           scaler_features, scaler_targets,
                           feature_cols, target_cols):
    """
    Salva dados pré-processados e scalers
    
    Args:
        X_train, X_val, y_train, y_val: Arrays numpy
        scaler_features, scaler_targets: Scalers do sklearn
        feature_cols, target_cols: Nomes das colunas
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Salvar arrays
    np.save(OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'X_val.npy', X_val)
    np.save(OUTPUT_DIR / 'y_train.npy', y_train)
    np.save(OUTPUT_DIR / 'y_val.npy', y_val)
    
    # Salvar scalers
    with open(OUTPUT_DIR / 'scaler_features.pkl', 'wb') as f:
        pickle.dump(scaler_features, f)
    
    with open(OUTPUT_DIR / 'scaler_targets.pkl', 'wb') as f:
        pickle.dump(scaler_targets, f)
    
    # Salvar nomes das colunas
    with open(OUTPUT_DIR / 'feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    with open(OUTPUT_DIR / 'target_cols.pkl', 'wb') as f:
        pickle.dump(target_cols, f)
    
    print(f"💾 Dados salvos em: {OUTPUT_DIR}")


def main():
    print("=" * 70)
    print("🌍 URBANSHIELD AI - PRÉ-PROCESSAMENTO DE DADOS CLIMÁTICOS")
    print("=" * 70)
    
    # 1. Carregar dados
    df = load_data(DATA_PATH)
    
    # 2. Feature engineering
    df = feature_engineering(df)
    
    # 3. Remover NaNs (das rolling features)
    df = df.dropna()
    print(f"✅ Dados após remoção de NaNs: {len(df)} dias")
    
    # 4. Selecionar features
    features, targets, feature_cols, target_cols = select_features(df)
    
    # 5. Normalizar
    features_scaled, targets_scaled, scaler_features, scaler_targets = normalize_data(
        features, targets
    )
    
    # 6. Criar sequências
    X, y = create_sequences(
        features_scaled, 
        targets_scaled, 
        SEQUENCE_LENGTH, 
        FORECAST_HORIZON
    )
    
    # 7. Dividir em treino/validação
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    # 8. Salvar
    save_preprocessed_data(
        X_train, X_val, y_train, y_val,
        scaler_features, scaler_targets,
        feature_cols, target_cols
    )
    
    print("\n" + "=" * 70)
    print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO!")
    print("=" * 70)
    print(f"📊 Resumo:")
    print(f"   Sequências de treino: {len(X_train)}")
    print(f"   Sequências de validação: {len(X_val)}")
    print(f"   Janela temporal: {SEQUENCE_LENGTH} dias")
    print(f"   Horizonte de previsão: {FORECAST_HORIZON} dias")
    print(f"\n📌 Próximo passo: python src/forecasting/train_lstm.py")


if __name__ == "__main__":
    main()
