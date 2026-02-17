"""
UrbanShield AI - Treinamento da LSTM para Previsão Climática

Treina modelo LSTM para prever temperatura e precipitação 7 dias à frente
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from lstm_model import create_lstm_model

# Configurações
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/lstm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15  # Early stopping se não melhorar por 15 épocas


def load_preprocessed_data():
    """
    Carrega dados pré-processados
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler_targets)
    """
    X_train = np.load(DATA_DIR / 'X_train.npy')
    X_val = np.load(DATA_DIR / 'X_val.npy')
    y_train = np.load(DATA_DIR / 'y_train.npy')
    y_val = np.load(DATA_DIR / 'y_val.npy')
    
    with open(DATA_DIR / 'scaler_targets.pkl', 'rb') as f:
        scaler_targets = pickle.load(f)
    
    print(f"📂 Dados carregados:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_val: {y_val.shape}")
    
    return X_train, X_val, y_train, y_val, scaler_targets


def create_callbacks(model_path):
    """
    Cria callbacks para treinamento
    
    Callbacks:
    - ModelCheckpoint: Salva melhor modelo
    - EarlyStopping: Para se não melhorar
    - ReduceLROnPlateau: Reduz learning rate se estagnar
    
    Returns:
        list: Lista de callbacks
    """
    callbacks = [
        # Salvar melhor modelo
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduzir learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path):
    """
    Plota curvas de loss e MAE
    
    Args:
        history: Objeto History do Keras
        save_path: Onde salvar gráfico
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Loss (MSE)
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss (MSE)', fontsize=14)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. MAE
    axes[1].plot(history.history['mae'], label='Treino')
    axes[1].plot(history.history['val_mae'], label='Validação')
    axes[1].set_title('MAE (Mean Absolute Error)', fontsize=14)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Gráfico salvo em: {save_path}")


def evaluate_model(model, X_val, y_val, scaler_targets):
    """
    Avalia modelo no conjunto de validação
    
    Args:
        model: Modelo treinado
        X_val: Features de validação
        y_val: Targets de validação
        scaler_targets: Scaler para desnormalizar
    
    Returns:
        dict: Métricas de avaliação
    """
    # Fazer predições
    y_pred = model.predict(X_val)
    
    # Desnormalizar para valores reais
    # Reshape para 2D: (samples*7, 2)
    y_val_reshaped = y_val.reshape(-1, 2)
    y_pred_reshaped = y_pred.reshape(-1, 2)
    
    y_val_real = scaler_targets.inverse_transform(y_val_reshaped)
    y_pred_real = scaler_targets.inverse_transform(y_pred_reshaped)
    
    # Reshape de volta: (samples, 7, 2)
    y_val_real = y_val_real.reshape(-1, 7, 2)
    y_pred_real = y_pred_real.reshape(-1, 7, 2)
    
    # Calcular métricas separadas para temperatura e precipitação
    # Temperatura (índice 0)
    temp_mae = np.mean(np.abs(y_val_real[:, :, 0] - y_pred_real[:, :, 0]))
    temp_rmse = np.sqrt(np.mean((y_val_real[:, :, 0] - y_pred_real[:, :, 0])**2))
    
    # Precipitação (índice 1)
    precip_mae = np.mean(np.abs(y_val_real[:, :, 1] - y_pred_real[:, :, 1]))
    precip_rmse = np.sqrt(np.mean((y_val_real[:, :, 1] - y_pred_real[:, :, 1])**2))
    
    metrics = {
        'temp_mae': temp_mae,
        'temp_rmse': temp_rmse,
        'precip_mae': precip_mae,
        'precip_rmse': precip_rmse
    }
    
    return metrics


def main():
    print("=" * 70)
    print("🌍 URBANSHIELD AI - TREINAMENTO LSTM")
    print("=" * 70)
    
    # Verificar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detectada: {gpus[0].name}")
    else:
        print("⚠️  Nenhuma GPU detectada, usando CPU")
    
    # 1. Carregar dados
    X_train, X_val, y_train, y_val, scaler_targets = load_preprocessed_data()
    
    # 2. Criar modelo
    print(f"\n🧠 Criando modelo LSTM...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 12)
    model = create_lstm_model(input_shape)
    
    print(f"✅ Modelo criado:")
    print(f"   Parâmetros: {model.count_params():,}")
    
    # 3. Callbacks
    model_path = MODEL_DIR / 'best_lstm_model.keras'
    callbacks = create_callbacks(model_path)
    
    # 4. Treinar
    print("\n" + "=" * 70)
    print("🚀 INICIANDO TREINAMENTO")
    print("=" * 70)
    print(f"   Épocas: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Early stopping: {PATIENCE} épocas")
    print()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Plotar histórico
    plot_path = MODEL_DIR / 'training_history.png'
    plot_training_history(history, plot_path)
    
    # 6. Avaliar
    print("\n" + "=" * 70)
    print("📊 AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO")
    print("=" * 70)
    
    metrics = evaluate_model(model, X_val, y_val, scaler_targets)
    
    print(f"\n🌡️  TEMPERATURA:")
    print(f"   MAE: {metrics['temp_mae']:.2f}°C")
    print(f"   RMSE: {metrics['temp_rmse']:.2f}°C")
    
    print(f"\n💧 PRECIPITAÇÃO:")
    print(f"   MAE: {metrics['precip_mae']:.2f} mm")
    print(f"   RMSE: {metrics['precip_rmse']:.2f} mm")
    
    # 7. Salvar métricas
    metrics_path = MODEL_DIR / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"MÉTRICAS DO MODELO LSTM\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"TEMPERATURA:\n")
        f.write(f"  MAE: {metrics['temp_mae']:.2f}°C\n")
        f.write(f"  RMSE: {metrics['temp_rmse']:.2f}°C\n\n")
        f.write(f"PRECIPITAÇÃO:\n")
        f.write(f"  MAE: {metrics['precip_mae']:.2f} mm\n")
        f.write(f"  RMSE: {metrics['precip_rmse']:.2f} mm\n")
    
    print(f"\n💾 Métricas salvas em: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("=" * 70)
    print(f"📁 Modelo salvo em: {model_path}")
    print(f"📊 Gráficos em: {plot_path}")
    print("\n📌 Próximo passo: Motor de Inferência (cruzar U-Net + LSTM)")


if __name__ == "__main__":
    main()
