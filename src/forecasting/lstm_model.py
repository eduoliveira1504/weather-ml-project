"""
UrbanShield AI - Arquitetura LSTM para Previsão Climática

LSTM (Long Short-Term Memory) é uma rede neural recorrente que:
- Aprende padrões temporais de longo prazo
- Lembra informações relevantes e esquece irrelevantes
- Ideal para séries temporais climáticas
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_lstm_model(input_shape, output_steps=7, output_features=2):
    """
    Cria modelo LSTM para previsão multi-step
    
    Arquitetura:
        Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense → Output
    
    Args:
        input_shape: (seq_length, n_features) ex: (30, 12)
        output_steps: Quantos dias prever (7)
        output_features: Quantas variáveis prever (2: temp + chuva)
    
    Returns:
        keras.Model: Modelo LSTM compilado
    """
    
    model = keras.Sequential([
        # ========== CAMADA 1: LSTM ==========
        # 128 unidades LSTM (células de memória)
        # return_sequences=True → Passa sequência completa para próxima camada
        layers.LSTM(
            128, 
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        ),
        
        # Dropout para evitar overfitting (desliga 20% dos neurônios)
        layers.Dropout(0.2, name='dropout_1'),
        
        # ========== CAMADA 2: LSTM ==========
        # 64 unidades LSTM (menos que camada 1)
        # return_sequences=False → Retorna apenas último timestep
        layers.LSTM(
            64,
            return_sequences=False,
            name='lstm_2'
        ),
        
        # Dropout
        layers.Dropout(0.2, name='dropout_2'),
        
        # ========== CAMADA 3: DENSE ==========
        # Camada densa para combinar features aprendidas
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout_3'),
        
        # ========== CAMADA DE SAÍDA ==========
        # Produz output final: (7 dias, 2 variáveis) = 14 valores
        layers.Dense(output_steps * output_features, name='output'),
        
        # Reshape para (7, 2)
        layers.Reshape((output_steps, output_features), name='reshape')
    ])
    
    # ========== COMPILAR MODELO ==========
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model


def test_model():
    """
    Testa arquitetura com dados dummy
    """
    print("=" * 70)
    print("🧠 TESTE DA ARQUITETURA LSTM")
    print("=" * 70)
    
    # Criar modelo
    input_shape = (30, 12)  # 30 dias, 12 features
    model = create_lstm_model(input_shape)
    
    # Sumário
    model.summary()
    
    # Teste com dados dummy
    X_dummy = np.random.rand(1, 30, 12)  # 1 amostra
    prediction = model.predict(X_dummy, verbose=0)
    
    print(f"\n✅ Teste de predição:")
    print(f"   Input shape: {X_dummy.shape}")  # (1, 30, 12)
    print(f"   Output shape: {prediction.shape}")  # (1, 7, 2)
    print(f"   Output: {prediction.shape[1]} dias, {prediction.shape[2]} variáveis")
    
    print("\n" + "=" * 70)
    print("✅ ARQUITETURA VALIDADA!")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
