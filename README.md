# UrbanShield AI

Sistema de monitoramento de riscos climáticos urbanos para Curitiba-PR usando Deep Learning.

## Sobre

Projeto de TCC que combina:
- **U-Net** para segmentação de imagens de satélite (detecta áreas urbanas vs vegetação)
- **LSTM** para previsão climática (temperatura e chuva nos próximos 7 dias)
- **Motor de inferência** que cruza os dois para identificar riscos de ilhas de calor e alagamentos

## Tecnologias

- Python 3.10+
- PyTorch (U-Net)
- TensorFlow/Keras (LSTM)
- Streamlit (Dashboard)
- Open-Meteo API (dados climáticos gratuitos)