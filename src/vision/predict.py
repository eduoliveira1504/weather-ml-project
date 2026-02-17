"""
UrbanShield AI - Inferência com U-Net Treinada
Faz segmentação de uma imagem de satélite e visualiza o resultado
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms

from unet_model import UNet

# Configurações
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/unet/best_model.pth.tar"
IMAGE_SIZE = 256


def load_model(model_path, device):
    """
    Carrega modelo treinado
    
    Args:
        model_path (str): Caminho do checkpoint
        device (str): 'cuda' ou 'cpu'
    
    Returns:
        model: U-Net carregada
    """
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print(f"✅ Modelo carregado de: {model_path}")
    if "best_dice" in checkpoint:
        print(f"   Dice Score do modelo: {checkpoint['best_dice']:.4f}")
    
    return model


def preprocess_image(image_path):
    """
    Pré-processa imagem para inferência
    
    Args:
        image_path (str): Caminho da imagem
    
    Returns:
        tuple: (tensor, imagem_original)
    """
    # Carregar imagem
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    
    # Redimensionar se necessário
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Transformar para tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Adicionar batch dimension
    
    return image_tensor, original


def predict(model, image_tensor, device):
    """
    Faz predição da máscara de segmentação
    
    Args:
        model: U-Net carregada
        image_tensor: Tensor da imagem
        device: 'cuda' ou 'cpu'
    
    Returns:
        numpy.array: Máscara predita (256x256)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)  # Converter para probabilidades
        prediction = (prediction > 0.5).float()  # Threshold 0.5
    
    # Converter para numpy
    mask = prediction.squeeze().cpu().numpy()
    
    return mask


def visualize_result(original_image, predicted_mask, save_path=None):
    """
    Visualiza imagem original, máscara predita e overlay
    
    Args:
        original_image: Imagem RGB original
        predicted_mask: Máscara predita (0 ou 1)
        save_path: Onde salvar (opcional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Imagem original
    axes[0].imshow(original_image)
    axes[0].set_title("Imagem Original (Satélite)", fontsize=14)
    axes[0].axis('off')
    
    # 2. Máscara predita
    axes[1].imshow(predicted_mask, cmap='gray')
    axes[1].set_title("Segmentação\n(Branco=Urbano, Preto=Vegetação)", fontsize=14)
    axes[1].axis('off')
    
    # 3. Overlay (sobrepor máscara na imagem)
    # Redimensionar imagem original para 256x256
    original_resized = np.array(Image.fromarray(original_image).resize((256, 256)))
    overlay = original_resized.copy()
    
    # Pintar áreas urbanas de vermelho
    overlay[predicted_mask == 1] = [255, 0, 0]  # Vermelho para urbano
    
    # Blend (misturar original com overlay)
    blended = (0.6 * original_resized + 0.4 * overlay).astype(np.uint8)
    
    axes[2].imshow(blended)
    axes[2].set_title("Overlay\n(Vermelho=Áreas Urbanas)", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualização salva em: {save_path}")
    
    plt.show()


def calculate_urban_percentage(mask):
    """
    Calcula % de área urbana na imagem
    
    Args:
        mask: Máscara binária
    
    Returns:
        float: Porcentagem urbana
    """
    urban_pixels = (mask == 1).sum()
    total_pixels = mask.size
    percentage = (urban_pixels / total_pixels) * 100
    
    return percentage


def main():
    print("=" * 70)
    print("🌍 URBANSHIELD AI - INFERÊNCIA U-NET")
    print("=" * 70)
    
    # 1. Carregar modelo
    model = load_model(MODEL_PATH, DEVICE)
    
    # 2. Selecionar imagem para testar
    # Vamos usar uma imagem do dataset de treino como exemplo
    image_path = "data/raw/deepglobe/train/images/sat_000.png"
    
    if not Path(image_path).exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        print("   Coloque uma imagem de teste em: data/raw/deepglobe/train/images/")
        return
    
    print(f"\n📷 Processando imagem: {image_path}")
    
    # 3. Pré-processar
    image_tensor, original_image = preprocess_image(image_path)
    
    # 4. Predição
    print("🔮 Fazendo predição...")
    predicted_mask = predict(model, image_tensor, DEVICE)
    
    # 5. Calcular estatísticas
    urban_pct = calculate_urban_percentage(predicted_mask)
    vegetation_pct = 100 - urban_pct
    
    print(f"\n📊 ANÁLISE DA IMAGEM:")
    print(f"   🏙️  Área Urbana (impermeável): {urban_pct:.2f}%")
    print(f"   🌳 Área Vegetação (permeável): {vegetation_pct:.2f}%")
    
    # 6. Visualizar
    output_path = Path("outputs/unet_prediction.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_result(original_image, predicted_mask, save_path=output_path)
    
    print("\n" + "=" * 70)
    print("✅ INFERÊNCIA CONCLUÍDA!")
    print("=" * 70)


if __name__ == "__main__":
    main()
