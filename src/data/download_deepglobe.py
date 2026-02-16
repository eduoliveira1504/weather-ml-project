"""
UrbanShield AI - Download do DeepGlobe Land Cover Dataset
Dataset: 803 imagens de satélite (2448x2448) com máscaras de segmentação
Classes: Urban, Agriculture, Rangeland, Forest, Water, Barren, Unknown
Fonte: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, output_path):
    """
    Download de arquivo com barra de progresso
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_deepglobe_sample():
    """
    Baixa um subset pequeno do DeepGlobe para prototipagem rápida
    """
    print("=" * 60)
    print("🛰️  URBANSHIELD AI - DOWNLOAD DEEPGLOBE DATASET")
    print("=" * 60)
    
    # Criar diretórios
    output_dir = Path("data/raw/deepglobe")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 Para este TCC, vamos usar uma abordagem híbrida:")
    print("   1. Treinar U-Net com DeepGlobe (genérico)")
    print("   2. Fine-tuning com imagens de Curitiba (específico)")
    print("\n⚠️  O dataset completo tem ~2GB. Vou baixar um SAMPLE primeiro.")
    
    # URLs de exemplo (você pode expandir depois)
    # Vou criar um script que gera dados sintéticos para começar
    print("\n🔧 Gerando dataset sintético para prototipagem...")
    
    create_synthetic_dataset(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARADO!")
    print("=" * 60)
    print(f"📂 Localização: {output_dir}")
    print("\n📌 Próximo passo: Treinar U-Net")
    print("   python src/vision/train_unet.py")

def create_synthetic_dataset(output_dir):
    """
    Cria um dataset sintético pequeno para testar o pipeline
    Depois você substitui por dados reais
    """
    import numpy as np
    from PIL import Image
    
    train_dir = output_dir / "train"
    train_images = train_dir / "images"
    train_masks = train_dir / "masks"
    
    for dir in [train_images, train_masks]:
        dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎨 Gerando 50 imagens sintéticas para teste...")
    
    np.random.seed(42)
    
    for i in tqdm(range(50), desc="Criando imagens"):
        # Imagem RGB simulando satélite (256x256)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Máscara binária (0=vegetação, 1=urbano)
        # Simula manchas urbanas
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Criar algumas "manchas urbanas" aleatórias
        for _ in range(5):
            x, y = np.random.randint(50, 206, 2)
            size = np.random.randint(20, 50)
            mask[y:y+size, x:x+size] = 1
        
        # Salvar
        Image.fromarray(img).save(train_images / f"sat_{i:03d}.png")
        Image.fromarray(mask * 255).save(train_masks / f"mask_{i:03d}.png")
    
    print(f"✅ Criadas 50 imagens em: {train_images}")
    print(f"✅ Criadas 50 máscaras em: {train_masks}")
    print("\n💡 NOTA: Este é um dataset sintético para TESTE.")
    print("   Para produção, você pode:")
    print("   - Baixar DeepGlobe completo do Kaggle")
    print("   - Usar imagens reais do Sentinel-2")

def main():
    download_deepglobe_sample()

if __name__ == "__main__":
    main()
