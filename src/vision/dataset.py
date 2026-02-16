"""
UrbanShield AI - Dataset Loader para U-Net (Versão Simplificada)
Usa apenas torchvision (sem albumentations)
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class SatelliteDataset(Dataset):
    """
    Dataset para imagens de satélite e máscaras de segmentação
    """
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = sorted(os.listdir(image_dir))
        
        print(f"📂 Dataset inicializado:")
        print(f"   Imagens: {len(self.images)}")
        print(f"   Localização: {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Caminhos dos arquivos
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, 
            self.images[index].replace("sat_", "mask_")
        )
        
        # Carregar imagem (RGB)
        image = Image.open(img_path).convert("RGB")
        
        # Carregar máscara (Grayscale)
        mask = Image.open(mask_path).convert("L")
        
        # Aplicar transformações
        if self.transform is not None:
            image = self.transform(image)
            # Converter máscara para tensor
            mask = transforms.ToTensor()(mask)
        
        # Converter máscara para binário (0 ou 1)
        mask = (mask > 0.5).float()
        
        return image, mask


def get_transforms(train=True):
    """
    Transformações usando torchvision
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])


def get_loaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    batch_size=16,
    num_workers=2,
    pin_memory=True
):
    """
    Cria DataLoaders para treino e validação
    """
    
    train_dataset = SatelliteDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=get_transforms(train=True)
    )
    
    val_dataset = SatelliteDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n📊 DataLoaders criados:")
    print(f"   Treino: {len(train_dataset)} imagens, {len(train_loader)} batches")
    print(f"   Validação: {len(val_dataset)} imagens, {len(val_loader)} batches")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader


def test_dataset():
    """
    Testa o dataset loader
    """
    print("=" * 60)
    print("🧪 TESTE DO DATASET LOADER")
    print("=" * 60)
    
    img_dir = "data/raw/deepglobe/train/images"
    mask_dir = "data/raw/deepglobe/train/masks"
    
    if not os.path.exists(img_dir):
        print(f"❌ Diretório não encontrado: {img_dir}")
        print("   Execute primeiro: python src/data/download_deepglobe.py")
        return
    
    dataset = SatelliteDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        transform=get_transforms(train=True)
    )
    
    image, mask = dataset[0]
    
    print(f"\n✅ Amostra carregada:")
    print(f"   Imagem shape: {image.shape}")
    print(f"   Máscara shape: {mask.shape}")
    print(f"   Imagem min/max: {image.min():.3f} / {image.max():.3f}")
    print(f"   Máscara valores únicos: {torch.unique(mask)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_images, batch_masks = next(iter(loader))
    
    print(f"\n✅ Batch carregado:")
    print(f"   Batch de imagens: {batch_images.shape}")
    print(f"   Batch de máscaras: {batch_masks.shape}")
    
    print("\n" + "=" * 60)
    print("✅ DATASET LOADER FUNCIONANDO!")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()
