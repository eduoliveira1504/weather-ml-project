"""
UrbanShield AI - Treinamento da U-Net para Segmentação de Satélite

Este script implementa o treinamento completo da U-Net com:
- Função de perda combinada (Dice Loss + Binary Cross Entropy)
- Métricas de avaliação (Accuracy, IoU, Dice Score)
- Salvamento do melhor modelo
- Validação a cada época
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import os
from pathlib import Path

# Importar nossos módulos
from unet_model import UNet
from dataset import SatelliteDataset, get_transforms

# ==========================
# CONFIGURAÇÕES
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

# Diretórios
DATA_DIR = Path("data/raw/deepglobe/train")
MODEL_DIR = Path("models/unet")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# FUNÇÕES DE PERDA
# ==========================

class DiceLoss(nn.Module):
    """
    Dice Loss para segmentação
    
    Dice Score mede sobreposição entre predição e ground truth.
    Fórmula: 2 * |A ∩ B| / (|A| + |B|)
    
    Vantagens:
    - Funciona bem com classes desbalanceadas
    - Penaliza erros em regiões pequenas
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Aplicar sigmoid para converter logits em probabilidades
        predictions = torch.sigmoid(predictions)
        
        # Flatten (achatar) para calcular interseção
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calcular interseção e união
        intersection = (predictions * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # Retornar 1 - dice (loss)
        return 1 - dice_score


class DiceBCELoss(nn.Module):
    """
    Combinação de Dice Loss + Binary Cross Entropy
    
    BCE: Penaliza predições incorretas pixel por pixel
    Dice: Penaliza falta de sobreposição global
    
    Combinação funciona melhor que cada uma isolada!
    """
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        dice_loss = self.dice(predictions, targets)
        bce_loss = self.bce(predictions, targets)
        return dice_loss + bce_loss


# ==========================
# MÉTRICAS DE AVALIAÇÃO
# ==========================

def calculate_accuracy(predictions, targets):
    """Acurácia: % de pixels classificados corretamente"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def calculate_iou(predictions, targets, threshold=0.5):
    """
    IoU (Intersection over Union) ou Jaccard Index
    Mede sobreposição: |A ∩ B| / |A ∪ B|
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_dice_score(predictions, targets):
    """
    Dice Score: 2 * |A ∩ B| / (|A| + |B|)
    Semelhante ao IoU, mas pondera interseção 2x
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection + 1e-6) / (
        predictions.sum() + targets.sum() + 1e-6
    )
    return dice.item()


# ==========================
# TREINAMENTO E VALIDAÇÃO
# ==========================

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """
    Treina o modelo por uma época
    
    Args:
        loader: DataLoader com dados de treino
        model: Modelo U-Net
        optimizer: Otimizador (Adam)
        loss_fn: Função de perda
        device: 'cuda' ou 'cpu'
    
    Returns:
        dict: Métricas médias da época
    """
    model.train()
    
    loop = tqdm(loader, desc="Treinando")
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_iou = 0
    epoch_dice = 0
    
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calcular métricas
        acc = calculate_accuracy(predictions, masks)
        iou = calculate_iou(predictions, masks)
        dice = calculate_dice_score(predictions, masks)
        
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_iou += iou
        epoch_dice += dice
        
        # Atualizar barra de progresso
        loop.set_postfix(
            loss=loss.item(),
            acc=acc,
            iou=iou,
            dice=dice
        )
    
    # Médias
    num_batches = len(loader)
    return {
        'loss': epoch_loss / num_batches,
        'accuracy': epoch_acc / num_batches,
        'iou': epoch_iou / num_batches,
        'dice': epoch_dice / num_batches
    }


def validate(loader, model, loss_fn, device):
    """
    Valida o modelo no conjunto de validação
    
    Args:
        loader: DataLoader com dados de validação
        model: Modelo U-Net
        loss_fn: Função de perda
        device: 'cuda' ou 'cpu'
    
    Returns:
        dict: Métricas médias de validação
    """
    model.eval()
    
    val_loss = 0
    val_acc = 0
    val_iou = 0
    val_dice = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validando"):
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            
            val_loss += loss.item()
            val_acc += calculate_accuracy(predictions, masks)
            val_iou += calculate_iou(predictions, masks)
            val_dice += calculate_dice_score(predictions, masks)
    
    num_batches = len(loader)
    return {
        'loss': val_loss / num_batches,
        'accuracy': val_acc / num_batches,
        'iou': val_iou / num_batches,
        'dice': val_dice / num_batches
    }


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Salva checkpoint do modelo"""
    print(f"💾 Salvando checkpoint: {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """Carrega checkpoint do modelo"""
    print(f"📂 Carregando checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# ==========================
# FUNÇÃO PRINCIPAL
# ==========================

def main():
    print("=" * 70)
    print("🌍 URBANSHIELD AI - TREINAMENTO U-NET")
    print("=" * 70)
    
    # ========== 1. PREPARAR DADOS ==========
    print(f"\n📂 Carregando dataset...")
    
    # Dataset completo
    full_dataset = SatelliteDataset(
        image_dir=str(DATA_DIR / "images"),
        mask_dir=str(DATA_DIR / "masks"),
        transform=get_transforms(train=True)
    )
    
    # Dividir em treino (80%) e validação (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    # Criar DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"✅ Dataset preparado:")
    print(f"   Treino: {train_size} imagens ({len(train_loader)} batches)")
    print(f"   Validação: {val_size} imagens ({len(val_loader)} batches)")
    
    # ========== 2. CRIAR MODELO ==========
    print(f"\n🧠 Inicializando U-Net...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    print(f"✅ Modelo criado:")
    print(f"   Device: {DEVICE}")
    print(f"   Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 3. OTIMIZADOR E LOSS ==========
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = DiceBCELoss()
    
    print(f"\n⚙️  Configurações de treino:")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Épocas: {NUM_EPOCHS}")
    print(f"   Loss: Dice + BCE")
    
    # Carregar checkpoint se existir
    if LOAD_MODEL:
        checkpoint_path = MODEL_DIR / "checkpoint.pth.tar"
        if checkpoint_path.exists():
            load_checkpoint(torch.load(checkpoint_path), model, optimizer)
    
    # ========== 4. LOOP DE TREINAMENTO ==========
    print("\n" + "=" * 70)
    print("🚀 INICIANDO TREINAMENTO")
    print("=" * 70)
    
    best_dice = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📊 ÉPOCA {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)
        
        # Treinar
        train_metrics = train_one_epoch(
            train_loader, model, optimizer, loss_fn, DEVICE
        )
        
        # Validar
        val_metrics = validate(val_loader, model, loss_fn, DEVICE)
        
        # Exibir métricas
        print(f"\n📈 RESULTADOS DA ÉPOCA {epoch+1}:")
        print(f"   TREINO    → Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"IoU: {train_metrics['iou']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}")
        print(f"   VALIDAÇÃO → Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f}")
        
        # Salvar melhor modelo
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice
            }
            save_checkpoint(checkpoint, MODEL_DIR / "best_model.pth.tar")
            print(f"   ⭐ Novo melhor modelo! Dice: {best_dice:.4f}")
        
        # Salvar checkpoint regular
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        save_checkpoint(checkpoint, MODEL_DIR / "checkpoint.pth.tar")
    
    # ========== 5. FINALIZAÇÃO ==========
    print("\n" + "=" * 70)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("=" * 70)
    print(f"🏆 Melhor Dice Score: {best_dice:.4f}")
    print(f"💾 Modelo salvo em: {MODEL_DIR / 'best_model.pth.tar'}")
    print("\n📌 Próximo passo: python src/forecasting/train_lstm.py")


if __name__ == "__main__":
    main()
