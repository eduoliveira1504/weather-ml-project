"""
UrbanShield AI - Implementação da U-Net para Segmentação de Imagens de Satélite

ARQUITETURA U-NET:
- Encoder (contração): Extrai features em múltiplas escalas
- Decoder (expansão): Reconstrói máscara de segmentação
- Skip Connections: Preserva detalhes espaciais
"""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Bloco básico: 2x (Conv2D → BatchNorm → ReLU)
    Usado tanto no encoder quanto no decoder
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net para Segmentação Semântica
    
    Input: Imagem RGB (3, 256, 256)
    Output: Máscara binária (1, 256, 256)
    
    Parâmetros:
        in_channels: Número de canais de entrada (3 para RGB)
        out_channels: Número de classes (1 para binário)
        features: Lista com número de filtros em cada nível [64, 128, 256, 512]
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ============== ENCODER (Downsampling) ==============
        # Extrai features hierárquicas: texturas → formas → contexto
        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # ============== BOTTLENECK ==============
        # Camada mais profunda com maior campo receptivo
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # ============== DECODER (Upsampling) ==============
        # Reconstrói resolução espacial com skip connections
        for feature in reversed(features):
            # Transposed convolution para upsampling
            self.decoder_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # DoubleConv após concatenação com skip connection
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))
        
        # ============== CAMADA FINAL ==============
        # 1x1 convolution para gerar máscara de saída
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass com skip connections
        
        Args:
            x: Tensor de entrada (batch, 3, 256, 256)
        
        Returns:
            Tensor de saída (batch, 1, 256, 256)
        """
        skip_connections = []
        
        # ========== ENCODER ==========
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)  # Salva para skip connection
            x = self.pool(x)  # Reduz resolução (downsampling)
        
        # ========== BOTTLENECK ==========
        x = self.bottleneck(x)
        
        # Inverte lista de skip connections (para concatenar na ordem correta)
        skip_connections = skip_connections[::-1]
        
        # ========== DECODER ==========
        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)  # Upsampling
            skip_connection = skip_connections[idx // 2]
            
            # Verificar dimensões (necessário caso input não seja múltiplo de 16)
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            
            # Concatenar com skip connection
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder_blocks[idx + 1](x)  # DoubleConv
        
        # ========== OUTPUT ==========
        return self.final_conv(x)


def test_model():
    """Testa a arquitetura com input dummy"""
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # Batch de 1 imagem
    output = model(x)
    
    print("=" * 60)
    print("🧠 TESTE DA ARQUITETURA U-NET")
    print("=" * 60)
    print(f"Input shape:  {x.shape}")  # (1, 3, 256, 256)
    print(f"Output shape: {output.shape}")  # (1, 1, 256, 256)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print("\n✅ Arquitetura validada!")


if __name__ == "__main__":
    test_model()
