"""
MorphEmbedding训练脚本
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from m_t_calculator import MorphEmbedding
from config import *

# 中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置日志


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MorphSemanticDataset(Dataset):
    """形态-语义配对数据集"""
    
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"加载训练数据: {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        m_t = torch.from_numpy(sample['m_t']).float()  # [254]
        h_t = sample['h_t'].float()  # [896]
        return m_t, h_t


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for m_t, h_t in dataloader:
        m_t = m_t.to(device)
        h_t = h_t.to(device)
        # LayerNorm 归一化
        layer_norm = torch.nn.LayerNorm(h_t.shape[-1]).to(h_t.device)
        phi_m_t = model(m_t)  # [batch, 896]
        phi_m_t = layer_norm(phi_m_t)
        h_t = layer_norm(h_t)
        # MSE损失
        loss = F.mse_loss(phi_m_t, h_t)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for m_t, h_t in dataloader:
            m_t = m_t.to(device)
            h_t = h_t.to(device)
            # LayerNorm 归一化
            layer_norm = torch.nn.LayerNorm(h_t.shape[-1]).to(h_t.device)
            phi_m_t = model(m_t)
            phi_m_t = layer_norm(phi_m_t)
            h_t = layer_norm(h_t)
            loss = F.mse_loss(phi_m_t, h_t)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_training_curve(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', linewidth=2)
    plt.plot(val_losses, label='验证损失', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('MorphEmbedding训练曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"训练曲线已保存到 {save_path}")



def train_morph_embedding(
    data_file,
    model_save_path,
    curve_save_path,
    log_file,
    morph_dim,
    hidden_dim,
    device,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    early_stop_patience=EARLY_STOP_PATIENCE,
    learning_rate=LEARNING_RATE,
):
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info(f"MorphEmbedding训练开始 | 数据: {data_file}")
    logger.info("="*70)
    # 1. 加载数据
    logger.info("[1/5] 加载数据...")
    dataset = MorphSemanticDataset(data_file)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  Batch大小: {batch_size}")
    # 2. 初始化模型
    logger.info("[2/5] 初始化模型...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = MorphEmbedding(morph_dim=morph_dim, hidden_dim=hidden_dim).to(device)
    logger.info(f"  模型: MorphEmbedding({morph_dim} -> {hidden_dim})")
    logger.info(f"  设备: {device}")
    logger.info(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    # 3. 配置优化器
    logger.info("[3/5] 配置优化器...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info(f"  优化器: Adam")
    logger.info(f"  学习率: {learning_rate}")
    logger.info(f"  权重衰减: {weight_decay}")
    # 4. 训练循环
    logger.info(f"[4/5] 开始训练 ({num_epochs} epochs, early stop patience={early_stop_patience})...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        logger.info(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"  → 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"\n  Early Stopping! 验证损失{early_stop_patience}轮未改善")
                logger.info(f"  最佳epoch: {epoch - early_stop_patience}, 最佳Val Loss: {best_val_loss:.6f}")
                break
    # 5. 保存训练曲线
    logger.info("[5/5] 保存训练结果...")
    plot_training_curve(train_losses, val_losses, curve_save_path)
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    logger.info(f"模型保存路径: {model_save_path}")
    logger.info(f"训练曲线: {curve_save_path}")
    logger.info("="*70)


if __name__ == "__main__":
    # 自动遍历 layer_models 下所有层目录
    base_dir = Path(__file__).parent / "layer_models"
    layer_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "training_data.pkl").exists()]
    print(f"发现 {len(layer_dirs)} 个层目录：{[d.name for d in layer_dirs]}")
    for layer_dir in layer_dirs:
        print(f"\n=== 训练 {layer_dir.name} ===")
        data_file = layer_dir / "training_data.pkl"
        model_save_path = layer_dir / "morph_embedding_best.pt"
        curve_save_path = layer_dir / "training_curve.png"
        log_file = layer_dir / "training.log"
        train_morph_embedding(
            data_file=data_file,
            model_save_path=model_save_path,
            curve_save_path=curve_save_path,
            log_file=log_file,
            morph_dim=MORPH_DIM,
            hidden_dim=HIDDEN_DIM,
            device=DEVICE,
        )