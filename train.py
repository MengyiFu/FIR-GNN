import argparse
import os
import time
import warnings
import numpy as np
import torch
import yaml
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from models.gnn import GNNFactory
from utils.data_processor import DataProcessor

# 过滤 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class GNNTrainer:
    """
    Attributes:
        config (dict)
        model (nn.Module)
        device (torch.device)
        optimizer (torch.optim.Optimizer)
        criterion (nn.Module)
        monitor (Monitor)
        scheduler (optional)
        best_metric (float)
    """

    def __init__(self, config: dict, model: torch.nn.Module, num_classes):
        self.config = config
        self.model = model.to(config['train']['device'])
        self.device = config['train']['device']
        self.epochs = config['train']['epochs']
        self.patience = config['train'].get('patience', 10)  # 早停耐心值

        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train'].get('weight_decay', 5e-4)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        # 学习率调度器
        self.scheduler = None
        if config['train'].get('use_scheduler', False):
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )

        self.best_metric = 0.0
        self.early_stop_counter = 0

        # 模型保存路径
        self.checkpoint_dir = os.path.join(config['train']['save_dir'], 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return out, loss.item()

    @torch.no_grad()
    def evaluate(self, data, stage='val'):
        """
        Args:
            data: PyG Data
            stage: ('train', 'val', 'test')
        Returns:
            metrics
            all_probs
        """
        self.model.eval()
        if stage == 'train':
            mask = data.train_mask
        else:
            mask = data.test_mask

        out = self.model(data)
        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true = data.y[mask].cpu().numpy()
        y_pred = preds[mask].cpu().numpy()
        y_probs = probs[mask].cpu().numpy()

        metrics = {
            'loss': self.criterion(out[mask], data.y[mask]).item(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_probs, multi_class='ovr')
        }

        return metrics, y_true, y_pred, y_probs

    def update_best_model(self, current_metric: float, epoch: int):
        """update best model and save checkpoints"""
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.early_stop_counter = 0
            self.save_checkpoint(epoch, is_best=True)
        else:
            self.early_stop_counter += 1

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric
        }

        filename = f'checkpoint_epoch{epoch}.pth.tar'
        if is_best:
            filename = 'model_best.pth.tar'

        torch.save(state, os.path.join(self.checkpoint_dir, filename))

    def train(self, data: Data):
        print(f"Start training. There are a total of {self.epochs} epochs.")
        data = data.to(self.device)

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            # tain
            out, train_loss = self.train_epoch(data)

            # evaluate
            train_metrics, _, _, _ = self.evaluate(data, 'train')
            val_metrics, val_true, val_preds, val_probs = self.evaluate(data, 'test')  # 假设使用test_mask作为验证

            if self.scheduler:
                self.scheduler.step(val_metrics['auc'])

            self.update_best_model(val_metrics['auc'], epoch)

            # 打印日志
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{self.epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} (Best: {self.best_metric:.4f})")

            # 早停检查
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 训练结束保存最终模型
        self.save_checkpoint(self.epochs)

    def test(self, data: Data) -> dict:
        print("\nStart the final test...")
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model_best.pth.tar')
        self.model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

        test_metrics, test_true, test_preds, test_probs  = self.evaluate(data, 'test')

        print(f"Test results："
              f"Accuracy: {test_metrics['accuracy']:.4f} | "
              f"F1: {test_metrics['f1']:.4f} | "
              f"AUC: {test_metrics['auc']:.4f}")
        return test_metrics

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    parser = argparse.ArgumentParser(description="GNN Training")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--graph_dir", type=str, required=True,
                        help="Graph file path")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    processor = DataProcessor(config, args.graph_dir)
    node_data, edge_data = processor.load_data()
    pyg_data, num_classes = processor.create_pyg_data(node_data, edge_data)

    model = GNNFactory.create_model(config, pyg_data.num_node_features, num_classes)
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量：{total_params}，可训练参数量：{trainable_params}")

    trainer = GNNTrainer(config, model, num_classes)
    trainer.train(pyg_data)
    trainer.test(pyg_data)


if __name__ == "__main__":
    main()