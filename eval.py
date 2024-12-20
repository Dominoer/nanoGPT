# evaluation script was adapted from the following sources:
# https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/eval.py

import os
import math
import argparse
import torch
import numpy as np
from torch.nn import functional as F
from model import GPTConfig, GPT
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, model, data_path, batch_size=32, block_size=512, device='cuda'):
        self.model = model.to(device)
        self.data_path = data_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.nb = math.ceil(len(self.data) / block_size)
        
    def _prepare_batch(self, start_idx, batch_size_actual):
        """Prepare a single batch of data."""
        x = torch.zeros((batch_size_actual, self.block_size), dtype=torch.long, device=self.device)
        y = torch.zeros((batch_size_actual, self.block_size), dtype=torch.long, device=self.device)
        
        for b in range(batch_size_actual):
            start = (start_idx + b) * self.block_size
            end = start + self.block_size
            chunk = self.data[start:end]
            x[b, :len(chunk)] = torch.from_numpy((chunk).astype(np.int64))
            y[b, :len(chunk)-1] = x[b, 1:len(chunk)]
            y[b, len(chunk)-1] = x[b, 0]
            
        return x, y
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on enwik8 dataset."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(0, self.nb - 1, self.batch_size):
            batch_size_actual = min(self.batch_size, self.nb - 1 - i)
            x, y = self._prepare_batch(i, batch_size_actual)
            
            with torch.cuda.amp.autocast(enabled=True):
                logits, _ = self.model(x, y)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction='mean'
                )
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        
        avg_loss = total_loss / total_tokens
        bpb = avg_loss / math.log(2)
        
        return {
            'bpb': bpb,
            'loss': avg_loss,
            'total_tokens': total_tokens
        }
    
    @torch.no_grad()
    def analyze_positions(self):
        """Analyze prediction difficulty by position."""
        self.model.eval()
        
        stats = {
            'losses': torch.zeros(self.block_size, device=self.device),
            'counts': torch.zeros(self.block_size, device=self.device)
        }
        
        for i in range(0, self.nb - 1, self.batch_size):
            batch_size_actual = min(self.batch_size, self.nb - 1 - i)
            x, y = self._prepare_batch(i, batch_size_actual)
            
            with torch.cuda.amp.autocast(enabled=True):
                logits, _ = self.model(x, y)
                
                # Calculate metrics
                loss_per_pos = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction='none'
                ).view(batch_size_actual, -1)
                
                # Update statistics
                stats['losses'].add_(loss_per_pos.sum(0))
                stats['counts'].add_(torch.ones_like(loss_per_pos).sum(0))
        
        # Calculate average losses
        position_losses = (stats['losses'] / stats['counts']).cpu().numpy()
        
        return {
            'position_losses': position_losses,
            'total_samples': stats['counts'][0].item()
        }

def plot_position_metrics(results, save_path='position_analysis.png'):
    """Plot position-wise loss."""
    plt.figure(figsize=(10, 6))
    positions = np.arange(1, len(results['position_losses']) - 1)
    
    plt.plot(positions, results['position_losses'][1:-1])
    plt.title('Loss by Position')
    plt.xlabel('Position')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_model(model_dir):
    """Load model from checkpoint."""
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    
    model = GPT(GPTConfig(**checkpoint['model_args']))
    state_dict = checkpoint['model']
    
    # Clean up state dict
    unwanted_prefix = '__orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    return model

def print_results(results):
    """Print evaluation results."""
    print('=' * 80)
    print(f"Enwik8 Evaluation Results:")
    print(f"Bits per byte (bpb): {results['bpb']:.4f}")
    print(f"Loss (nats): {results['loss']:.4f}")
    print(f"Total tokens evaluated: {results['total_tokens']:,}")
    print('=' * 80)

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    args = parser.parse_args()
    
    # Initialize evaluator
    model = load_model(args.model_dir)
    evaluator = ModelEvaluator(
        model=model,
        data_path=args.test_data,
        batch_size=args.batch_size,
        block_size=args.block_size
    )
    
    # Run evaluations
    results = evaluator.evaluate()
    print_results(results)
    
    # Run position analysis
    if args.plot:
        print("\nRunning position difficulty analysis...")
        position_results = evaluator.analyze_positions()
    
        plot_position_metrics(position_results)
    
        print("\nPosition Analysis Summary (excluding first and last positions):")
        print(f"Average loss first 64 positions: {position_results['position_losses'][1:65].mean():.4f}")
        print(f"Average loss last 64 positions: {position_results['position_losses'][-65:-1].mean():.4f}")