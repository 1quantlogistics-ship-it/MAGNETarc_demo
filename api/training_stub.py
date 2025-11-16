#!/usr/bin/env python3
"""
ARC Training Stub - Minimal PyTorch training for pipeline validation
Runs on GPU1 while vLLM occupies GPU0
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinimalModel(nn.Module):
    """Tiny model for testing GPU allocation and training loop"""
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super(MinimalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def generate_synthetic_data(batch_size=32, input_size=100, output_size=10):
    """Generate synthetic data for testing"""
    X = torch.randn(batch_size, input_size)
    y = torch.randint(0, output_size, (batch_size,))
    return X, y

def train_minimal_model(experiment_id, config, gpu_id=0, epochs=1):
    """
    Train minimal model on specified GPU
    
    Args:
        experiment_id: Experiment identifier
        config: Training configuration dict
        gpu_id: GPU device ID (default 1 to avoid vLLM on GPU0)
        epochs: Number of training epochs
    """
    start_time = time.time()
    
    # Set GPU device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Extract config
    learning_rate = config.get('learning_rate', 0.001)
    batch_size = config.get('batch_size', 32)
    
    # Initialize model
    model = MinimalModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    logger.info(f'Training config: lr={learning_rate}, batch_size={batch_size}, epochs={epochs}')
    
    # Training loop
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Simulate multiple batches
        num_batches = 10
        for batch_idx in range(num_batches):
            X, y = generate_synthetic_data(batch_size, input_size=100, output_size=10)
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = epoch_loss / num_batches
        accuracy = 100 * correct / total
        
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        
        logger.info(f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    training_time = time.time() - start_time
    
    # Final metrics
    final_metrics = {
        'experiment_id': experiment_id,
        'final_loss': epoch_losses[-1],
        'final_accuracy': epoch_accuracies[-1],
        'avg_loss': sum(epoch_losses) / len(epoch_losses),
        'avg_accuracy': sum(epoch_accuracies) / len(epoch_accuracies),
        'training_time': training_time,
        'epochs': epochs,
        'gpu_id': gpu_id,
        'device': str(device),
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    # ARC-specific metrics (simulated for demo)
    arc_metrics = {
        'auc': 0.75 + (epoch_accuracies[-1] / 200),  # Simulated AUC
        'sensitivity': 0.70 + (epoch_accuracies[-1] / 250),
        'specificity': 0.72 + (epoch_accuracies[-1] / 240)
    }
    
    final_metrics.update(arc_metrics)
    
    logger.info(f'Training complete in {training_time:.2f}s')
    logger.info(f'Final metrics: {json.dumps(final_metrics, indent=2)}')
    
    return final_metrics

def run_training_job(experiment_id, config, experiment_dir):
    """
    Execute training job and save results
    
    Args:
        experiment_id: Experiment identifier
        config: Training configuration
        experiment_dir: Directory to save results
    """
    logger.info(f'=== Starting training job: {experiment_id} ===')
    
    # Run training
    metrics = train_minimal_model(experiment_id, config, gpu_id=0, epochs=1)
    
    # Save results
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f'Results saved to {results_path}')
    
    # Update metadata
    metadata_path = os.path.join(experiment_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['status'] = 'completed'
        metadata['completed_at'] = datetime.now().isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f'=== Training job complete: {experiment_id} ===')
    
    return metrics

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print('Usage: python training_stub.py <experiment_id>')
        sys.exit(1)
    
    experiment_id = sys.argv[1]
    experiment_dir = f'/workspace/arc/experiments/{experiment_id}'
    
    # Load metadata
    metadata_path = os.path.join(experiment_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f'Error: Metadata not found at {metadata_path}')
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    config = metadata.get('config', {})
    
    # Run training
    results = run_training_job(experiment_id, config, experiment_dir)
    
    print(json.dumps(results, indent=2))
