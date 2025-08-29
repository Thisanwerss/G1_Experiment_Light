#!/usr/bin/env python3
"""
Export PyTorch model to ONNX format for faster inference

Usage:
    python export_model_to_onnx.py --model_path nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt
"""

import argparse
import time
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import onnx
import onnxruntime as ort


class MLPRegressor(pl.LightningModule):
    """Neural network model definition (same as run_policy_pruned.py)"""
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim=1, learning_rate=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            
            nn.Linear(hidden_dim3, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def load_pytorch_model(model_path: str, device: str) -> MLPRegressor:
    """Load PyTorch model from checkpoint"""
    print(f"📦 Loading PyTorch model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same architecture
    net = MLPRegressor(95, 512, 512, 512, 164)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device).eval()
    
    print("✅ PyTorch model loaded successfully")
    return net


def export_to_onnx(model: MLPRegressor, output_path: str, device: str) -> None:
    """Export PyTorch model to ONNX format"""
    print(f"🔄 Exporting to ONNX format: {output_path}")
    
    # Create dummy input (input_dim = 95: qpos + qvel)
    dummy_input = torch.randn(1, 95, device=device, dtype=torch.float32)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("✅ ONNX export completed")


def verify_onnx_model(onnx_path: str) -> None:
    """Verify ONNX model"""
    print(f"🔍 Verifying ONNX model: {onnx_path}")
    
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print("✅ ONNX model verification passed")


def benchmark_models(pytorch_model: MLPRegressor, onnx_path: str, device: str, num_runs: int = 1000) -> Tuple[float, float]:
    """Benchmark PyTorch vs ONNX inference speed"""
    print(f"⚡ Benchmarking models ({num_runs} runs)...")
    
    # Prepare test data
    test_input_np = np.random.randn(1, 95).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np).to(device)
    
    # Benchmark PyTorch model
    pytorch_model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = pytorch_model(test_input_torch)
        
        # Actual timing
        start_time = time.time()
        for _ in range(num_runs):
            _ = pytorch_model(test_input_torch)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        pytorch_time = time.time() - start_time
    
    # Benchmark ONNX model
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, {'input': test_input_np})
    
    # Actual timing
    start_time = time.time()
    for _ in range(num_runs):
        _ = ort_session.run(None, {'input': test_input_np})
    onnx_time = time.time() - start_time
    
    pytorch_avg = pytorch_time / num_runs * 1000  # ms
    onnx_avg = onnx_time / num_runs * 1000  # ms
    
    print(f"\n📊 Benchmark Results:")
    print(f"   PyTorch: {pytorch_avg:.3f} ms/inference")
    print(f"   ONNX:    {onnx_avg:.3f} ms/inference")
    print(f"   Speedup: {pytorch_avg/onnx_avg:.2f}x")
    
    return pytorch_avg, onnx_avg


def test_output_consistency(pytorch_model: MLPRegressor, onnx_path: str, device: str) -> None:
    """Test that PyTorch and ONNX models produce consistent outputs"""
    print(f"🔧 Testing output consistency...")
    
    # Prepare test data
    test_input_np = np.random.randn(1, 95).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np).to(device)
    
    # PyTorch prediction
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input_torch).cpu().numpy()
    
    # ONNX prediction
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    onnx_output = ort_session.run(None, {'input': test_input_np})[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
    
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✅ Output consistency test passed")
    else:
        print(f"⚠️ Output difference may be too large: {max_diff}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX format for faster inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="exported_models/model.onnx",
        help="Output ONNX model path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=1000,
        help="Number of runs for benchmarking"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"🚀 Starting ONNX export process")
    print(f"   Input model: {args.model_path}")
    print(f"   Output model: {args.output_path}")
    print(f"   Device: {args.device}")
    
    try:
        # 1. Load PyTorch model
        pytorch_model = load_pytorch_model(args.model_path, args.device)
        
        # 2. Export to ONNX
        export_to_onnx(pytorch_model, args.output_path, args.device)
        
        # 3. Verify ONNX model
        verify_onnx_model(args.output_path)
        
        # 4. Test output consistency
        test_output_consistency(pytorch_model, args.output_path, args.device)
        
        # 5. Benchmark performance
        pytorch_time, onnx_time = benchmark_models(
            pytorch_model, args.output_path, args.device, args.benchmark_runs
        )
        
        print(f"\n🎉 Export completed successfully!")
        print(f"   ONNX model saved to: {args.output_path}")
        print(f"   Performance improvement: {pytorch_time/onnx_time:.2f}x faster")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 