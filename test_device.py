#!/usr/bin/env python3
"""Test script to verify device detection"""
import torch

def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("CUDA available:", torch.cuda.is_available())
    device = get_device()
    print(f"\nSelected device: {device.upper()}")

    # Test tensor creation on the device
    try:
        test_tensor = torch.randn(3, 3).to(device)
        print(f"Successfully created tensor on {device}:")
        print(test_tensor)
    except Exception as e:
        print(f"Error creating tensor on {device}: {e}")

