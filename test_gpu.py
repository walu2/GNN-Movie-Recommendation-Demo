#!/usr/bin/env python3
"""
Test GPU availability and performance on MacBook.
"""

try:
    import torch
    import time
    
    print("🔍 Testing GPU Setup...")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check available devices
    print("\n📱 Available Devices:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if hasattr(torch.backends, 'mps'):
        print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU detected!")
        else:
            print("❌ MPS not available")
    else:
        print("❌ MPS not supported in this PyTorch version")
    
    # Determine best device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    print(f"\nSelected device: {device}")
    
    # Performance test
    print("\n⚡ Performance Test:")
    size = 2048
    
    # CPU test
    print("Testing CPU performance...")
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    start_time = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.3f} seconds")
    
    # GPU test (if available)
    if device != torch.device('cpu'):
        print(f"Testing {device} performance...")
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        
        # Warm up
        torch.matmul(x_gpu, y_gpu)
        
        start_time = time.time()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        if device.type == 'mps':
            torch.mps.synchronize()  # Wait for MPS operations to complete
        elif device.type == 'cuda':
            torch.cuda.synchronize()  # Wait for CUDA operations to complete
        gpu_time = time.time() - start_time
        
        print(f"{device} time: {gpu_time:.3f} seconds")
        print(f"🏃 Speedup: {cpu_time/gpu_time:.1f}x faster!")
        
        # Verify results match
        z_gpu_cpu = z_gpu.to('cpu')
        if torch.allclose(z_cpu, z_gpu_cpu, rtol=1e-3):
            print("✅ Results match between CPU and GPU")
        else:
            print("❌ Results don't match - potential device issue")
    
    print("\n🎯 Ready for GNN Training!")
    if device.type == 'mps':
        print("Your MacBook GPU will accelerate the neural network training significantly!")
    
except ImportError as e:
    print(f"❌ PyTorch not available: {e}")
    print("Please install PyTorch first:")
    print("pip install torch torchvision torchaudio")

except Exception as e:
    print(f"❌ Error during testing: {e}")