import torch

# التأكد من أن PyTorch شايف كارت الشاشة
cuda_available = torch.cuda.is_available()

print("="*30)
if cuda_available:
    print("✅ Success! Your GPU is ready to work.")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # تجربة سريعة لنقل عملية حسابية للـ GPU
    x = torch.tensor([1.0, 2.0]).to("cuda")
    print(f"Test Tensor on GPU: {x}")
else:
    print("❌ Oh no! PyTorch is only seeing the CPU.")
    print("We might need to reinstall the CUDA version of Torch.")
print("="*30)