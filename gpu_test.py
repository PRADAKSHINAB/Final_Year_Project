import torch

print("=" * 50)
print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name        :", torch.cuda.get_device_name(0))
    print("CUDA Version    :", torch.version.cuda)

    x = torch.randn(5000, 5000, device="cuda")
    y = torch.randn(5000, 5000, device="cuda")

    z = torch.matmul(x, y)

    print("GPU Computation Successful!")
    print("Tensor Device:", z.device)
else:
    print("CUDA is NOT available!")

print("=" * 50)