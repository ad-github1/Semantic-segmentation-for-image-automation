import torch
from model.enet import ENet

model = ENet(num_classes=20)
model.load_state_dict(torch.load('enet_cityscapes.pth', map_location='cpu'))
model.eval()

example_input = torch.randn(1, 3, 256, 256)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("enet_cityscapes_torchscript.pt")
print(" TorchScript model exported.")
