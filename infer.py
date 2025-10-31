import torch, cv2, numpy as np
from model.enet import ENet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ENet(num_classes=20).to(device)
model.load_state_dict(torch.load('enet_cityscapes.pth', map_location=device))
model.eval()

img = cv2.imread('data/sample_cityscapes/images/frankfurt_000001_000019_leftImg8bit.png')
img_resized = cv2.resize(img, (256,256))
inp = torch.from_numpy(img_resized.transpose(2,0,1)).unsqueeze(0).float().to(device)/255.0

with torch.no_grad():
    out = model(inp)
pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

colored = cv2.applyColorMap(pred*10, cv2.COLORMAP_JET)
cv2.imwrite('segmented_output.png', colored)
print(" Segmentation saved as segmented_output.png")
