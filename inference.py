import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------- CONFIG ----------------

IMG_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model.pth"
)

DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

# ---------------- MODEL ----------------

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 64)

        self.pool = torch.nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)

        self.up = torch.nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )

        self.dec = DoubleConv(128, 64)

        self.out = torch.nn.Conv2d(64, 3, 1)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        d = self.up(e2)

        d = torch.cat([d, e1], dim=1)

        d = self.dec(d)

        return torch.sigmoid(self.out(d))


# ---------------- LOAD MODEL ----------------

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully")


# ---------------- IMAGE TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])


# ---------------- MAIN FUNCTION ----------------

def run_inference(input_path, output_path):

    try:

        # Load image
        img = Image.open(input_path).convert("L")

        # Transform image
        x = transform(img).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():

            out = model(x)

            out = (
                out.cpu()
                .squeeze()
                .permute(1, 2, 0)
                .numpy()
            )

        # Convert output
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

        # Save output
        Image.fromarray(out).save(output_path)

        print(f"Saved → {output_path}")

        return True

    except Exception as e:

        print("Inference Error:", str(e))

        return False