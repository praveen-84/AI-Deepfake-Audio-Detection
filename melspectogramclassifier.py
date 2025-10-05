import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the same architecture used during training
class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        resnet = models.resnet18(weights=None)
        layers = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

# Load ResNet model
resnet_model = MyResNet().to(device)
resnet_model.load_state_dict(torch.load("resnet_feature_extractor.pth", map_location=device))
resnet_model.eval()

# Load Random Forest model
rf_model = joblib.load("random_forest_model.pkl")  # Ensure this file exists

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# GUI Application
class AudioClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Fake Audio Classifier")
        self.master.geometry("500x600")

        self.label = tk.Label(master, text="Upload an Audio Spectrogram Image", font=("Arial", 14))
        self.label.pack(pady=20)

        self.canvas = tk.Label(master)
        self.canvas.pack()

        self.button = tk.Button(master, text="Browse Image", command=self.load_image)
        self.button.pack(pady=10)

        self.result_label = tk.Label(master, text="", font=("Arial", 16), fg="blue")
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = Image.open(file_path).convert('RGB')
            self.display_image(image)
            self.classify_image(image)

    def display_image(self, image):
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.canvas.configure(image=photo)
        self.canvas.image = photo

    def classify_image(self, image):
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet_model(input_tensor)
        features_np = features.cpu().numpy()

        prediction = rf_model.predict(features_np)
        result_text = "FAKE Audio Detected" if prediction[0] == 1 else "REAL Audio Detected"
        self.result_label.config(text=result_text)

# Run GUI
root = tk.Tk()
app = AudioClassifierApp(root)
root.mainloop()
