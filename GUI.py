import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import torchvision.models as models
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class ResNetCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetCNN, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Load the model
model_path = r"E:\FINAL YEAR PROJECT\CODE files\Ensembled RF,Resnet\resnet_cnn_fake_audio.pth"
model = ResNetCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocessing function
def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel_spec = transform(waveform)
    mel_spec = mel_spec.unsqueeze(0)
    time_dim = mel_spec.shape[-1]
    if time_dim < 224:
        mel_spec = F.pad(mel_spec, (0, 224 - time_dim))
    else:
        mel_spec = mel_spec[:, :, :, :224]
    mel_spec = mel_spec.expand(-1, 3, -1, -1)
    return mel_spec.squeeze(0), waveform

# Prediction
def predict(file_path):
    mel_spec, waveform = preprocess_audio(file_path)
    mel_spec = mel_spec.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(mel_spec)
        _, prediction = torch.max(output, 1)
    return "REAL" if prediction.item() == 0 else "FAKE", waveform

# Browse and analyze
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.flac;*.mp3")])
    if file_path:
        prediction, waveform = predict(file_path)
        display_spectrogram(file_path, prediction, waveform)

# Show spectrogram
def display_spectrogram(file_path, prediction, waveform):
    plt.figure(figsize=(5, 4))
    transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    mel_spec = transform(waveform)
    plt.imshow(mel_spec.log2().numpy()[0], aspect='auto', origin='lower')
    plt.title(f"Mel Spectrogram\n{os.path.basename(file_path)}", color="white")
    plt.xlabel("Time Frames", color="white")
    plt.ylabel("Mel Frequency Bins", color="white")
    plt.colorbar(label="Intensity (dB)")
    plt.tight_layout()
    plt.savefig("spectrogram.png", facecolor='black')
    plt.close()
    
    img = Image.open("spectrogram.png")
    img = img.resize((400, 300), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    
    label_img.config(image=img)
    label_img.image = img
    label_prediction.config(text=f"Prediction: {prediction}", fg="white")
    
    root.configure(bg="green" if prediction == "REAL" else "red")

# GUI Setup
root = Tk()
root.title("Fake Audio Detector")
root.geometry("500x500")
root.configure(bg="black")

frame = Frame(root, bg="black")
frame.pack(pady=20)

label_img = Label(frame, bg="black")
label_img.pack()

label_prediction = Label(root, text="", font=("Arial", 14, "bold"), bg="black", fg="white")
label_prediction.pack(pady=10)

btn_browse = Button(root, text="Browse Audio File", command=browse_file, font=("Arial", 12), bg="gray", fg="white")
btn_browse.pack()

root.mainloop()
