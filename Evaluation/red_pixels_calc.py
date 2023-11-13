import os
from PIL import Image
from collections import defaultdict
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import icecream as ic
import csv
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn
import random

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb # DO: wandb login (d20484dd403502739b2e1ba11d31da009237fffb) - API key
import numpy as np
import torch.nn as nn
import torch.optim as optim
import itertools
from sklearn.model_selection import train_test_split
import torch.nn.init as init 

from PIL import Image

folder1 = "/fhome/mapsiv/QuironHelico/CroppedPatches"
folder2 = "/fhome/gia05/project/window_metadata.csv"
folder3 = "/fhome/gia05/project/metadata.csv"

window_metadata = pd.read_csv(folder2)
metadata = pd.read_csv(folder3)

metadata['helicobacter'] = metadata['DENSITAT'].apply(lambda x: 0 if x == 'NEGATIVA' else 1)

# Split the DataFrame into two datasets based on the 'helicobacter' column
helicobacter_0 = metadata[metadata['helicobacter'] == 0]
helicobacter_1_baixa = metadata[metadata['DENSITAT'] == "BAIXA"]
helicobacter_1_alta = metadata[metadata['DENSITAT'] == "ALTA"]

codis = list(helicobacter_0["CODI"].unique())
train_len = int(len(codis)*0.8)
test_len = int(len(codis))-train_len

random.seed(42) # Així cada vegada que correm el codi donarà el mateix
train_samples = random.sample(codis, train_len)
test_samples = [codi for codi in codis if codi not in train_samples]
print("1")
transform = transforms.ToTensor()

def get_sample(dict):
    test = {}
    stopper = 10
    for filename in os.listdir(folder1):
        if filename.split(".")[0].strip("_1") in dict:
            test[filename.split(".")[0].strip("_1")] = []
            sub_path = os.path.join(folder1, filename.split(".")[0])
            
            for image in os.listdir(sub_path):
                image_path = os.path.join(sub_path, image)
                with Image.open(image_path) as img:
                    test[filename.split(".")[0].strip("_1")].append(transform(img))
            stopper -= 1
            if stopper == 0:
                break
    return test
test = get_sample(test_samples)
alta_samples = list(helicobacter_1_alta['CODI'])
baixa_samples = list(helicobacter_1_baixa['CODI'])
baixa = get_sample(baixa_samples)
alta = get_sample(alta_samples)

class CustomImageDataset(Dataset):
    
    def __init__(self, dict_values, transform=None, normalize=True):

        # self.keys = list(dict_values.keys())
        # self.values = list(dict_values.values())
        self.keys = list(dict_values.keys())
        self.dict = dict_values
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
        ])

        self.normalize = normalize

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # image = self.values[idx]
        # image = self.transform(image)
        # name = self.keys[idx]
        values = []
        for v in self.dict[self.keys[idx]]:
            values.append(self.transform(v))
        return values, self.keys[idx]
        # return image, name
        # return 0, 0
        
dataset_test = CustomImageDataset(test,transform=None, normalize=False)
dataset_baixa = CustomImageDataset(baixa,transform=None, normalize=False)
dataset_alta = CustomImageDataset(alta,transform=None, normalize=False)

batch_size = 1

test_dataloader = DataLoader(dataset_test, batch_size=batch_size)
baixa_dataloader = DataLoader(dataset_baixa, batch_size=batch_size)
alta_dataloader = DataLoader(dataset_alta, batch_size=batch_size)

class Autoencoder(nn.Module):
    
    def __init__(self, structure, weight_init):

        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        for in_channels, out_channels in zip(structure["encoder"][:-1], structure["encoder"][1:]):
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.MaxPool2d(2, stride=2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for in_channels, out_channels in zip(structure["decoder"][:-1], structure["decoder"][1:]):
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Upsample(scale_factor=2))
        
        decoder_layers.append(nn.Upsample(scale_factor=2))
        decoder_layers.append(nn.ConvTranspose2d(structure["decoder"][-1], 4, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        if weight_init == "xavier":
            self.initialize_weights_xavier()
        elif weight_init == "he":
            self.initialize_weights_he()
        elif weight_init == "normal":
            self.initialize_weights_normal()

    def initialize_weights_xavier(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def initialize_weights_he(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def initialize_weights_normal(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
structure = {"encoder": [4,128, 64, 32, 8], "decoder": [8, 32, 64, 128]} # model1
#structure = {"encoder": [4, 64, 32, 8], "decoder": [8, 32, 64]} # model2
#structure = {"encoder": [4, 128, 64, 32], "decoder": [32, 64, 128]} # model3

model= Autoencoder(structure=structure, weight_init="xavier")

model.load_state_dict(torch.load("/fhome/gia05/project/helicobacter_detection/model1.pth"))
# Training loop
model.eval()

def red_pixels_percentage(image, info=True):
    
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = img.convert('RGB')
    pixels = list(img.getdata())

    total_pixels = len(pixels)
    red_pixels = sum(1 for pixel in pixels if pixel[0] > 100 and pixel[1] < 100 and pixel[2] < 100)

    red_percentage = (red_pixels / total_pixels) * 100
    if info:
        print("Percentage of red content =", red_percentage, "%")
    return red_percentage


to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()

criterion = nn.MSELoss()
print("START llegir red pixels")
def get_red_dif(dataloader):
    #list of tuples on cada tupla tindra dos valors, el primer representa el red percentage del input i el segon el del output
    dict = {}
    with torch.no_grad():
        for inputs, name in dataloader:
            name = name[0]
            outputs = model(inputs)
            sample_input = inputs[0]
            sample_output = outputs[0]
            loss = criterion(sample_input, sample_output)

            # print("MSE Loss:", loss.item())

            sample_input = sample_input.permute(1, 2, 0).detach().numpy().clip(0, 1)
            sample_output = sample_output.permute(1, 2, 0).detach().numpy().clip(0, 1)
            input_red_percentage = red_pixels_percentage(sample_input, info=False)
            output_red_percentage = red_pixels_percentage(sample_output, info=False)
            
            if name in dict:
                dict[name].append(input_red_percentage - output_red_percentage)
            else:
                dict[name] = []
                dict[name].append(input_red_percentage - output_red_percentage)    
    return dict

def red_pixel_dif(dataloader):
    red_pixels = {}
    with torch.no_grad():
        for images, name in dataloader:
            # input = to_tensor(image)
            for image in images:
                output = model(image)
                sample_input = image[0]
                
                sample_input = sample_input.permute(1, 2, 0).detach().numpy().clip(0, 1)
                sample_output = output[0].permute(1, 2, 0).detach().numpy().clip(0, 1)
                
                input_red_percentage = red_pixels_percentage(sample_input, info=False)
                output_red_percentage = red_pixels_percentage(sample_output, info=False)
            
                if name[0] in red_pixels:
                    red_pixels[name[0]].append(input_red_percentage - output_red_percentage)
                else:
                    red_pixels[name[0]] = []
                    red_pixels[name[0]].append(input_red_percentage - output_red_percentage)
                    print(red_pixels.keys())
    return red_pixels

healthy_red_values = red_pixel_dif(test_dataloader)
baixa_red_values = red_pixel_dif(baixa_dataloader)
alta_red_values = red_pixel_dif(alta_dataloader)


with open("/fhome/gia05/project/dicts/healthy_sample_extended2", 'wb') as pickle_file:
    pickle.dump(healthy_red_values, pickle_file)
with open("/fhome/gia05/project/dicts/baixa_sample_extended2", 'wb') as pickle_file:
    pickle.dump(baixa_red_values, pickle_file)
with open("/fhome/gia05/project/dicts/alta_sample_extended2", 'wb') as pickle_file:
    pickle.dump(alta_red_values, pickle_file)