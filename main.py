import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

from utils import *
from dataset import *
from augmentations import *

from torchvision.transforms.functional import to_pil_image

import matplotlib
matplotlib.use('TkAgg') 

print("***********************************")
print("******* INSTANTIATING MODEL *******")
print("***********************************")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("***********************************")
print("******** STARTING TRAINING ********")
print("***********************************")

mean, std = get_avg_std_entropy(model, processor, dataloader, classnames, iterations = 10)
print(mean, std)
