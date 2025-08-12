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

acc_og = 0 
acc_top5_og = 0

import time
start = time.time()

mean = 4.8375
std = 0.0137

tot_time = time.time() - start
hyps, refs, data =  training_loop(model, processor, dataloader, classnames, threshold, mean, 1e-3)

print(f"execution took {tot_time} seconds")

print("***********************************")
print("*********** STATISTICS ************")
print("***********************************")

acc = accuracy(refs, hyps) 
acc_top5 = top_5_accuracy(refs, hyps)

print(f"accuracy: {acc}")
print(f"top 5 accuracy: {acc_top5}")

# plotting training data
x = [k for k in data.keys()]
y = [v for v in data.values()]
plt.bar(x, y)

plt.savefig("training-data.png")
