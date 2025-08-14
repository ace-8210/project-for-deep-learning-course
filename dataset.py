import torch
import os
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from utils import params

# read dataset
dataset_dir = params['dataset_dir']
dataset_path = f"{dataset_dir}/imagenet-a"
readme_path = f"{dataset_dir}/imagenet-a/README.txt"

# assert dataset path exists
assert os.path.exists(dataset_path), "dataset not found (specify dataset directory in config file, code will look for directory imagenet-a)"
assert os.path.isfile(readme_path), "readme not found, it is necessary to get classnames"

classnames = []
with open(readme_path, "r") as f:
  for i, line in enumerate(f):
    if i < 12: 
      continue
    data = line.strip().split()
    name = " ".join(data[1:])
    if name != '':
      classnames.append(name)

# import augmentations
from augmentations import AddGaussianNoise, AddSaltAndPepperNoise, PatchGaussianAugmentation, RandomCropAndRescale, CannyEdgesAugmentation, SpectralResidualAugmentation

# create custom dataset class
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, augmentations):
    self.data = dataset
    self.augmentations = augmentations

  def __getitem__(self, index):
    image, labels = self.data[index]

    images = []
    for a in augmentations:
      images.append(a(image))

    return images, [labels] * len(augmentations)

  def __len__(self):
    return len(self.data)

# creating the actual datasets and dataloaders
resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()

transform = transforms.Compose([
  resize
])

from augmentations import AddGaussianNoise, AddSaltAndPepperNoise, PatchGaussianAugmentation, RandomCropAndRescale, CannyEdgesAugmentation, SpectralResidualAugmentation
from torchvision.transforms import AugMix

# patching AugMix for reasons
class AugMixFixed(AugMix):
    def _sample_dirichlet(self, params):
        params = params.to(dtype=torch.float32)
        return torch._sample_dirichlet(params)

resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()
convert_uint8 = transforms.Lambda(lambda x: (x * 255).type(torch.uint8)) # assumes you start from float32
convert_float32 = transforms.Lambda(lambda x: x.type(torch.float32) / 255) # assumes you start from uint8

transform = transforms.Compose([
  resize
])

from torchvision.transforms import AugMix
from augmentations import KeepAugment
from torchvision.transforms import ColorJitter

AugMixT = transforms.Compose([
  to_tensor,
  convert_uint8,
  AugMixFixed(),      
  convert_float32
])

augmentations = [
    # original image
    transforms.Compose([resize]),
    
    # inject noise
    transforms.Compose([to_tensor, AddGaussianNoise(0.0, 0.1), to_pil_image]),
    transforms.Compose([to_tensor, AddGaussianNoise(0.0, 0.2), to_pil_image]),
    transforms.Compose([to_tensor, AddGaussianNoise(0.0, 0.3), to_pil_image]),
    transforms.Compose([to_tensor, AddGaussianNoise(0.1, 0.1), to_pil_image]),
    transforms.Compose([to_tensor, AddGaussianNoise(-0.1, 0.1), to_pil_image]),
    
    transforms.Compose([to_tensor, PatchGaussianAugmentation(1, 64, 0.5), to_pil_image]),
    transforms.Compose([to_tensor, PatchGaussianAugmentation(2, 128, 0.5), to_pil_image]),
    transforms.Compose([to_tensor, PatchGaussianAugmentation(3, 128, 0.5), to_pil_image]),
    transforms.Compose([to_tensor, PatchGaussianAugmentation(1, 128, 0.5), to_pil_image]),
    transforms.Compose([to_tensor, PatchGaussianAugmentation(5, 32, 0.5), to_pil_image]),

    # transform color space
    transforms.Compose([to_tensor, ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue=0.5), to_pil_image]),
    transforms.Compose([to_tensor, ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue=0.5), to_pil_image]),
    transforms.Compose([to_tensor, ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue=0.5), to_pil_image]),
    transforms.Compose([to_tensor, ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue=0.5), to_pil_image]),
    transforms.Compose([to_tensor, ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue=0.5), to_pil_image]),

    # AugMix transformations
    transforms.Compose([AugMixT, to_pil_image]),
    transforms.Compose([AugMixT, to_pil_image]),
    transforms.Compose([AugMixT, to_pil_image]),
    transforms.Compose([AugMixT, to_pil_image]),
    transforms.Compose([AugMixT, to_pil_image]),
    
    # KeepAugment + GaussianNoise
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = AddGaussianNoise(mean = 0.0, std = 0.1)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = AddGaussianNoise(mean = 0.0, std = 0.2)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = AddGaussianNoise(mean = 0.0, std = 0.3)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = AddGaussianNoise(mean = 0.2, std = 0.1)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = AddGaussianNoise(mean = -0.2, std = 0.1)),
      to_pil_image]),
    
    # KeepAugment + ColorJitter
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5)),
      to_pil_image]),
    transforms.Compose([
      to_tensor,
      KeepAugment(window_size = 128, filter_size = 12, 
      augment_image = ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5)),
      to_pil_image]),

    # 5 saliency based augmentations
    transforms.Compose([to_tensor, SpectralResidualAugmentation(9), to_pil_image]),
    transforms.Compose([to_tensor, SpectralResidualAugmentation(12), to_pil_image]),
    transforms.Compose([to_tensor, SpectralResidualAugmentation(21), to_pil_image]),
    transforms.Compose([to_tensor, SpectralResidualAugmentation(27), to_pil_image]),
    transforms.Compose([to_tensor, SpectralResidualAugmentation(32), to_pil_image]),
]

def collate_fn(batch):
  batch = batch[0]
  images, labels = batch
  return images, labels 

print("########################")
print("# creating dataloaders #")
print("########################")
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(
    CustomDataset(dataset, augmentations),
    shuffle=True,
    batch_size=1,
    collate_fn=collate_fn
)

t = transforms.Compose([resize])
original_dataset = datasets.ImageFolder(root=dataset_path, transform=t)
dataloader_plain = DataLoader(original_dataset, shuffle=True, batch_size=8)
