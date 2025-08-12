import torch

# GaussianNoise custom class
class AddGaussianNoise:
  def __init__(self, mean=0.0, std=0.1):
    self.mean = mean
    self.std = std

  def __call__(self, img):
    noise = torch.randn(img.size()) * self.std + self.mean
    output = img + noise
    return torch.clamp(output, 0.0, 1.0)

# SaltAndPepper custom class
class AddSaltAndPepperNoise(object):
  def __init__(self, prob=0.01):
    self.prob = prob

  def __call__(self, tensor):
    mask = torch.rand_like(tensor)
    tensor[mask < self.prob] = 0  # Salt
    tensor[mask > (1 - self.prob)] = 1  # Pepper
    return tensor

import random

class PatchGaussianAugmentation(torch.nn.Module):
  def __init__(self, num_patches=5, patch_size=32, std=0.1):
    super().__init__()
    self.num_patches = num_patches
    self.patch_size = patch_size
    self.std = std

  def forward(self, img):
    c, h, w = img.shape
    for _ in range(self.num_patches):
      x = random.randint(0, w - self.patch_size)
      y = random.randint(0, h - self.patch_size)

      noise = torch.randn((c, self.patch_size, self.patch_size), device=img.device) * self.std
      img[:, y:y+self.patch_size, x:x+self.patch_size] += noise
      img = img.clamp(0, 1)
    return img


import torchvision.transforms.functional as F
from PIL import Image

class RandomCropAndRescale:
  def __init__(self, output_size=(64, 64), scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
    self.output_size = output_size
    self.scale = scale
    self.ratio = ratio

  def __call__(self, img):
    width, height = img.size
    area = height * width

    for _ in range(10):  # Try 10 times to find a valid crop
      target_area = random.uniform(self.scale[0], self.scale[1]) * area
      log_ratio = torch.log(torch.tensor(self.ratio))
      aspect_ratio = torch.exp(
          random.uniform(log_ratio[0], log_ratio[1])
      ).item()

      crop_w = int(round(torch.sqrt(torch.tensor(target_area / aspect_ratio)).item()))
      crop_h = int(round(torch.sqrt(torch.tensor(target_area * aspect_ratio)).item()))

      if crop_w <= width and crop_h <= height:
          max_x = width - crop_w
          max_y = height - crop_h
          start_x = random.randint(0, max_x)
          start_y = random.randint(0, max_y)
          crop = F.crop(img, start_y, start_x, crop_h, crop_w)
          return F.resize(crop, self.output_size)

    return F.resize(F.center_crop(img, min(height, width)), self.output_size)

# augmentations that highlights the image around canny edges
import cv2 as cv
import numpy as np

class CannyEdgesAugmentation(torch.nn.Module):
  def __init__(self, kernel_size=9):
    super(CannyEdgesAugmentation, self).__init__()
    self.kernel_size = kernel_size

  def forward(self, img):
    # compute canny edges
    greyscale = torch.mean(img, dim=0).numpy()
    greyscale = (greyscale * 255).astype('uint8')
    canny_edges = cv.Canny(greyscale, 150, 180)

    # spatial smoothing of canny edges
    k = self.kernel_size # kernel size for blurring (hyperparameter ?)
    blurred_edges = cv.GaussianBlur(canny_edges, (k, k), 0)
    blurred_edges = torch.tensor(blurred_edges)
    blurred_edges = torch.clip(blurred_edges, 0, 1)

    # use spatially smoothed canny edges as a mask on the image
    C, _, _ = img.shape
    new_image = img.clone()
    for i in range(C):
      new_image[i] = img[i] * blurred_edges
    return new_image

class SpectralResidualAugmentation(torch.nn.Module):
  def __init__(self, kernel=9):
    super(SpectralResidualAugmentation, self).__init__()
    self.k = kernel

  def saliency_masks(self, img):
    C, _, _ = img.shape

    masks = []
    for c in range(C):  # Process each channel independently
        channel = img[c].cpu().numpy()

        # Apply FFT and shift to center
        fft = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft)

        # Compute log magnitude spectrum
        magnitude = np.abs(fft_shift)
        log_magnitude = np.log1p(magnitude)

        # Smooth the spectrum using a box filter (simple low-pass filter)
        k = self.k
        smoothed = cv.blur(log_magnitude, (k, k))

        # Compute the spectral residual
        spectral_residual = log_magnitude - smoothed

        # Reconstruct FFT with modified spectrum
        magnitude_residual = np.expm1(spectral_residual)
        fft_new = (magnitude_residual / magnitude) * fft_shift

        # Inverse FFT to get the saliency map
        fft_ishift = np.fft.ifftshift(fft_new)
        saliency = np.abs(np.fft.ifft2(fft_ishift))

        # Normalize to [0, 1] range
        saliency -= saliency.min()
        saliency /= saliency.max()
        masks.append(saliency)

    return masks

  def forward(self, img):
    # type(img) == torch.Tensor
    saliency_maps = self.saliency_masks(img)
    C, _, _ = img.shape

    for i in range(C):
      img[i] = img[i] * saliency_maps[i]
    return img

def spectral_residual_saliency(img, method="mean"):
    C, _, _ = img.shape
    k = 9
    masks = []
    for c in range(C):  # Process each channel independently
        channel = img[c].cpu().numpy()

        # Apply FFT and shift to center
        fft = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft)

        # Compute log magnitude spectrum
        magnitude = np.abs(fft_shift)
        log_magnitude = np.log1p(magnitude)

        # Smooth the spectrum using a box filter (simple low-pass filter)
        smoothed = cv.blur(log_magnitude, (k, k))

        # Compute the spectral residual
        spectral_residual = log_magnitude - smoothed

        # Reconstruct FFT with modified spectrum
        magnitude_residual = np.expm1(spectral_residual)
        fft_new = (magnitude_residual / magnitude) * fft_shift

        # Inverse FFT to get the saliency map
        fft_ishift = np.fft.ifftshift(fft_new)
        saliency = np.abs(np.fft.ifft2(fft_ishift))

        # Normalize to [0, 1] range
        saliency -= saliency.min()
        saliency /= saliency.max()
        masks.append(saliency)
    masks = np.array(masks)
    if method == "mean":
      mask = masks.mean(axis=0)
    return mask

from scipy.ndimage import uniform_filter
class KeepAugment(torch.nn.Module):
  def __init__(self, window_size, filter_size, augment_image):
    super(KeepAugment, self).__init__()

    self.window_size = window_size
    self.filter_size = filter_size
    self.augment_image = augment_image

  def forward(self, img):
    window_size = self.window_size
    saliency_map = spectral_residual_saliency(img)

    # H, W or W, H?
    H, W = saliency_map.shape

    filtered = uniform_filter(saliency_map, size=self.filter_size, mode='constant')
    filtered = torch.tensor(filtered)

    res = ((filtered == torch.max(filtered)).nonzero()).squeeze() 
    x = res[0]
    y = res[1]

    # check that the box does not go out of bounds
    x = max(self.window_size / 2, x)
    x = min(H-self.window_size / 2, x)

    y = max(self.window_size / 2, y)
    y = min(W-self.window_size / 2, y)

    # apply augmentations
    aug = self.augment_image(img)

    # copy the region to be kept intact 
    xmin, xmax = (x - self.window_size / 2), (x + self.window_size / 2)
    ymin, ymax = (y - self.window_size / 2), (y + self.window_size / 2)

    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    aug[:, xmin: xmax, ymin: ymax] = img[:, xmin: xmax, ymin: ymax]
    return aug
