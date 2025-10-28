# dataset.py
import os
import random
import cv2
import numpy as np
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import degrade_frame # Import our new helper

# The standard transform pipeline from Cell 4
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class UCF101Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading and degrading UCF101 frames.
    From Cell 3.
    """
    def __init__(self, data_dir, transform=transform, use_first_100=True):
        all_video_paths = glob(os.path.join(data_dir, '*', '*.avi'))
        
        if use_first_100:
            self.video_paths = all_video_paths[:100]
            print(f"Total videos found: {len(all_video_paths)}")
            print(f"Using limited dataset: {len(self.video_paths)} videos.")
        else:
            self.video_paths = all_video_paths
            print(f"Using full dataset: {len(self.video_paths)} videos.")
            
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Error handling for broken videos
        if frame_count == 0:
            cap.release()
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 1. Get a random clean frame
        frame_idx = random.randint(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        # Error handling for broken frames
        if not ret:
            return self.__getitem__(random.randint(0, len(self) - 1))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Create the degraded version
        degraded_np = degrade_frame(frame_rgb) # Use our helper
        
        # 3. Convert to PIL Images
        original_pil = Image.fromarray(frame_rgb)
        degraded_pil = Image.fromarray(degraded_np)
        
        # 4. Apply transforms
        if self.transform:
            original_tensor = self.transform(original_pil)
            degraded_tensor = self.transform(degraded_pil)
            
        return degraded_tensor, original_tensor
