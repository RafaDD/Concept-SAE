import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class ImageNetDataset(Dataset):
    def __init__(self, img_dir, split_dict, class_dict, target_split, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = [
            (img, class_dict[img])
            for img, split in split_dict.items()
            if split == target_split
        ]
        concepts = json.load(open("segs_imagenet/imagenet.json"))
        self.concepts = []
        self.seg_names = list(concepts.keys())

        concept_labels = json.load(open("segs_imagenet/imagenet_answer.json"))
        self.labels = []
        for img_name, _ in self.samples:
            key = 'images/full_imagenet/imagenet_val/' + img_name
            label = [0]
            self.labels.append(torch.tensor(label, dtype=torch.float))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]
        
        return image, label


class CelebAGenderDataset(Dataset):
    def __init__(self, img_dir, split_dict, gender_dict, target_split, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = [
            (img, gender_dict[img])
            for img, split in split_dict.items()
            if split == target_split
        ]

        concept_labels = json.load(open("segs_celeba/celeba_answer.json"))
        self.labels = []
        for img_name, _ in self.samples:
            key = 'images/celeba/img_align_celeba/' + img_name
            label = [0]
            self.labels.append(torch.tensor(label, dtype=torch.float))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]
        
        return image, label