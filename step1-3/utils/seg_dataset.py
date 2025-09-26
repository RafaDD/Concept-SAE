import os
import torch
from torch.utils.data import Dataset
import pickle
import json
from PIL import Image


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

        self.concept_to_seg = {}
        for seg_name in self.seg_names:
            for concept in concepts[seg_name]:
                self.concept_to_seg[concept] = seg_name
                self.concepts.append(concept)

        self.full_segs = {}
        for seg_name in self.seg_names:
            with open(f"segs_imagenet/{seg_name}.pkl", 'rb') as f:
                self.full_segs[seg_name] = pickle.load(f)

        concept_labels = json.load(open("segs_imagenet/imagenet_answer.json"))
        self.labels = []
        for img_name, _ in self.samples:
            key = 'images/full_imagenet/imagenet_val/' + img_name
            label = []
            for concept in self.concepts:
                if concept in concept_labels[key] and concept_labels[key][concept].lower() == "yes":
                    label.append(1)
                else:
                    label.append(0)
            self.labels.append(torch.tensor(label, dtype=torch.float))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]

        seg_maps = []
        for concept in self.concepts:
            seg_name = self.concept_to_seg[concept]
            k = 'images/full_imagenet/imagenet_val/' + img_name
            seg_map = self.full_segs[seg_name][k]
            seg_maps.append(torch.tensor(seg_map))
        seg_maps = torch.stack(seg_maps, dim=0)

        for i, concept_label in enumerate(label):
            if concept_label == 0:
                seg_maps[i,:,:] = 0
        
        return img_path, image, label, seg_maps


class CelebAGenderDataset(Dataset):
    def __init__(self, img_dir, split_dict, gender_dict, target_split, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = [
            (img, gender_dict[img])
            for img, split in split_dict.items()
            if split == target_split
        ]
        concepts = json.load(open("segs_celeba/celeba.json"))
        self.concepts = []
        self.seg_names = list(concepts.keys())

        self.concept_to_seg = {}
        for seg_name in self.seg_names:
            for concept in concepts[seg_name]:
                self.concept_to_seg[concept] = seg_name
                self.concepts.append(concept)

        self.full_segs = {}
        for seg_name in self.seg_names:
            with open(f"segs_celeba/{seg_name}.pkl", 'rb') as f:
                self.full_segs[seg_name] = pickle.load(f)

        concept_labels = json.load(open("segs_celeba/celeba_answer.json"))
        self.labels = []
        for img_name, _ in self.samples:
            key = 'images/celeba/img_align_celeba/' + img_name
            label = []
            for concept in self.concepts:
                if concept in concept_labels[key] and concept_labels[key][concept].lower() == "yes":
                    label.append(1)
                else:
                    label.append(0)
            self.labels.append(torch.tensor(label, dtype=torch.float))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]

        seg_maps = []
        for concept in self.concepts:
            seg_name = self.concept_to_seg[concept]
            k = 'images/celeba/img_align_celeba/' + img_name
            seg_map = self.full_segs[seg_name][k]
            seg_maps.append(torch.tensor(seg_map))
        seg_maps = torch.stack(seg_maps, dim=0)

        for i, concept_label in enumerate(label):
            if concept_label == 0:
                seg_maps[i,:,:] = 0
        
        return img_path, image, label, seg_maps