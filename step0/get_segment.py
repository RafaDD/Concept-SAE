import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
from tqdm import tqdm
import random
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoProcessor, CLIPSegForImageSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"
seg_processor = AutoProcessor.from_pretrained("segmentation/clipseg/processor/models--CIDAS--clipseg-rd64-refined/snapshots/999e0328d9e10b484360c477313983f9afdd7050")
seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

def get_log_dir(task_name=""):
    log_dir = os.path.join('data/outputs/target', task_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

class Image_dataset(Dataset):
    def __init__(self, root, rand_transform=None, to_tensor_transform=None):
        self.root = root
        self.rand_transform = rand_transform
        self.to_tensor_transform = to_tensor_transform
        self.image_paths = []
        for i, path in enumerate(os.listdir(root)):
            path = os.path.join(root, path)
            self.image_paths.append(path)
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.rand_transform is not None:
            image = self.rand_transform(image)
        image = self.to_tensor_transform(image)
        return image_path, image

def save_segs(train_loader, seg_targets):
    num_seg_concepts = len(seg_targets)
    seg_maps = []
    all_paths = []
    for it, (paths, image) in enumerate(tqdm(train_loader)):
        texts = seg_targets * len(paths)
        all_paths += paths
        paths = [i for i in paths for _ in range(num_seg_concepts)]
        seg_images = [Image.open(path).convert("RGB") for path in paths]
        with torch.no_grad():
            inputs = seg_processor(text=texts, images=seg_images, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            seg_map = seg_model(**inputs).logits    
            seg_map = seg_map.unsqueeze(1)
            seg_map = F.interpolate(seg_map, size=(32, 32), mode='bilinear', align_corners=False)
            seg_map = seg_map.squeeze(1).reshape(-1, num_seg_concepts, 32, 32)
            seg_map = (seg_map[:, indices_tensor, :, :] + 14)/20
        seg_maps.append(seg_map)

    seg_maps = torch.cat(seg_maps, dim=0)
    print(seg_maps.shape, len(all_paths))
    for i, target in enumerate(seg_targets):
        results = {}
        for j, path in enumerate(all_paths):
            results[path] = seg_maps[j][i].detach().cpu().numpy().astype(np.float16)
        save_path = os.path.join("images/segs_celeba", f"{target}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    
if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    unique_seg_targets = list(json.load(open("images/celeba/celeba.json")).keys())
    key_to_index = {key: idx for idx, key in enumerate(unique_seg_targets)}
    seg_indices = [key_to_index[key] for key in unique_seg_targets]
    indices_tensor = torch.tensor(seg_indices, dtype=torch.long, device=device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    to_tensor_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    learner_model = None
    dataset = "images/celeba/img_align_celeba"
    train_dataset = Image_dataset(dataset, None, to_tensor_transform)
    dataset = "images/celeba/img_align_celeba"
    test_dataset = Image_dataset(dataset, None, to_tensor_transform)
    train_loader = DataLoader(train_dataset, batch_size=80, num_workers=4, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=80, num_workers=4, shuffle=False, drop_last=False)
    save_segs( train_loader, unique_seg_targets)
