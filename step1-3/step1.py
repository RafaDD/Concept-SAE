import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.nn.functional as F
import argparse
import json
import numpy as np
from replace_model import Image_Tokenizer
from utils.seg_dataset import ImageNetDataset, CelebAGenderDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
catch_outputs = None



def train_tokenizer(args, model, target_model, node_name,
                                         train_loader, test_loader):
        
    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for it, (paths, image, concept_labels, seg_map) in enumerate(pbar):
            seg_map = F.interpolate(seg_map.to(device), size=(model.seg_size, model.seg_size), mode='bilinear', align_corners=False)
            optimizer.zero_grad()
            b, n, h, w = seg_map.shape
            seg_map = seg_map.view(b, n, -1)
            image = image.to(device).float()
            concept_labels = concept_labels.to(device).float().unsqueeze(-1)
            
            with torch.no_grad():
                model_out = get_model_layer_output(target_model, node_name, image)
            tokens, scores = model(model_out)  # token_maps: [B, T, HW]
            seg_loss = criterion(tokens.float(), seg_map.float())
            score_loss = criterion(scores.float(), concept_labels.float())
            l1_loss = 0.
            for layer in model.token_merge:
                linear = layer[0]
                l1_loss += torch.norm(linear.weight, p=1)
            l1_loss = 0.01 * l1_loss / len(model.token_merge)

            loss = seg_loss + score_loss + l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({
                "seg": f"{seg_loss.item():.3f}",
                "score": f"{score_loss.item():.3f}",
                "l1": f"{l1_loss.item():.3f}",
            })


        scheduler.step()
        avg_segloss, avg_scoreloss = test_tokenizer(model, target_model, node_name, test_loader)
        print(f"Epoch {epoch+1} | Test Loss: {avg_segloss:.4f} {avg_scoreloss:.4f}")
        avg_test_loss = avg_segloss + avg_scoreloss
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
            print(f"[✓] Model saved at epoch {epoch+1}")

def test_tokenizer(model, target_model, node_name, test_loader):
    total_pred_loss = 0
    total_score_loss = 0
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for paths, image, concept_labels, seg_map in tqdm(test_loader):
            seg_map = F.interpolate(seg_map.to(device), size=(model.seg_size, model.seg_size), mode='bilinear', align_corners=False)
            b, n, h, w = seg_map.shape
            seg_map = seg_map.view(b, n, -1)
            image = image.to(device).float()
            concept_labels = concept_labels.to(device).float().unsqueeze(-1)
            with torch.no_grad():
                model_out = get_model_layer_output(target_model, node_name, image)
            tokens, scores = model(model_out)
            seg_loss =  criterion(tokens, seg_map) 
            score_loss = criterion(scores, concept_labels)
            total_pred_loss += seg_loss.item()
            total_score_loss += score_loss.item()
    avg_pred_loss = total_pred_loss / len(test_loader)
    avg_score_loss = total_score_loss / len(test_loader)
    return avg_pred_loss, avg_score_loss


def get_model_layer_output(model, node_name, image):
    layer = dict(model.named_modules())[node_name]
    hook = layer.register_forward_hook(hook_fn)
    image_features = model(image)
    hook.remove()
    return catch_outputs

def hook_fn(module, input, output):
    global catch_outputs
    catch_outputs = output
    return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--node", type=int, default=5)
    parser.add_argument("--task", type=str, default='imagenet')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_args()

    target_model = models.resnet18(weights='DEFAULT').cuda().eval()

    node_names = [name for name in dict(target_model.named_modules()) if "conv" in name]
    print(node_names)
    node_name = node_names[args.node]

    model_out = get_model_layer_output(target_model, node_name, torch.randn(1,3,224,224).cuda())
    task_name = "tokenizer_bb_resnet" + node_name

    if args.task == 'imagenet':
        args.log_dir = os.path.join('data/outputs/target', task_name)
    elif args.task == 'celeba':
        args.log_dir = os.path.join('data/outputs/celeba', task_name)
    os.makedirs(args.log_dir, exist_ok=True)

    print(args.node, len(node_names), args.log_dir)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    to_tensor_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.task == 'imagenet':

        concept_dict = json.load(open("segs_imagenet/imagenet.json"))
        seg_targets = []
        for seg_name, concepts in concept_dict.items():
            for concept in concepts:
                seg_targets.append(concept)

        IMG_DIR = "data/datasets/imagenet_val"
        ATTR_FILE = "data/datasets/imagenet_val/list_attr.txt"     # Path to attributes file
        SPLIT_FILE = "data/datasets/imagenet_val/list_eval_partition.txt" # Path to partition file

        with open(ATTR_FILE, "r") as f:
            lines = f.read().strip().split("\n")

        class_dict = {}
        for line in lines:
            parts = line.split()
            img, cls = parts[0], int(parts[1])
            class_dict[img] = cls

        split_dict = {}
        with open(SPLIT_FILE, "r") as f:
            for line in f:
                img, split = line.strip().split()
                split_dict[img] = int(split)

        train_dataset = ImageNetDataset(IMG_DIR, split_dict, class_dict, target_split=0, transform=to_tensor_transform)
        test_dataset = ImageNetDataset(IMG_DIR, split_dict, class_dict, target_split=1, transform=to_tensor_transform)
    
    elif args.task == 'celeba':
        concept_dict = json.load(open("segs_celeba/celeba.json"))
        seg_targets = []
        for seg_name, concepts in concept_dict.items():
            for concept in concepts:
                seg_targets.append(concept)

        IMG_DIR = "./data/datasets/img_align_celeba"           # Path to CelebA aligned images
        ATTR_FILE = "./data/datasets/img_align_celeba/list_attr_celeba.txt"     # Path to attributes file
        SPLIT_FILE = "./data/datasets/img_align_celeba/list_eval_partition.txt" # Path to partition file

        with open(ATTR_FILE, "r") as f:
            lines = f.read().strip().split("\n")
        header = lines[1].split()
        male_idx = header.index("Male") + 1

        gender_dict = {}
        for line in lines[2:]:
            parts = line.split()
            img, male = parts[0], int(parts[male_idx])
            gender_dict[img] = 1 if male == 1 else 0  # 1 = male, 0 = female

        split_dict = {}
        with open(SPLIT_FILE, "r") as f:
            for line in f:
                img, split = line.strip().split()
                split_dict[img] = int(split)

        train_dataset = CelebAGenderDataset(IMG_DIR, split_dict, gender_dict, target_split=0, transform=to_tensor_transform)
        test_dataset = CelebAGenderDataset(IMG_DIR, split_dict, gender_dict, target_split=1, transform=to_tensor_transform)
    

    model_out = get_model_layer_output(target_model, node_name, torch.randn(1,3,224,224).cuda())

    learner_model = Image_Tokenizer(model_out.shape[1], model_out.shape[1], model_out.shape[-1], model_out.shape[-1],
                                            num_tokens=len(seg_targets)).to(device) # 16, 512, 512, 1024, 5295

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False)

    learner_model = Image_Tokenizer(model_out.shape[1], model_out.shape[1], model_out.shape[-1], model_out.shape[-1],
                                        num_tokens=len(seg_targets)).to(device) # 16, 512, 512, 1024, 5295
    train_tokenizer(args, learner_model, target_model, node_name, train_loader, test_loader)

