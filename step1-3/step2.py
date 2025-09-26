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
from replace_model import Concept_Block
import numpy as np
from utils.img_dataset import ImageNetDataset, CelebAGenderDataset
device = "cuda" if torch.cuda.is_available() else "cpu"


def test_model(model, target_model, node_name, test_loader):
    total_pred_loss = 0
    model.eval()
    criterion = nn.MSELoss()
    for image, label in tqdm(test_loader, ncols=150):
        image, label = image.to(device).float(), label.to(device).float()
        model_out = get_model_layer_output(target_model, node_name, image)
        pred_activation, tokens, relu_tokens = model(model_out)
        pred_loss = criterion(pred_activation, model_out)
        total_pred_loss += pred_loss.item()
    avg_pred_loss = total_pred_loss / len(test_loader)
    return avg_pred_loss


def hook_fn(module, input, output):
    global catch_outputs
    catch_outputs = output
    return output

def get_model_layer_output(model, node_name, image):
    layer = dict(model.named_modules())[node_name]
    hook = layer.register_forward_hook(hook_fn)
    image_features = model(image)
    hook.remove()
    return catch_outputs


def train_model(args, model, target_model, node_name,
                                         train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    best_loss = float('inf')

    merge_weight = []
    for i in range(len(model.tokenizer.token_merge)):
        w = model.tokenizer.token_merge[i][0].weight.detach()
        merge_weight.append(w)
    merge_weight = torch.cat(merge_weight, dim=0) # [num_tokens, channel]
    merge_weight = F.softmax(merge_weight.abs() / 0.02, dim=1)

    for epoch in range(args.epochs):
        model.train()
        total_pred_loss = total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=150)
        for it, (image, label) in enumerate(pbar):
            optimizer.zero_grad()
            image = image.to(device).float()
            with torch.no_grad():
                model_out = get_model_layer_output(target_model, node_name, image)
            pred_activation, tokens, token_aggregator = model(model_out)  # token_maps: [B, T, HW]
            pred_loss = criterion(pred_activation, model_out)

            token_aggregator = token_aggregator.T
            token_aggregator_sm = F.softmax(token_aggregator.abs() / 0.02, dim=1)
            divergence_loss = torch.sum(merge_weight * torch.log(merge_weight / token_aggregator_sm), dim=1).mean()

            sparse_loss = token_aggregator.abs().mean()

            loss = pred_loss + 1e-2 * divergence_loss + sparse_loss
            
            loss.backward()
            optimizer.step()

            total_pred_loss += pred_loss.item()
            total_loss += loss.item()

            pbar.set_postfix({
                "pred": f"{pred_loss.item():.3f}",
                "div": f"{divergence_loss.item():.3f}",
                "sparse": f"{sparse_loss.item():.3f}"
            })

        scheduler.step()
        avg_testloss = test_model(model, target_model, node_name, test_loader)
        print(f"Epoch {epoch+1} | Test Loss: {avg_testloss:.4f}")

        if avg_testloss < best_loss:
            best_loss = avg_testloss
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
            print(f"[✓] Model saved at epoch {epoch+1}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--node", type=int, default=5)
    parser.add_argument("--task", type=str, default="imagenet")
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
    node_name = node_names[args.node]

    model_out = get_model_layer_output(target_model, node_name, torch.randn(1,3,224,224).cuda())
    task_name = "concept_module_resnet" + node_name
    
    if args.task == 'imagenet':
        args.log_dir = os.path.join('data/outputs/target', task_name)
    elif args.task == 'celeba':
        args.log_dir = os.path.join('data/outputs/celeba', task_name)
    os.makedirs(args.log_dir, exist_ok=True)

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

    learner_model = Concept_Block(model_out.shape[1], model_out.shape[1], model_out.shape[-1], model_out.shape[-1],
                                        num_tokens=len(seg_targets)).to(device) 
    if args.task == 'imagenet':
        learner_model.tokenizer.load_state_dict(torch.load(os.path.join("data/outputs/target/tokenizer_bb_resnet%s"%node_name, 'best_model.pth')))
    elif args.task == 'celeba':
        learner_model.tokenizer.load_state_dict(torch.load(os.path.join("data/outputs/celeba/tokenizer_bb_resnet%s"%node_name, 'best_model.pth')))


    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False)
    train_model(args, learner_model, target_model, node_name, train_loader, test_loader)