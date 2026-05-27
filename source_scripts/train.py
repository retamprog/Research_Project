import os
import sys
import numpy as np
import time
import pandas as pd
from utility_scripts.metrics import compute_all_metrics
from tqdm import tqdm
from config import (
    DEVICE,NUM_WORKERS,NUM_EPOCHS,LEARNING_RATE,WEIGHT_DECAY,MODEL_SAVE_DIR
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# A getter function for loading the model
def get_model(model:str,freq_bins:int =40,in_channels:int =1):
    from models.custom_cnn import Custom_cnn
    from models.efficient_net_lite import EfficientNetLite
    from models.mobile_net import MobileNetCNN

    if model == 'custom_cnn':
        return Custom_cnn(in_channels = in_channels,freq_bins=freq_bins)
    elif model =='efficientnet_lite':
        return EfficientNetLite(in_channels=in_channels)
    elif model == 'mobilenet_cnn':
        return MobileNetCNN(in_channels=in_channels)
    else:
        raise ValueError(f'Unknown model: {model}')
    
def train_one_epoch(model:nn.Module,
                    loader:DataLoader,
                    optimizer:torch.optim.Optimizer,
                    criterion:nn.Module,
                    device:torch.device,
                    scaler=None
                    ) -> dict:
    # this function is just one part of the training check pipeline for single epoch
    model.train() # transitioning the model into training mode
    total_loss = 0.0
    all_preds , all_labels = [], []
    
    for features,labels , _ in tqdm(loader,desc ="  Train",leave=False):
        # moving the feature data and label data to the GPU inorder to load the model and the data both in the GPU otherwise gone case
        features = features.to(device,non_blocking = True)
        labels = labels.to(device,non_blocking = True)
        # initializing the optimizer
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                logits = model(features)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()*labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()*100
    return { 'Loss': round(avg_loss,4), 'Accuracy': round(acc,2)}

@torch.no_grad()    
def evaluate(model:nn.Module,loader:DataLoader,criterion: nn.Module,device:torch.device):
    model.eval() # Transistioning model in eval mode
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    for features,labels,_ in tqdm(loader,desc=" Eval ",leave = False):
        features = features.to(device,non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        logits = model(features)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1).cpu().numpy()
        scores = probs[:, 1].cpu().numpy()  # spoof score

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores)

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_all_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores)
    )
    metrics["loss"] = round(avg_loss, 4)
    return metrics

def train(model_name: str,
          feature_type: str,
          train_loader: DataLoader,
          dev_loader: DataLoader,
          freq_bins: int = 40,
          num_epochs: int = NUM_EPOCHS,
          lr: float = LEARNING_RATE,
          device_str: str = DEVICE,
          save_best: bool = True) -> dict:
    """
    Full training pipeline for one model + feature combination.
    Returns history dict with per-epoch metrics.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()} | Feature: {feature_type.upper()}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    model     = get_model(model_name, freq_bins=freq_bins).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = torch.amp.GradScaler(device='cuda',) if device.type == "cuda" else None

    print(f"Model parameters: {model.count_parameters():,}")

    history     = {"train_loss": [], "train_acc": [], "val_loss": [], "val_metrics": []}
    best_eer    = float("inf")
    best_f1     = 0.0
    best_path   = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{feature_type}_best.pt")

    for epoch in range(1, num_epochs + 1):
        t0         = time.time()
        train_info = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_info   = evaluate(model, dev_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{num_epochs} | "
              f"Loss: {train_info['loss']:.4f} → {val_info['loss']:.4f} | "
              f"Acc: {train_info['accuracy']:.1f}% → {val_info['accuracy']:.1f}% | "
              f"EER: {val_info['eer']:.2f}% | "
              f"F1: {val_info['f1_score']:.2f}% | "
              f"{elapsed:.1f}s")

        history["train_loss"].append(train_info["loss"])
        history["train_acc"].append(train_info["accuracy"])
        history["val_loss"].append(val_info["loss"])
        history["val_metrics"].append(val_info)

        # Save best model by EER (lower is better) with F1 as tiebreaker
        if val_info["eer"] < best_eer or (val_info["eer"] == best_eer and val_info["f1_score"] > best_f1):
            best_eer = val_info["eer"]
            best_f1  = val_info["f1_score"]
            if save_best:
                torch.save(model.state_dict(), best_path)
                print(f"  ✔ New best model saved (EER={best_eer:.2f}%)")

    print(f"\nTraining complete. Best EER: {best_eer:.2f}%")
    history["best_eer"]  = best_eer
    history["best_f1"]   = best_f1
    history["best_path"] = best_path
    return history


