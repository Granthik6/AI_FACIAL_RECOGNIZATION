"""Shared training, validation and held-out evaluation for the FER category task."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from prepare_fer2013 import DATA, ROOT, RESULTS, FIGURES, CLASSES


def network(kind, pretrained=False):
    if kind == "mobilenet":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,len(CLASSES))
        return model
    layers = []
    inputs = 1
    for outputs in (32,64,96):
        layers.extend([nn.Conv2d(inputs,outputs,3,padding=1,bias=False),
                       nn.BatchNorm2d(outputs),nn.ReLU(),nn.MaxPool2d(2)])
        inputs = outputs
    return nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.25),
                         nn.Linear(inputs,len(CLASSES)))


class Faces(Dataset):
    def __init__(self, data, kind, augment=False):
        self.data = data.reset_index(drop=True)
        self.kind = kind
        size = 96 if kind=="mobilenet" else 48
        steps = [transforms.Resize((size,size))]
        if augment:
            steps += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8)]
        mean,std = ((.485,.456,.406),(.229,.224,.225)) if kind=="mobilenet" else ((.5,),(.5,))
        self.transform = transforms.Compose([*steps,transforms.ToTensor(),transforms.Normalize(mean,std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        with Image.open(DATA/row.path) as img:
            img = img.convert("RGB" if self.kind=="mobilenet" else "L")
            return self.transform(img), CLASSES.index(row.label)


def epoch(model, loader, loss_fn, device, optimizer=None, description=None):
    model.train(optimizer is not None)
    total = 0.
    actual, predicted = [], []
    correct = 0
    with torch.set_grad_enabled(optimizer is not None), tqdm(
        total=len(loader), desc=description, unit="batch", dynamic_ncols=True,
        disable=description is None,
    ) as progress:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits,labels)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()*len(labels)
            batch_labels = labels.tolist()
            batch_predictions = logits.argmax(1).tolist()
            actual.extend(batch_labels)
            predicted.extend(batch_predictions)
            correct += sum(a == b for a, b in zip(batch_labels, batch_predictions))
            progress.set_postfix(loss=f"{total/len(actual):.4f}",
                                 accuracy=f"{correct/len(actual):.1%}", refresh=False)
            progress.update()
    return total/len(actual),float(np.mean(np.equal(actual,predicted))),actual,predicted


def subset(data):
    return pd.concat([g.sample(min(12,len(g)),random_state=42) for _,g in data.groupby("label")])


def train(kind):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=24)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--patience",type=int,default=5)
    for flag in ("resume","smoke","pretrained","class-weights"):
        p.add_argument("--"+flag,action="store_true")
    args = p.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_csv(RESULTS/"fer_manifest.csv")
    training, validation = data[data.split=="train"], data[data.split=="val"]
    prefix = kind
    if args.smoke:
        training,validation = subset(training),subset(validation)
        args.epochs = 1
        prefix += "_smoke"
    loaders = [DataLoader(Faces(rows,kind,aug),batch_size=args.batch_size,shuffle=aug)
               for rows,aug in ((training,True),(validation,False))]
    model = network(kind,args.pretrained and not args.resume).to(device)
    weights = None
    if args.class_weights:
        counts = training.label.value_counts().reindex(CLASSES)
        weights = torch.tensor((1/counts).values,dtype=torch.float32,device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)
    best, stale, history = float("inf"),0,[]
    best_path = ROOT/f"models/fer2013_{prefix}_best.pth"
    latest = RESULTS/f"{prefix}_resume.pth"
    if args.resume:
        saved = torch.load(latest,map_location="cpu")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best,stale,history = saved["best"],saved["stale"],saved["history"]
        torch.set_rng_state(saved["rng"])
        if device=="cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    (RESULTS/f"{prefix}_config.json").write_text(json.dumps(dict(vars(args),device=device,seed=42),indent=2))
    hardware = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"{kind.upper()} on {hardware} | {len(training):,} train / {len(validation):,} validation"
          f" | batch size {args.batch_size} | {len(history)}/{args.epochs} epochs completed", flush=True)
    for step in range(len(history)+1,args.epochs+1):
        label = f"Epoch {step}/{args.epochs}"
        loss,acc,*_ = epoch(model,loaders[0],criterion,device,optimizer,label+" train")
        val_loss,val_acc,*_ = epoch(model,loaders[1],criterion,device,description=label+" validation")
        history.append(dict(epoch=step,loss=loss,accuracy=acc,val_loss=val_loss,val_accuracy=val_acc))
        scheduler.step(val_loss)
        if val_loss < best:
            best,stale = val_loss,0
            torch.save(model.state_dict(),best_path.with_suffix(".tmp"))
            best_path.with_suffix(".tmp").replace(best_path)
        else:
            stale += 1
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),history=history,stale=stale,
                        best=best,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()
                        if device=="cuda" else []),latest.with_suffix(".tmp"))
        latest.with_suffix(".tmp").replace(latest)
        pd.DataFrame(history).to_csv(RESULTS/f"{prefix}_training_history.csv",index=False)
        print(f"{label} saved | train loss {loss:.4f}, accuracy {acc:.1%}"
              f" | validation loss {val_loss:.4f}, accuracy {val_acc:.1%}"
              f" | best validation loss {best:.4f}",flush=True)
        if stale >= args.patience:
            print(f"Early stopping after {stale} epochs without lower validation loss.", flush=True)
            break
    for metric in ("accuracy","loss"):
        df = pd.DataFrame(history).set_index("epoch")
        ax = df[[metric,"val_"+metric]].plot(marker="o",title=f"{prefix} training and validation")
        ax.set_ylabel(metric.title())
        plt.tight_layout()
        plt.savefig(FIGURES/f"{prefix}_{metric}.png",dpi=220)
        plt.close()


def evaluate(kind):
    p = argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    args = p.parse_args()
    prefix = kind + ("_smoke" if args.smoke else "")
    path = ROOT/f"models/fer2013_{prefix}_best.pth"
    model = network(kind)
    model.load_state_dict(torch.load(path,map_location="cpu"))
    torch.set_num_threads(4)
    data = pd.read_csv(RESULTS/"fer_manifest.csv")
    # Development smoke checks never touch the test split.
    rows = subset(data[data.split=="val"]) if args.smoke else data[data.split=="test"]
    loader = DataLoader(Faces(rows,kind),batch_size=64)
    loss,acc,y,pred = epoch(model,loader,nn.CrossEntropyLoss(),"cpu")
    report = classification_report(y,pred,labels=list(range(7)),target_names=CLASSES,
                                   zero_division=0,output_dict=True)
    cm = confusion_matrix(y,pred,labels=list(range(7)))
    info = dict(split="validation smoke subset" if args.smoke else "test",samples=len(y),
                accuracy=acc,balanced_accuracy=balanced_accuracy_score(y,pred),loss=loss,
                macro_f1=report["macro avg"]["f1-score"],micro_f1=acc,classes=report,
                parameters=sum(p.numel() for p in model.parameters()),file_bytes=path.stat().st_size,
                confusion_matrix=cm.tolist())
    (RESULTS/f"{prefix}_evaluation.json").write_text(json.dumps(info,indent=2))
    fig,ax = plt.subplots(figsize=(7,6))
    ax.imshow(cm,cmap="Blues")
    for i,j in np.ndindex(cm.shape):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set(xticks=range(7),yticks=range(7),xticklabels=CLASSES,yticklabels=CLASSES,
           xlabel="Predicted category",ylabel="Dataset category",title=f"{prefix} | {info['split']}")
    fig.tight_layout()
    fig.savefig(FIGURES/f"{prefix}_confusion.png",dpi=220)
    plt.close(fig)
