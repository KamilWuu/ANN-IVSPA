import os
import time
import numpy as np
from datetime import datetime
from itertools import product
import cpuinfo

from skimage.io import imread
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# --- Data loading ---
def get_categories(file_location):
    try:
        with open(file_location, 'r') as f:
            return [line.split()[0] for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: {file_location} not found.")
        return []

def load_data(path, categories, cnn=False):
    X, y = [], []
    for c in categories:
        for file in os.listdir(os.path.join(path, c)):
            if file.endswith(".ppm"):
                img = imread(os.path.join(path, c, file))
                img = resize(img, (15,15,3))
                if cnn:
                    X.append(np.transpose(img, (2,0,1)))
                else:
                    X.append(img.flatten())
                y.append(categories.index(c))
    return np.array(X), np.array(y)

# --- Paths & datasets ---
datadir = '../ANN_data/100x/'
cats = get_categories(os.path.join(datadir, "description.txt"))

def load_all(base):
    X, y = load_data(os.path.join(datadir, base), cats, cnn=False)
    Xc, _ = load_data(os.path.join(datadir, base), cats, cnn=True)
    return torch.tensor(X).float(), torch.tensor(y).long(), torch.tensor(Xc).float()

xtr, ytr, xtr_c = load_all("Training")
xv, yv, xv_c = load_all("Validation")
xt, yt, xt_c = load_all("Test")

Dtr = DataLoader(TensorDataset(xtr, ytr), 32, shuffle=True)
Dv = DataLoader(TensorDataset(xv, yv), 32)
Dt = DataLoader(TensorDataset(xt, yt), 32)

Dtr_c = DataLoader(TensorDataset(xtr_c, ytr), 32, shuffle=True)
Dv_c = DataLoader(TensorDataset(xv_c, yv), 32)
Dt_c = DataLoader(TensorDataset(xt_c, yt), 32)

# --- Models ---
class TrafficSignNet(nn.Module):
    def __init__(self, h1=128,h2=64):
        super().__init__()
        self.fc1 = nn.Linear(15*15*3,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,len(cats))
    def forward(self,x):
        x=x.view(-1,15*15*3)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

class TrafficSignCNN(nn.Module):
    def __init__(self, c1=32,c2=64,fc=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3,c1,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(c1,c2,3,padding=1)
        self.fc1 = nn.Linear(c2*3*3,fc)
        self.fc2 = nn.Linear(fc,len(cats))
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=x.view(-1,x.size(1)*x.size(2)*x.size(3))
        return self.fc2(torch.relu(self.fc1(x)))

# --- Training & evaluation ---
def train_model(model, loader, val_loader, lr, epochs=15, patience=3):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_val=0; stagnant=0
    history=[]
    for ep in range(epochs):
        model.train()
        tot_loss=0
        for Xb,yb in loader:
            opt.zero_grad()
            loss=crit(model(Xb), yb)
            loss.backward(); opt.step()
            tot_loss+=loss.item()*Xb.size(0)
        val_acc=0
        if val_loader:
            model.eval()
            correct=0; total=0
            for Xb,yb in val_loader:
                pred = model(Xb).argmax(dim=1)
                correct+= (pred==yb).sum().item(); total+=len(yb)
            val_acc = correct/total
            if val_acc>best_val: best_val,stagnant=val_acc,0
            else: stagnant+=1
        history.append((tot_loss/len(loader.dataset), val_acc))
        print(f"Epoch {ep+1}, loss {history[-1][0]:.3f}, val_acc {val_acc:.3f}")
        if stagnant>=patience:
            print(f" Early stopping at ep {ep+1}")
            break
    return history

def test_model(model, loader):
    model.eval()
    preds, gts = [],[]
    with torch.no_grad():
        for Xb,yb in loader:
            preds += model(Xb).argmax(dim=1).tolist()
            gts += yb.tolist()
    acc = sum(p==g for p,g in zip(preds,gts))/len(gts)
    return acc, preds, gts

# --- Grid search ---
def get_combos(grid):
    return [dict(zip(grid, vals)) for vals in product(*grid.values())]

ann_grid={"lr":[1e-3,5e-4],"h1":[128,256],"h2":[64,128]}
cnn_grid={"lr":[1e-3,5e-4],"c1":[32,64],"c2":[64,128],"fc":[128,256]}

best_ann={'acc':0}
ann_records=[]
for p in get_combos(ann_grid):
    print("\nANN params:",p)
    m=TrafficSignNet(p['h1'],p['h2'])
    history = train_model(m, Dtr, Dv, p['lr'])
    acc, _, _ = test_model(m, Dt)
    print("Test acc %.3f"%acc)
    p['acc']=acc; ann_records.append((p,history))
    if acc>best_ann['acc']:
        best_ann={
            'params':p,
            'model':m,
            'hist':history,
            'preds':None,
            'gts':None,
            'acc': acc
            }

best_ann['acc'], best_ann

best_cnn={'acc':0}
cnn_records=[]
for p in get_combos(cnn_grid):
    print("\nCNN params:",p)
    m=TrafficSignCNN(p['c1'],p['c2'],p['fc'])
    history = train_model(m, Dtr_c, Dv_c, p['lr'])
    acc, _, _ = test_model(m, Dt_c)
    print("Test acc %.3f"%acc)
    p['acc']=acc; cnn_records.append((p,history))
    if acc>best_cnn['acc']:
        best_cnn={
            'params':p,
            'model':m,
            'hist':history,
            'preds':None,
            'gts':None,
            'acc': acc
            }

best_cnn['acc'], best_cnn

# --- Reporting ---
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
rep_dir = os.path.join("reports", now)
os.makedirs(rep_dir, exist_ok=True)

# Write CSV
csv_path = os.path.join(rep_dir, "results.csv")
with open(csv_path,"w") as f:
    f.write(f"Processor used to perform tests: {cpuinfo.get_cpu_info()['brand_raw']}\n")
    f.write("model,parameters,accuracy\n")
    for rec in ann_records:
        f.write(f"ANN,\"{rec[0]}\",{rec[0]['acc']:.4f}\n")
    for rec in cnn_records:
        f.write(f"CNN,\"{rec[0]}\",{rec[0]['acc']:.4f}\n")

# Plotting
def plot_history(hist, title, outpath):
    xs = [i+1 for i in range(len(hist))]
    loss = [h[0] for h in hist]
    valloss = [h[1] for h in hist]
    plt.figure()
    plt.plot(xs, loss, label="Train Loss")
    plt.plot(xs, valloss, label="Val Acc")
    plt.xlabel("Epoch"); plt.legend()
    plt.title(title)
    plt.savefig(outpath); plt.close()

plot_history(best_ann['hist'], "ANN learning curve", os.path.join(rep_dir,"ann_learning_curve.png"))
plot_history(best_cnn['hist'], "CNN learning curve", os.path.join(rep_dir,"cnn_learning_curve.png"))

# Confusion matrices
for key,best in [('ann', best_ann), ('cnn', best_cnn)]:
    acc, preds, gts = test_model(best['model'], Dt if key=='ann' else Dt_c)
    disp = ConfusionMatrixDisplay.from_predictions(gts, preds, display_labels=cats, cmap="Greens")
    disp.ax_.set_title("ANN Confusion Matrix (acc=...)")
    plt.tight_layout()
    plt.savefig(os.path.join(rep_dir,f"{key}_confusion_matrix.png"))
    plt.close()

print("Report saved in", rep_dir)
