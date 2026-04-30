import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sae_model import SparseAutoencoder

# Настройки
FEATURES_PATH = "./features/features.npy"
HIDDEN_DIM = 256
L1_COEF = 1e-3       # очень важный параметр – регулируйте
LR = 1e-3
EPOCHS = 1000
BATCH_SIZE = 32
SAVE_DIR = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = f"{SAVE_DIR}/sae.pth"

# Загрузка
X = np.load(FEATURES_PATH)
X_tensor = torch.tensor(X, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)

D = X.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SparseAutoencoder(D, HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

for epoch in tqdm(range(1, EPOCHS+1), desc="Training SAE"):
    model.train()
    total_loss = 0.0
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        z, x_hat = model(batch_x)
        recon = F.mse_loss(x_hat, batch_x)
        l1 = L1_COEF * z.abs().mean()
        loss = recon + l1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 100 == 0:
        tqdm.write(f"Ep {epoch}: loss {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"SAE сохранён в {SAVE_PATH}")
