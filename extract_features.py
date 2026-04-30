import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset

# ---------- НАСТРОЙКИ ----------
MODEL_ID = "medgemma-4b-it-sft-lora-crc100k"   # ваша обученная модель
TARGET_LAYER = "vision"                        # "vision" (SigLIP) или "projector"
SAVE_DIR = "./features"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Загрузка модели и процессора ----------
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()

vision_encoder = model.vision_model

def extract_features(image: Image.Image):
    """Возвращает глобальный вектор признаков для одного изображения."""
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        if TARGET_LAYER == "vision":
            # выход последнего слоя SigLIP: ([B, N_patches, hidden])
            out = vision_encoder(**inputs).last_hidden_state
        elif TARGET_LAYER == "projector":
            hidden = vision_encoder(**inputs).last_hidden_state
            out = model.multi_modal_projector(hidden)
        else:
            raise ValueError("TARGET_LAYER must be 'vision' or 'projector'")
        # глобальный пуллинг (усреднение по пространственным патчам)
        global_vec = out.mean(dim=1)  # [B, D]
    return global_vec.squeeze(0).cpu().numpy()

# ---------- Загрузка датасета ----------
data = load_dataset("./NCT-CRC-HE-100K", split="train")
data = data.train_test_split(train_size=9000, test_size=1000, seed=42)
dataset = data["train"]  # или data["validation"]

features_list = []
labels_list = []
for sample in tqdm(dataset, desc="Extracting"):
    img = sample["image"].convert("RGB")
    feat = extract_features(img)
    features_list.append(feat)
    labels_list.append(sample["label"])

features = np.vstack(features_list)  # [N, D]
labels = np.array(labels_list)

# Сохраняем
np.save(f"{SAVE_DIR}/features.npy", features)
np.save(f"{SAVE_DIR}/labels.npy", labels)

# Классы (скопируйте ваши названия или сохраните отдельно)
class_names = [
    "A: adipose", "B: background", "C: debris", "D: lymphocytes",
    "E: mucus", "F: smooth muscle", "G: normal colon mucosa",
    "H: cancer-associated stroma", "I: colorectal adenocarcinoma epithelium"
]
np.save(f"{SAVE_DIR}/class_names.npy", class_names)

print(f"Фичи сохранены: {features.shape}, метки: {labels.shape}")
