import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoImageProcessor  # <-- используем AutoImageProcessor
from peft import PeftModel
from datasets import load_dataset
from safetensors.torch import load_file
HF_TOKEN = ""
MODEL_ID = "google/medgemma-4b-it"
ADAPTER_PATH = "./medgemma-4b-it-sft-lora-crc100k"                # или ваш реальный путь
TARGET_LAYER = "vision"
SAVE_DIR = "./features"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Загрузка модели ----------
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, token=HF_TOKEN, ensure_weight_tying=True)
model.eval()
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=False )

# Получаем vision tower
vision_encoder = model.base_model.model.model.vision_tower

def extract_features(image: Image.Image):
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device)
    with torch.no_grad():
        if TARGET_LAYER == "vision":
            out = vision_encoder(pixel_values=pixel_values)
            features = out.last_hidden_state
        elif TARGET_LAYER == "projector":
            hidden = vision_encoder(pixel_values=pixel_values).last_hidden_state
            features = model.base_model.multi_modal_projector(hidden)
        else:
            raise ValueError("TARGET_LAYER must be 'vision' or 'projector'")
        global_vec = features.mean(dim=1)  # [B, D]
    return global_vec.squeeze(0).cpu().float().numpy()

# ---------- Датасет и извлечение (без изменений) ----------
data = load_dataset("./NCT-CRC-HE-100K", split="train")
data = data.train_test_split(train_size=9000, test_size=1000, seed=42)
dataset = data["train"]

features_list, labels_list = [], []
for sample in tqdm(dataset, desc="Extracting"):
    img = sample["image"].convert("RGB")
    feat = extract_features(img)
    features_list.append(feat)
    labels_list.append(sample["label"])

features = np.vstack(features_list)
labels = np.array(labels_list)

np.save(f"{SAVE_DIR}/features.npy", features)
np.save(f"{SAVE_DIR}/labels.npy", labels)

class_names = [
    "A: adipose", "B: background", "C: debris", "D: lymphocytes",
    "E: mucus", "F: smooth muscle", "G: normal colon mucosa",
    "H: cancer-associated stroma", "I: colorectal adenocarcinoma epithelium"
]
np.save(f"{SAVE_DIR}/class_names.npy", class_names)

print(f"Фичи сохранены: {features.shape}, метки: {labels.shape}")
