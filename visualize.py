import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import AutoModelForImageTextToText, AutoImageProcessor  # <-- используем AutoImageProcessor
from peft import PeftModel
from sae_model import SparseAutoencoder

# ---------- Конфигурация ----------
HF_TOKEN = ""
MODEL_ID = "google/medgemma-4b-it"
ADAPTER_PATH = "./medgemma-4b-it-sft-lora-crc100k"
FEATURES_PATH = "./features/features.npy"
LABELS_PATH = "./features/labels.npy"
CLASS_NAMES_PATH = "./features/class_names.npy"
SAE_PATH = "./models/sae.pth"
OUTPUT_DIR = "./visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Загружаем модель Med‑Gemma и процессор
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, token=HF_TOKEN, ensure_weight_tying=True)
model.eval()
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=False )

# Получаем vision tower
vision_encoder = model.base_model.model.model.vision_tower

# 2. Загружаем обученный SAE
features_np = np.load(FEATURES_PATH)
D = features_np.shape[1]          # 1152 для vision или 2560 для проектора
HIDDEN_DIM = 256                   # должно совпадать с train_sae.py
sae = SparseAutoencoder(D, HIDDEN_DIM).to(device)
sae.load_state_dict(torch.load(SAE_PATH, map_location=device))
sae.eval()

# 3. Загружаем метки и имена классов (для анализа)
labels = np.load(LABELS_PATH)
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
unique_labels = np.unique(labels)

# ================================================================
# A. Средние активации латентов по классам (аналог Figure 1a)
# ================================================================
@torch.no_grad()
def compute_class_means():
    X = torch.tensor(features_np, dtype=torch.float32).to(device)
    Z = sae.encoder(X).cpu().numpy()
    class_means = {}
    for lb in unique_labels:
        mask = labels == lb
        class_means[lb] = Z[mask].mean(axis=0)
    return class_means

class_means = compute_class_means()
means_matrix = np.array([class_means[lb] for lb in unique_labels])  # [num_classes, HIDDEN_DIM]

plt.figure(figsize=(12, 6))
plt.imshow(np.abs(means_matrix), aspect='auto', cmap='hot')
plt.colorbar(label='|mean activation|')
plt.xlabel('Latent index')
plt.ylabel('Class index')
plt.title('Class-conditional mean latent activations')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_latent_means.png")
plt.close()

# ================================================================
# B. Grad‑FAM для конкретного латента на произвольном изображении
# ================================================================

def grad_fam(image: Image.Image, target_latent_idx: int):
    """
    Возвращает heatmap (numpy массив [H, W]) для заданного латента.
    image: PIL RGB (должен соответствовать ожиданиям SigLIP – размер преобразуется процессором)
    target_latent_idx: индекс латента, для которого строим карту.
    """
    # Подготовка входных тензоров
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    pixel_values = inputs["pixel_values"]  # [1, 3, H, W] – для SigLIP обычно 384x384

    # Нам нужно получить активации последнего слоя vision_encoder и градиенты по ним.
    # Выберем слой, с которого будем брать feature maps – последний блок энкодера.
    target_layer = vision_encoder.vision_model.encoder.layers[-1]  # SiglipEncoderLayer

    # Регистрируем хуки
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        # output – BaseModelOutput, берём last_hidden_state
        activations = output[0]  # [1, num_patches, hidden_dim]

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]  # градиент по last_hidden_state

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Прямой проход до SAE латента с градиентами
    # Мы должны пропустить pixel_values через vision_encoder, взять средний пул,
    # затем через SAE энкодер и получить значение нужного латента.
    # Включаем градиенты на pixel_values? Нет, градиенты нам нужны по активациям,
    # поэтому мы можем спокойно выключить градиенты у пикселей, а граф построить от активаций.
    # Будем использовать torch.autograd.grad вручную.

    # 1. Запускаем vision_encoder с графом, чтобы активации были вычислены в torch.enable_grad()
    with torch.enable_grad():
        # Отключаем градиенты для параметров модели, чтобы не засорять
        for param in model.parameters():
            param.requires_grad = False

        # Прогоняем через vision_encoder (он не требует градиентов, но мы можем включить граф, подставив вход с requires_grad=True)
        # Создадим копию pixel_values с requires_grad=True – но это лишнее.
        # Более чисто: временно превратим pixel_values в тензор с grad, или используем .detach()
        # Но чтобы градиент шёл от латента к активациям, нужно, чтобы на пути был граф.
        # Мы можем вручную выполнить:
        #   vision_out = vision_encoder(pixel_values)   # здесь pixel_values не требует градиентов -> графа не будет.
        # Тогда мы применим хитрость: создадим тензор-заглушку с графом, который зависит от vision_out.
        # Удобный трюк: обернуть выход vision_encoder в torch.autograd.Variable с retain_grad=False,
        # но лучше сделать так:

        # Включим граф, потребовав градиент у выхода vision_encoder путём создания переменной.
        # Используем torch.tensor(..., requires_grad=True) – но это не свяжет с параметрами.
        # Правильный путь: вызвать vision_encoder с входом, у которого requires_grad=True.
        # pixel_values из процессора – без градиентов. Присвоим requires_grad_().
        pixel_values_grad = pixel_values.detach().requires_grad_(True)
        vision_out = vision_encoder(pixel_values_grad).last_hidden_state.float()  # [1, N, D]

        # Средний пул
        pooled = vision_out.mean(dim=1)  # [1, D]

        # SAE энкодер
        latent = sae.encoder(pooled)  # [1, HIDDEN_DIM]

        # Берём целевой латент
        target_score = latent[0, target_latent_idx]

        # Вычисляем градиент target_score по activations (которые мы поймали хуком)
        # Но для этого нам нужно, чтобы activations присутствовали в графе.
        # После прямого прохода activations заполнены.
        # Мы можем вызвать torch.autograd.grad(target_score, activations, retain_graph=False)
        # Это даст градиент d(target_score)/d(activations).
        grads = torch.autograd.grad(target_score, activations, retain_graph=False, allow_unused=False)[0]
        # grads: [1, N, D]

    # Убираем хуки
    handle_fwd.remove()
    handle_bwd.remove()

    # grads – это градиент по картам активаций. Теперь считаем веса каналов: усредняем по пространственной размерности (патчи).
    # Размерность: [1, N_patches, D]. Для ResNet мы усредняли по H*W; здесь N_patches = количество патчей (например, 576 = 24x24).
    # Считаем важность каждого канала (размерность D):
    weights = grads.mean(dim=1)  # [1, D] – средний градиент по патчам

    # Активации тоже [1, N_patches, D]
    activations_detached = activations.detach()  # [1, N_patches, D]

    # Взвешенная сумма: умножаем активации на веса и суммируем по каналам -> [1, N_patches]
    # В статье SALVE: L_l = | sum_k (beta_k * F_k) |
    # beta_k у нас – weights[0, k]; F_k – канал k активаций.
    # Реализация через einsum:
    heatmap_flat = torch.einsum('bnd,bd->bn', activations_detached.float(), weights.float())  # [1, N_patches]

    # Применяем абсолютное значение (как в статье, чтобы учесть и положительное и отрицательное влияние)
    heatmap_flat = heatmap_flat.abs().squeeze(0)  # [N_patches]

    # Восстанавливаем 2D форму патчей. Для SigLIP с размером изображения 384 и патчем 16:
    # patches_per_side = 384 // 16 = 24
    patches_per_side = int(np.sqrt(heatmap_flat.shape[0]))
    heatmap_2d = heatmap_flat.reshape(patches_per_side, patches_per_side)  # [24, 24]

    # Нормализуем для наложения
    heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
    heatmap_np = heatmap_2d.cpu().numpy()

    return heatmap_np

# Пример использования Grad‑FAM:
# Загрузите одно изображение из вашего датасета
from datasets import load_dataset
data_val = load_dataset("./NCT-CRC-HE-100K", split="train").train_test_split(100)["test"]
sample_img = data_val[3]["image"].convert("RGB")
sample_img.save(f"{OUTPUT_DIR}/image.png")
# Выберите латент, доминирующий для класса 8 (I: colorectal...)
dominant_latent = np.argmax(np.abs(class_means[1]))  # предположим, класс 8
hm = grad_fam(sample_img, dominant_latent)
plt.imshow(sample_img.resize((384,384)), alpha=0.7)
plt.imshow(hm, cmap='jet', alpha=0.3)
plt.axis('off')
plt.savefig(f"{OUTPUT_DIR}/gradfam_latent{dominant_latent}.png")
plt.close()

print("Grad‑FAM функция готова. Для теста раскомментируйте код выше.")

# ================================================================
# C. Activation Maximisation (синтез изображения для латента)
# ================================================================

def activation_maximisation(target_latent_idx: int, iterations=300, lr=0.05):
    """
    Синтезирует изображение, максимально активирующее заданный латент, используя градиентный подъём.
    Возвращает PIL.Image.
    """
    # Параметры изображения: размер, ожидаемый SigLIP (ширина/высота зависит от модели, обычно 384)
    img_size = 384  # для Med‑Gemma/SigLIP

    # Инициализируем случайный шум (цветной)
    torch.manual_seed(42)
    param_img = torch.randn(1, 3, img_size, img_size, device=device, dtype=torch.float32)
    param_img.requires_grad = True
    optimizer = torch.optim.Adam([param_img], lr=lr)

    # Средние значения и std для нормализации SigLIP (можно не использовать, если хотим прямое RGB)
    # Но лучше идти через препроцессинг процессора: процессор нормализует изображение.
    # Мы можем использовать трансформации процессора как фиксированный шаг предобработки.
    # Сохраним препроцессинг отдельно: mean, std.
    # Для простоты будем оптимизировать пиксели в диапазоне [0,1] и применять преобразование процессора через torchvision.
    # В Med‑Gemma процессор включает rescaling и нормализацию. Мы применим их внутри цикла.

    # Клонируем процессор, чтобы использовать только image_processor

    for i in tqdm(range(iterations), desc="Activation Maximisation"):
        optimizer.zero_grad()

        # Приводим значения к [0,1] (можно оставить без ограничений, тогда применяем трансформации)
        # Рекомендуется оставить пиксели неограниченными, а процессор их отшкалит.
        # Но для численной стабильности иногда используют сигмоиду или clamp.
        # Оставим как есть.

        # Применяем препроцессинг к текущему изображению
        # img_processor ожидает список PIL или тензор в формате (C,H,W) со значениями [0,1]? 
        # Для SigLIP обычно вход нормализован по mean/std. Мы можем вручную вызвать transform.
        # Проще: подадим param_img в img_processor.preprocess() если он принимает тензор.
        # Проверим метод: 
        # inputs = img_processor(images=param_img, return_tensors="pt") – не сработает, т.к. param_img с градиентами.
        # Сделаем обходной путь: извлечём нормализацию и применим вручную.
        # Достанем mean и std из img_processor.image_mean и image_std:
        mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
        # Нормализуем
        normalized_img = (param_img - mean) / std

        # Прогоняем через vision_encoder
        vision_out = vision_encoder(normalized_img, interpolate_pos_encoding=True).last_hidden_state.float()  # [1, N, D]

        # Глобальный пул
        pooled = vision_out.mean(dim=1)

        # SAE энкодер
        latent_vec = sae.encoder(pooled)  # [1, HIDDEN_DIM]

        # Целевая активация
        act = latent_vec[0, target_latent_idx]

        # Добавляем регуляризации: L2 на пиксели и Total Variation (TV) для гладкости
        l2_reg = 0.01 * (param_img ** 2).mean()
        # TV loss: разница между соседними пикселями
        tv_loss = (torch.abs(param_img[:, :, :-1, :] - param_img[:, :, 1:, :])).mean() + \
                  (torch.abs(param_img[:, :, :, :-1] - param_img[:, :, :, 1:])).mean()
        tv_reg = 1e-3 * tv_loss

        # Минимизируем отрицательную активацию (максимизируем её) плюс регуляризации
        loss = -act + l2_reg + tv_reg
        loss.backward()
        optimizer.step()

    # После оптимизации приводим тензор к изображению
    # Денормализуем обратно для визуализации
    with torch.no_grad():
        viz = (normalized_img * std + mean).clamp(0, 1)  # восстанавливаем RGB
    viz_np = viz.squeeze(0).permute(1, 2, 0).cpu().numpy()
    viz_pil = Image.fromarray((viz_np * 255).astype(np.uint8))
    return viz_pil

# Пример генерации для доминантного латента класса I
# idx = np.argmax(np.abs(class_means[0]))
# generated = activation_maximisation(idx, iterations=500)
# generated.save(f"{OUTPUT_DIR}/actmax_latent{idx}.png")

print("Функция activation_maximisation готова.")
