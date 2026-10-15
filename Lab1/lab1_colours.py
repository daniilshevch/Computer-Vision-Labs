import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def compute_histogram_channel(channel: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for val in channel.ravel():
        hist[val] += 1
    return hist


def shift_rgb_direct(image_rgb: np.ndarray, shift_value: int) -> np.ndarray:
    shifted_float = image_rgb.astype(np.float32) - shift_value
    clipped_float = np.clip(shifted_float, a_min=0, a_max=255)
    return clipped_float.astype(np.uint8)


def shift_hsv_v(image_pil: Image.Image, shift_value: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_hsv = np.array(image_pil.convert('HSV'), dtype=np.float32)
    v_orig = img_hsv[:, :, 2].astype(np.uint8).copy()

    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] - shift_value, a_min=0, a_max=255)
    v_shifted = img_hsv[:, :, 2].astype(np.uint8)

    img_rgb_shifted = np.array(Image.fromarray(img_hsv.astype(np.uint8), mode='HSV').convert('RGB'))
    return img_rgb_shifted, v_orig, v_shifted


def process_and_display_combined(image_path: str, title: str, shift_delta: int = 50) -> None:
    img_pil = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img_pil)

    img_rgb_shifted = shift_rgb_direct(img_rgb, shift_delta)
    img_hsv_shifted, v_orig, v_shifted = shift_hsv_v(img_pil, shift_delta)

    hist_r_orig = compute_histogram_channel(img_rgb[:, :, 0])
    hist_g_orig = compute_histogram_channel(img_rgb[:, :, 1])
    hist_b_orig = compute_histogram_channel(img_rgb[:, :, 2])

    hist_r_shift = compute_histogram_channel(img_rgb_shifted[:, :, 0])
    hist_g_shift = compute_histogram_channel(img_rgb_shifted[:, :, 1])
    hist_b_shift = compute_histogram_channel(img_rgb_shifted[:, :, 2])

    hist_v_orig = compute_histogram_channel(v_orig)
    hist_v_shift = compute_histogram_channel(v_shifted)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=120)
    fig.suptitle(f"{title} (Δ = {shift_delta})", fontsize=16, fontweight='bold', y=0.98)

    all_rgb_vals = np.concatenate([
        hist_r_orig, hist_g_orig, hist_b_orig,
        hist_r_shift, hist_g_shift, hist_b_shift
    ])
    y_max_rgb = np.percentile(all_rgb_vals[all_rgb_vals > 0], 98.5) * 1.25

    all_v_vals = np.concatenate([hist_v_orig, hist_v_shift])
    y_max_v = np.percentile(all_v_vals[all_v_vals > 0], 98.5) * 1.25

    axes[1, 0].set_ylim([0, y_max_rgb])
    axes[1, 1].set_ylim([0, y_max_rgb])
    axes[1, 2].set_ylim([0, y_max_v])

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Оригінальне RGB", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_rgb_shifted)
    axes[0, 1].set_title(f"2. RGB Поканальний зсув (-{shift_delta})", fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_hsv_shifted)
    axes[0, 2].set_title(f"3. HSV Зсув V-каналу (-{shift_delta})", fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].plot(range(256), hist_r_orig, color='red', alpha=0.7, label='R')
    axes[1, 0].plot(range(256), hist_g_orig, color='green', alpha=0.7, label='G')
    axes[1, 0].plot(range(256), hist_b_orig, color='blue', alpha=0.7, label='B')
    axes[1, 0].set_title("Початкові гістограми R, G, B", fontsize=12)
    axes[1, 0].set_xlim([0, 255])
    axes[1, 0].set_xlabel("Інтенсивність", fontsize=11)
    axes[1, 0].set_ylabel("Кількість пікселів", fontsize=11)
    axes[1, 0].legend(fontsize=10)

    axes[1, 1].plot(range(256), hist_r_shift, color='red', alpha=0.7, label='R')
    axes[1, 1].plot(range(256), hist_g_shift, color='green', alpha=0.7, label='G')
    axes[1, 1].plot(range(256), hist_b_shift, color='blue', alpha=0.7, label='B')
    axes[1, 1].set_title("Зсунуті гістограми RGB", fontsize=12)
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].set_xlabel("Інтенсивність", fontsize=11)
    axes[1, 1].legend(fontsize=10)

    axes[1, 2].bar(range(256), hist_v_shift, color='purple', width=1.0, alpha=0.8, label='Shifted V')
    axes[1, 2].set_title("Зсунута гістограма V-каналу", fontsize=12)
    axes[1, 2].set_xlim([0, 255])
    axes[1, 2].set_xlabel("Інтенсивність", fontsize=11)
    axes[1, 2].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def process_color_batch_combined(image_configs: list[tuple[str, str]], shift_deltas: list[int]) -> None:
    for image_path, title in image_configs:
        for shift_delta in shift_deltas:
            process_and_display_combined(image_path, title, shift_delta)


images_data = [
    ("Images/lviv.png", "Тест 1: Висока деталізація"),
    ("Images/sky.jpg", "Тест 2: Низька деталізація"),
    ("Images/lviv_night2.jpg", "Тест 3: Висока контрастність"),
    ("Images/deer.jpg", "Тест 4: Низька контрастність"),

]

deltas = [30, 60, 90, 120]

process_color_batch_combined(images_data, deltas)