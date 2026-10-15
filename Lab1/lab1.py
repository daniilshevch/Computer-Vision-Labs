import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def shift_histogram_left(image: np.ndarray, shift_value: int) -> np.ndarray:
    img_float = image.astype(np.float32)
    shifted_float = img_float - shift_value
    clipped_float = np.clip(shifted_float, a_min=0, a_max=255)
    return clipped_float.astype(np.uint8)


def compute_histogram_grayscale(image_gray: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for val in image_gray.ravel():
        hist[val] += 1
    return hist


def process_and_display(image_path: str, title: str, shift_delta: int = 50) -> None:
    img_pil = Image.open(image_path)
    img_gray = np.array(img_pil.convert('L'))
    img_shifted = shift_histogram_left(img_gray, shift_value=shift_delta)

    hist_orig = compute_histogram_grayscale(img_gray)
    hist_shifted = compute_histogram_grayscale(img_shifted)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{title} (Зсув Δ = {shift_delta})", fontsize=14, fontweight='bold')

    y_max = hist_orig.max() * 1.1
    axes[1, 0].set_ylim([0, y_max])
    axes[1, 1].set_ylim([0, y_max])

    axes[0, 0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Оригінальне зображення")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_shifted, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f"Результат (Зсув ліворуч -{shift_delta})")
    axes[0, 1].axis('off')

    axes[1, 0].bar(range(256), hist_orig, color='gray', width=1.0)
    axes[1, 0].set_title("Початкова гістограма")
    axes[1, 0].set_xlim([0, 255])
    axes[1, 0].set_xlabel("Інтенсивність")
    axes[1, 0].set_ylabel("Кількість пікселів")

    axes[1, 1].bar(range(256), hist_shifted, color='black', width=1.0)
    axes[1, 1].set_title("Зсунута гістограма")
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].set_xlabel("Інтенсивність")

    plt.tight_layout()
    plt.show()


def process_batch(image_configs: list[tuple[str, str]], shift_deltas: list[int]) -> None:
    for image_path, title in image_configs:
        for shift_delta in shift_deltas:
            process_and_display(image_path, title, shift_delta)


images_data = [
    ("Images/lviv.png", "Тест 1: Висока деталізація"),
    ("Images/sky.jpg", "Тест 2: Низька деталізація"),
    ("Images/lviv_night2.jpg", "Тест 3: Висока контрастність"),
    ("Images/deer.jpg", "Тест 4: Низька контрастність"),
]

deltas = [30, 60, 90, 120, 150]
process_batch(images_data, deltas)