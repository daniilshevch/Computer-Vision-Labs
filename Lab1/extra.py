import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def apply_erosion(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    pad = ksize // 2
    if img.ndim == 2:
        padded = np.pad(img, pad, mode='reflect')
        h, w = img.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.min(padded[i:i + ksize, j:j + ksize])
        return out
    else:
        h, w, c = img.shape
        out = np.zeros((h, w, c), dtype=np.uint8)
        for channel in range(c):
            out[:, :, channel] = apply_erosion(img[:, :, channel], ksize=ksize)
        return out


def apply_dilation(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    pad = ksize // 2
    if img.ndim == 2:
        padded = np.pad(img, pad, mode='reflect')
        h, w = img.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.max(padded[i:i + ksize, j:j + ksize])
        return out
    else:
        h, w, c = img.shape
        out = np.zeros((h, w, c), dtype=np.uint8)
        for ch in range(c):
            out[:, :, ch] = apply_dilation(img[:, :, ch], ksize=ksize)
        return out


def apply_opening(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    eroded = apply_erosion(img, ksize=ksize)
    return apply_dilation(eroded, ksize=ksize)


def apply_closing(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    dilated = apply_dilation(img, ksize=ksize)
    return apply_erosion(dilated, ksize=ksize)


def apply_morphological_gradient(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    dilated = apply_dilation(img, ksize=ksize).astype(np.float32)
    eroded = apply_erosion(img, ksize=ksize).astype(np.float32)
    return np.clip(dilated - eroded, 0, 255).astype(np.uint8)


def apply_morphology_hsv_v(img_pil: Image.Image, op_func, ksize: int) -> np.ndarray:
    img_hsv = np.array(img_pil.convert('HSV'))
    h_channel = img_hsv[:, :, 0]
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]

    v_transformed = op_func(v_channel, ksize=ksize)

    hsv_reconstructed = np.stack([h_channel, s_channel, v_transformed], axis=-1)
    return np.array(Image.fromarray(hsv_reconstructed, mode='HSV').convert('RGB'))


def process_and_display_all_types(
        image_path: str,
        ksize: int = 3,
        threshold_val: int = 120
) -> None:
    img_pil = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img_pil)

    img_pil_gray = img_pil.convert('L')
    img_gray = np.array(img_pil_gray)

    img_bin = np.where(img_gray > threshold_val, 255, 0).astype(np.uint8)

    modes = [
        ("Binary", img_bin, True),
        ("Grayscale", img_gray, True),
        ("RGB", img_rgb, False)
    ]

    titles = ["Оригінал", "Ерозія", "Дилатація", "Розкриття", "Закриття", "Морфологічний градієнт"]

    for mode_name, img_data, is_gray in modes:
        res_eroded = apply_erosion(img_data, ksize=ksize)
        res_dilated = apply_dilation(img_data, ksize=ksize)
        res_opened = apply_opening(img_data, ksize=ksize)
        res_closed = apply_closing(img_data, ksize=ksize)
        res_gradient = apply_morphological_gradient(img_data, ksize=ksize)

        images = [img_data, res_eroded, res_dilated, res_opened, res_closed, res_gradient]

        for curr_img, title_text in zip(images, titles):
            fig, ax = plt.subplots(figsize=(6, 6))
            if is_gray:
                ax.imshow(curr_img, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(curr_img)

            ax.set_title(f"{mode_name} — {title_text} (Ядро {ksize}x{ksize})", fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.show()

    hsv_eroded = apply_morphology_hsv_v(img_pil, apply_erosion, ksize=ksize)
    hsv_dilated = apply_morphology_hsv_v(img_pil, apply_dilation, ksize=ksize)
    hsv_opened = apply_morphology_hsv_v(img_pil, apply_opening, ksize=ksize)
    hsv_closed = apply_morphology_hsv_v(img_pil, apply_closing, ksize=ksize)
    hsv_gradient = apply_morphology_hsv_v(img_pil, apply_morphological_gradient, ksize=ksize)

    hsv_images = [img_rgb, hsv_eroded, hsv_dilated, hsv_opened, hsv_closed, hsv_gradient]

    for curr_img, title_text in zip(hsv_images, titles):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(curr_img)
        ax.set_title(f"HSV (Канал V) — {title_text} (Ядро {ksize}x{ksize})", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

selected_image = "Images/lviv.png"
process_and_display_all_types(selected_image, ksize=3, threshold_val=120)