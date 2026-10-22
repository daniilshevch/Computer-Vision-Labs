import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def load_or_create_image(image_path: str = None) -> np.ndarray:
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, (500, 500))
    else:
        return np.full((500, 500), 200, dtype=np.uint8)

def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def transform_image(img: np.ndarray, angle_degree: float, dx: float, dy: float, contrast_alpha: float = 1.3, scale = 1.0):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M_rotated = cv2.getRotationMatrix2D(center, angle_degree, scale) #scale = 1.0
    M_rotated[0, 2] += dx
    M_rotated[1, 2] += dy

    transformed_img = cv2.warpAffine(img, M_rotated, (w, h), borderValue=(255,255,255))
    transformed_img = cv2.convertScaleAbs(transformed_img, alpha=contrast_alpha, beta=-20)

    return transformed_img, M_rotated


def custom_binary_matcher(desc1: np.ndarray, desc2: np.ndarray, cross_check: bool = True, ratio_thresh: float = 0.8):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    xor_matrix = np.bitwise_xor(desc1[:, np.newaxis, :], desc2[np.newaxis, :, :])
    dist_matrix = np.unpackbits(xor_matrix, axis=-1).sum(axis=-1) #матриця N x M, де N - кількість дескрипторів у desc1, M - кількість дескрипторів у desc2

    matches = []

    if cross_check:
        best_in_desc2 = np.argmin(dist_matrix, axis=1) #по рядках
        best_in_desc1 = np.argmin(dist_matrix, axis=0) #по стопцях

        for first_descriptor_index, second_descriptor_index in enumerate(best_in_desc2):
            if best_in_desc1[second_descriptor_index] == first_descriptor_index:
                dmatch = cv2.DMatch(_queryIdx=first_descriptor_index, _trainIdx=second_descriptor_index, _imgIdx=0, _distance=float(dist_matrix[first_descriptor_index, second_descriptor_index]))
                matches.append(dmatch)
    else: #тест Девіда Лоу
        for first_descriptor_index in range(dist_matrix.shape[0]):
            sorted_indices = np.argsort(dist_matrix[first_descriptor_index])
            d1 = dist_matrix[first_descriptor_index, sorted_indices[0]]
            d2 = dist_matrix[first_descriptor_index, sorted_indices[1]]

            if d1 < ratio_thresh * d2:
                dmatch = cv2.DMatch(_queryIdx=first_descriptor_index, _trainIdx=sorted_indices[0], _imgIdx=0, _distance=float(d1))
                matches.append(dmatch)

    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def custom_ransac_affine(points1: np.ndarray, points2: np.ndarray, max_iters: int = 2000, inlier_threshold: float = 3.0):
    points_amount = len(points1)
    if points_amount < 3:
        return None, np.zeros(points_amount, dtype=bool)

    best_inliers = []
    best_M = None

    points1_homo = np.hstack([points1, np.ones((points_amount, 1))])

    for _ in range(max_iters):
        indices = np.random.choice(points_amount, 3, replace=False)
        source_sample_homo = points1_homo[indices]
        destination_sample = points2[indices]

        try:
            M_T = np.linalg.solve(source_sample_homo, destination_sample)
            M = M_T.T
            #M = cv2.getAffineTransform(source_sample.astype(np.float32), destination_sample.astype(np.float32))
        except np.linalg.LinAlgError:
            continue

        projected = (M @ points1_homo.T).T

        errors = np.linalg.norm(points2 - projected, axis=1)
        inliers = np.where(errors < inlier_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_M = M

    if len(best_inliers) >= 3:
        points1_inliers_homo = points1_homo[best_inliers].astype(np.float32)
        points2_inliers = points2[best_inliers].astype(np.float32)

        M_T, _, _, _ = np.linalg.lstsq(points1_inliers_homo, points2_inliers, rcond=None)
        best_M = M_T.T
        #best_M, _ = cv2.estimateAffinePartial2D(points1_inliers, points2_inliers)

    inlier_mask = np.zeros(points_amount, dtype=bool)
    inlier_mask[best_inliers] = True

    return best_M, inlier_mask


def extract_features(img: np.ndarray, threshold: int = 25, descriptor_bytes: int = 32):
    img = _to_gray(img)
    detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=descriptor_bytes)

    keypoints = detector.detect(img, None)
    keypoints, descriptors = brief.compute(img, keypoints)

    return keypoints, descriptors


def extract_features_orb(img: np.ndarray, nfeatures: int = 2000):
    img = _to_gray(img)
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        firstLevel=0
    )
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def benchmark_matchers(des1: np.ndarray, des2: np.ndarray):
    t0 = time.time()
    custom_matches = custom_binary_matcher(des1, des2, cross_check=True)
    t_custom = time.time() - t0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    t0 = time.time()
    opencv_matches = bf.match(des1, des2)
    t_opencv = time.time() - t0

    return custom_matches, opencv_matches, t_custom, t_opencv

def plot_matches(image1: np.ndarray, key_points1: list, image2: np.ndarray, key_points2: list,
                 matches: list, inlier_mask: np.ndarray, angle: float,
                 dx: float, dy: float, max_draw: int = 30):
    inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i]]
    img_matches = cv2.drawMatches(
        image1, key_points1, image2, key_points2, inlier_matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if len(img_matches.shape) == 3:
        img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 7))
    plt.imshow(img_matches, cmap='gray')
    plt.title(
        f"Топові збіги BRIEF (Кастомний Matcher + Custom RANSAC)\nОцінений кут: {angle:.1f}°, dx: {dx:.1f}px, dy: {dy:.1f}px"
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def decompose_affine_transform(M: np.ndarray, img_shape: tuple):
    h, w = img_shape[:2]
    cx, cy = w // 2, h // 2

    a = M[0, 0]
    b = M[0, 1]

    est_angle_rad = np.arctan2(b, a)
    est_angle_deg = float(np.degrees(est_angle_rad))

    est_dx = float(M[0, 2] - (1.0 - a) * cx + b * cy)
    est_dy = float(M[1, 2] - b * cx - (1.0 - a) * cy)

    return est_angle_deg, est_dx, est_dy

def decompose_affine_with_scale(M: np.ndarray, img_shape: tuple):
    ##додаткова функція, якщо враховується коефіцієнт масштабування
    h, w = img_shape[:2]
    cx, cy = w // 2, h // 2

    a = M[0, 0]
    b = M[0, 1]

    est_scale = float(np.hypot(a, b))
    est_angle_deg = float(np.degrees(np.arctan2(b, a)))
    est_dx = float(M[0, 2] - (1.0 - a) * cx + b * cy)
    est_dy = float(M[1, 2] - b * cx - (1.0 - a) * cy)

    return est_scale, est_angle_deg, est_dx, est_dy

def run_experiment(
    image_path: str = None,
    true_angle: float = 25.0,
    true_dx: float = 35.0,
    true_dy: float = -20.0,
    contrast_alpha: float = 1.4,
    scale_factor: float = 1.0,
    experiment_name: str = "Експеримент",
    descriptor: str = "BRIEF"
):
    print(f"\n{'='*20} {experiment_name} {'='*20}")
    print(f"Задані параметри: Кут = {true_angle}°, dx = {true_dx}px, dy = {true_dy}px")

    img1 = load_or_create_image(image_path)
    img2, _ = transform_image(
        img1,
        angle_degree=true_angle,
        dx=true_dx,
        dy=true_dy,
        contrast_alpha=contrast_alpha,
        scale=scale_factor
    )

    if descriptor == "BRIEF":
        kp1, des1 = extract_features(img1, threshold=25)
        kp2, des2 = extract_features(img2, threshold=25)
    else:
        kp1, des1 = extract_features_orb(img1, nfeatures=2000)
        kp2, des2 = extract_features_orb(img2, nfeatures=2000)
    print(f"Знайдено точок: Img1 = {len(kp1)}, Img2 = {len(kp2)}")

    if des1 is None or des2 is None:
        print("Не вдалося обчислити дескриптори.")
        return

    custom_matches, opencv_matches, t_custom, t_opencv = benchmark_matchers(des1, des2)
    print(f"Власний матчінг:  {len(custom_matches)} збігів за {t_custom * 1000:.2f} мс")
    print(f"OpenCV BFMatcher: {len(opencv_matches)} збігів за {t_opencv * 1000:.2f} мс")

    points1 = np.float32([kp1[m.queryIdx].pt for m in custom_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in custom_matches])

    M_est, inlier_mask = custom_ransac_affine(points1, points2, max_iters=2000, inlier_threshold=3.0)

    if M_est is not None:
        if scale_factor == 1.0:
            est_angle_deg, est_dx, est_dy = decompose_affine_transform(M_est, img1.shape)
        else:
            est_scale, est_angle_deg, est_dx, est_dy = decompose_affine_with_scale(M_est, img1.shape)

        err_angle = abs(true_angle - est_angle_deg)
        err_dx = abs(true_dx - est_dx)
        err_dy = abs(true_dy - est_dy)
        err_scale = abs(scale_factor - est_scale) if scale_factor != 1.0 else None

        print("\n--- Результати RANSAC ---")
        print(f"Оцінений кут: {est_angle_deg:.2f}° (Помилка: {err_angle:.2f}°)")
        print(f"Оцінений зсув X: {est_dx:.2f}px (Помилка: {err_dx:.2f}px)")
        print(f"Оцінений зсув Y: {est_dy:.2f}px (Помилка: {err_dy:.2f}px)")
        print(f"Інлаєри: {np.sum(inlier_mask)} / {len(custom_matches)}")
        if scale_factor != 1.0:
            print(f"Оцінений масштаб: {est_scale:.4f} (Помилка: {err_scale:.4f})")

        plot_matches(
            img1, kp1, img2, kp2,
            custom_matches, inlier_mask,
            est_angle_deg, est_dx, est_dy
        )
    else:
        print("Не вдалося обчислити трансформацію RANSAC.")

def main():
    run_experiment(
        image_path="Images/ternopil.jpg",
        true_angle=25.0,
        true_dx=35.0,
        true_dy=-20.0,
        experiment_name="Тернопіль, ніч"
    )

    run_experiment(
        image_path="Images/lviv.jpg",
        true_angle=-18.0,
        true_dx = -40.0,
        true_dy = 25.0,
        contrast_alpha = 0.75,
        experiment_name="Львів, день"
    )

    run_experiment(
        image_path="Images/naukova.jpg",
        true_angle= 20.0,
        true_dx = 35.0,
        true_dy = -20.0,
        contrast_alpha = 1.4,
        experiment_name="Наукова, ніч"
    )

    run_experiment(
        image_path="Images/input.jpg",
        true_angle=-22.0,
        true_dx = -20.0,
        true_dy = 30.0,
        contrast_alpha = 0.7,
        experiment_name="Львів, день"
    )

    run_experiment(
        image_path="Images/uz-mountain.jpg",
        true_angle= 30.0,
        true_dx = 25.0,
        true_dy = 20.0,
        contrast_alpha = 0.8,
        experiment_name="Поїзд в карпатах (день)"
    )

    run_experiment(
        image_path="Images/uz-mountain.jpg",
        true_angle= 30.0,
        true_dx = 25.0,
        true_dy = 20.0,
        contrast_alpha = 0.8,
        experiment_name="Поїзд в карпатах (день)",
        descriptor="ORB"
    )
    run_experiment(
        image_path="Images/uz-mountain.jpg",
        true_angle= 30.0,
        true_dx = 25.0,
        true_dy = 20.0,
        contrast_alpha = 0.8,
        scale_factor=0.7,
        experiment_name="Поїзд в карпатах (день)",
        descriptor="ORB"
    )
    run_experiment(
        image_path="Images/uz-mountain2.jpg",
        true_angle= -35.0,
        true_dx = 20.0,
        true_dy = 10.0,
        contrast_alpha = 0.8,
        scale_factor=0.5,
        experiment_name="Поїзд в карпатах (день) 2",
        descriptor="ORB"
    )



if __name__ == "__main__":
    main()