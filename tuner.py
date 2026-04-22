"""
Hybrid Filter 임계값 그리드 서치 튜너
======================================
cond1_thresh / cond2_thresh / cond3_thresh 조합을 자동으로 탐색하여
PSNR · SSIM 기준 최적 파라미터를 찾아 CSV로 저장합니다.

사용법:
    python tuner.py
"""

import cv2
import glob
import os
import csv
import numpy as np
from itertools import product
from datetime import datetime

from AHF import ahf_filter
from tester import calculate_metrics


# ────────────────────────────────────────────────
# 탐색 파라미터 범위 설정 (필요에 따라 조정)
# ────────────────────────────────────────────────
K_NEU_RANGE = [1, 2, 3]        # Neumann 임계값
K_MOO_RANGE = [3, 4, 5]        # Moore Group 임계값
K_MEAN_RANGE = [1, 2, 3, 4, 5] # Moore Mean 임계값


def load_images(folder_name: str):
    """노이즈 이미지 목록과 원본 이미지를 로드합니다."""
    base_path = os.path.join("imageset", folder_name)

    # 노이즈 이미지 탐색
    noise_imgs = glob.glob(f"{base_path}/*noise*.*")
    if not noise_imgs:
        for ext in ["*.bmp", "*.png", "*.jpg", "*.jpeg"]:
            noise_imgs += glob.glob(os.path.join(base_path, ext))
        # 원본(노이즈 없는) 이미지는 제외
        noise_imgs = [p for p in noise_imgs
                      if os.path.splitext(os.path.basename(p))[0] == folder_name]

    if not noise_imgs:
        raise FileNotFoundError(f"노이즈 이미지를 찾을 수 없습니다: {base_path}")

    # 원본 이미지 탐색
    original = None
    for ext in [".bmp", ".png", ".jpg", ".jpeg"]:
        p = os.path.join(base_path, f"{folder_name}{ext}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            original = img
            break

    if original is None:
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {base_path}/{folder_name}.*")

    return noise_imgs, original


def run_combo(noise_paths, original, c1, c2, c3):
    """하나의 임계값 조합으로 모든 노이즈 이미지를 처리하고 평균 지표를 반환합니다."""
    psnr_list, ssim_list, route_list = [], [], []

    for img_path in noise_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        result, routes = ahf_filter(
            img, return_route=True,
            k_neu_th=c1, k_moo_th=c2, k_mean_th=c3,
            verbose=False
        )
        p, s = calculate_metrics(original, result)
        psnr_list.append(p)
        ssim_list.append(s)
        route_list.append(routes)

    if not psnr_list:
        return None

    return {
        "avg_psnr":   float(np.mean(psnr_list)),
        "avg_ssim":   float(np.mean(ssim_list)),
        "avg_routes": float(np.mean(route_list)),
    }


def main():
    print("=" * 60)
    print("   Hybrid Filter 임계값 그리드 서치 튜너")
    print("=" * 60)

    folder = input("노이즈 이미지가 있는 폴더명을 입력하세요 (예: lena): ").strip()

    try:
        noise_paths, original = load_images(folder)
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        return

    print(f"\n노이즈 이미지 {len(noise_paths)}개 발견.")

    # 유효한 조합 생성 (k_mean <= k_moo 조건 적용)
    combos = [
        (kn, km_g, km_m)
        for kn, km_g, km_m in product(K_NEU_RANGE, K_MOO_RANGE, K_MEAN_RANGE)
        if km_m <= km_g
    ]
    total = len(combos)
    print(f"탐색 조합 수: {total}개\n")

    # 출력 CSV 준비
    os.makedirs("data_table", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data_table/Tuning_{folder}_{timestamp}.csv"

    results = []

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k_neu", "k_moo", "k_mean",
            "Avg PSNR (dB)", "Avg SSIM", "Avg Routes"
        ])

        for idx, (c1, c2, c3) in enumerate(combos, 1):
            print(f"[{idx:3d}/{total}] k_neu={c1}  k_moo={c2}  k_mean={c3} ... ", end="", flush=True)

            metrics = run_combo(noise_paths, original, c1, c2, c3)
            if metrics is None:
                print("건너뜀 (이미지 로드 실패)")
                continue

            p, s, r = metrics["avg_psnr"], metrics["avg_ssim"], metrics["avg_routes"]
            print(f"PSNR={p:.2f}  SSIM={s:.4f}  Routes={r:.1f}")

            writer.writerow([c1, c2, c3, round(p, 2), round(s, 4), round(r, 1)])
            results.append((c1, c2, c3, p, s, r))

    if not results:
        print("\n[오류] 유효한 결과가 없습니다.")
        return

    # ── 최적 조합 출력 ──
    best_psnr = max(results, key=lambda x: x[3])
    best_ssim = max(results, key=lambda x: x[4])

    print("\n" + "=" * 60)
    print("  최적 파라미터 (PSNR 기준)")
    print("=" * 60)
    print(f"  k_neu={best_psnr[0]}  k_moo={best_psnr[1]}  k_mean={best_psnr[2]}")
    print(f"  PSNR = {best_psnr[3]:.2f} dB   SSIM = {best_psnr[4]:.4f}   Routes = {best_psnr[5]:.1f}")

    if best_ssim[:3] != best_psnr[:3]:
        print("\n  최적 파라미터 (SSIM 기준, PSNR 기준과 다를 경우)")
        print(f"  k_neu={best_ssim[0]}  k_moo={best_ssim[1]}  k_mean={best_ssim[2]}")
        print(f"  PSNR = {best_ssim[3]:.2f} dB   SSIM = {best_ssim[4]:.4f}   Routes = {best_ssim[5]:.1f}")

    print(f"\n결과 CSV 저장 완료 → {csv_path}")


if __name__ == "__main__":
    main()
