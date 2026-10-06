import numpy as np
import time
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import njit

# ==========================================================
# [사용자 수정 부분] (여기만 수정하고 실행하면 됨)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset"))
ORIGINAL_FILE = "lena.bmp"                   # 원본 이미지 파일명
NOISY_FILE    = "lena_10noise.bmp"           # 노이즈 이미지 파일명
SAVE_DIR      = os.path.join(DATASET_DIR, "NAFSMF", "restore")            # 저장 폴더 경로
FILTER_NAME   = "NAFSMF"                                                        # 필터 이름(수정 할 필요 없음)
# ==========================================================

def find_image_path(filename, subdirs=None):
    """dataset 내부에서 이미지 파일을 찾습니다."""
    if subdirs is None:
        subdirs = ["", "NAFSMF"]

    # 우선 candidate 경로부터 확인
    for subdir in subdirs:
        candidate = os.path.join(DATASET_DIR, subdir, filename) if subdir else os.path.join(DATASET_DIR, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # dataset 전체 재귀 탐색
    for root, _, files in os.walk(DATASET_DIR):
        if filename in files:
            return os.path.join(root, filename)

    return None

@njit
def get_adaptive_median(img, mask, r, c, rows, cols):
    for s in range(1, 4): # 3x3 ~ 7x7 확장
        r0, r1 = max(0, r-s), min(rows, r+s+1)
        c0, c1 = max(0, c-s), min(cols, c+s+1)
        
        # Numba 호환 유효 픽셀 추출 (2차원 마스킹 에러 수정)
        valid_vals = []
        for i in range(r0, r1):
            for j in range(c0, c1):
                if mask[i, j] == 1:
                    valid_vals.append(img[i, j])
        
        if len(valid_vals) > 0:
            return np.median(np.array(valid_vals))
    return img[r, c]

@njit
def nafsm_core(img, mask, rows, cols):
    output = img.copy()
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 0:
                output[r, c] = get_adaptive_median(img, mask, r, c, rows, cols)
    return output

def run_restoration():
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)
    if orig_path is None or noisy_path is None:
        print("이미지 파일을 찾을 수 없습니다. 아래 파일명이 dataset 폴더 아래에 있는지 확인하세요.")
        print(f"  원본: {ORIGINAL_FILE}")
        print(f"  노이즈: {NOISY_FILE}")
        return

    orig = cv2.imread(orig_path, 0)
    nois = cv2.imread(noisy_path, 0)
    if orig is None or nois is None:
        print("이미지를 읽는 데 실패했습니다. 파일 형식 또는 경로를 확인하세요.")
        return

    start_time = time.time()
    rows, cols = nois.shape
    
    # 노이즈 마스크 생성
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[(nois > 0) & (nois < 255)] = 1

    # 복원 루프
    restored = nafsm_core(nois.astype(np.float64), mask, rows, cols)
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    
    elapsed_time = time.time() - start_time
    psnr_val = psnr(orig, restored, data_range=255)
    ssim_val = ssim(orig, restored, data_range=255)

    # 파일명 형식: (노이즈이미지이름)_restore_(필터이름)_(PSNR)_(SSIM)_(실행시간).bmp
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    save_name = f"{noise_filename}_restore_{FILTER_NAME}_{psnr_val:.2f}_{ssim_val:.4f}_{elapsed_time:.2f}s.bmp"
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    cv2.imwrite(os.path.join(SAVE_DIR, save_name), restored)

    print(f"[{FILTER_NAME} 복원 완료]")
    print(f"PSNR : {psnr_val:.2f}dB")
    print(f"SSIM : {ssim_val:.4f}")
    print(f"실행시간 : {elapsed_time:.2f}s")

if __name__ == "__main__":
    run_restoration()