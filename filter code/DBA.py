import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import njit

# ==========================================================
# [사용자 수정 부분] (파일명만 수정하고 실행하면 됨)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 현재 스크립트 위치에서 위쪽으로 올라가며 'dataset' 폴더를 찾습니다.
def find_dataset_dir(start_dir):
    current = start_dir
    while True:
        candidate = os.path.join(current, "dataset")
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent

DATASET_DIR = find_dataset_dir(SCRIPT_DIR)
if DATASET_DIR is None:
    raise FileNotFoundError("dataset 폴더를 찾을 수 없습니다. 스크립트 위치를 확인해주세요.")

ORIGINAL_FILE = "barbara.bmp"           # 원본 이미지 파일명
NOISY_FILE    = "barbara_80noise.bmp"   # 노이즈 이미지 파일명
# DBA 결과 전용 저장 경로 (dataset/DBA/restore)
SAVE_DIR      = os.path.join(DATASET_DIR, "DBA", "restore") 
FILTER_NAME   = "DBA"
# ==========================================================

def find_image_path(filename, subdirs=None):
    """dataset 폴더 내부에서 이미지 파일을 재귀적으로 찾습니다."""
    if subdirs is None:
        subdirs = ["", "1_original", "2_noise", "DBA"]

    for subdir in subdirs:
        candidate = os.path.join(DATASET_DIR, subdir, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    for root, _, files in os.walk(DATASET_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_image_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
    return img


def save_image(path, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        return False
    success, encoded = cv2.imencode(ext, img)
    if not success:
        return False
    try:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
        return True
    except Exception:
        return False

# --- 1. DBA 알고리즘 코어 (Numba 가속) ---
@njit
def dba_core(img, rows, cols):
    """Decision-Based Algorithm 핵심 연산 루프"""
    pad = 1
    # 테두리 처리를 위한 패딩 (3x3 윈도우)
    padded = np.zeros((rows + 2, cols + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = img
    output = img.copy()

    for i in range(rows):
        for j in range(cols):
            pixel = img[i, j]
            
            # [Step 1] 잡음 검출 (0: Pepper, 255: Salt)
            if pixel == 0 or pixel == 255:
                # [Step 2] 윈도우 추출 (3x3)
                r, c = i + 1, j + 1
                window = padded[r-1:r+2, c-1:c+2].flatten()
                
                # [Step 3] 정상 픽셀(0 < value < 255)만 필터링 재료로 선택
                valid_list = []
                for val in window:
                    if 0 < val < 255:
                        valid_list.append(val)
                
                if len(valid_list) > 0:
                    # 유효한 픽셀이 있으면 중간값으로 대체
                    output[i, j] = np.median(np.array(valid_list))
                else:
                    # 윈도우 전체가 잡음인 경우, '직전 픽셀' 사용
                    if j > 0:
                        output[i, j] = output[i, j-1]
                    elif i > 0:
                        output[i, j] = output[i-1, j]
            # 잡음이 아니면 원본 그대로 유지 (Switching)
            
    return output

# --- 2. 메인 복원 실행 함수 ---
def run_restoration():
    # (1) 경로 탐색
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)

    if orig_path is None or noisy_path is None:
        print("❌ 파일을 찾을 수 없습니다. dataset 폴더 구조를 확인하세요.")
        return

    # (2) 이미지 로드
    orig = load_image_grayscale(orig_path)
    nois = load_image_grayscale(noisy_path)

    if orig is None or nois is None:
        print("❌ 이미지 로드 실패!")
        return

    print(f"\n🚀 [{FILTER_NAME}] 복원 시작: {os.path.basename(noisy_path)}")
    
    # 시간 측정 (순수 알고리즘 구간만 측정)
    start_time = time.perf_counter()
    
    # (3) 알고리즘 실행
    res_raw = dba_core(nois.astype(np.float32), nois.shape[0], nois.shape[1])
    restored = np.clip(res_raw, 0, 255).astype(np.uint8)
    
    elapsed_time = time.perf_counter() - start_time

    # (4) 정량 지표 계산
    psnr_val = psnr(orig, restored, data_range=255)
    ssim_val = ssim(orig, restored, data_range=255)

    # (5) 결과 저장 경로 설정
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    # 파일명 형식 일치: (노이즈명)_res_(필터)_(PSNR)_(SSIM)_(시간).bmp
    save_filename = f"{noise_filename}_res_{FILTER_NAME}_{psnr_val:.2f}dB_{ssim_val:.4f}_{elapsed_time:.2f}s.bmp"
    save_full_path = os.path.join(SAVE_DIR, save_filename)
    
    if not save_image(save_full_path, restored):
        print(f"❌ 결과 저장 실패: {save_full_path}")
        return

    # (6) 결과 출력
    print("-" * 50)
    print(f"✅ 결과 저장 완료: {save_full_path}")
    print(f"📊 PSNR : {psnr_val:.2f} dB")
    print(f"📊 SSIM : {ssim_val:.4f}")
    print(f"⏱️ 알고리즘 연산 시간 : {elapsed_time:.4f} s")
    print("-" * 50)

if __name__ == "__main__":
    run_restoration()