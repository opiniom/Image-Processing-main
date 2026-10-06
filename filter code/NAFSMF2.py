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

# 현재 스크립트 위치에서 상위 폴더를 올라가며 'dataset' 폴더를 찾습니다.
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
# NAFSMF 결과 전용 저장 경로 (dataset/NAFSMF/restore)
SAVE_DIR      = os.path.join(DATASET_DIR, "NAFSMF", "restore") 
FILTER_NAME   = "NAFSMF"
# ==========================================================

def find_image_path(filename, subdirs=None):
    """dataset 폴더 내부에서 이미지 파일을 재귀적으로 찾습니다."""
    if subdirs is None:
        subdirs = ["", "1_original", "2_noise", "NAFSMF"]

    # 1. 우선 주요 서브디렉토리부터 확인
    for subdir in subdirs:
        candidate = os.path.join(DATASET_DIR, subdir, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # 2. 못 찾으면 dataset 전체 폴더를 재귀 탐색
    for root, _, files in os.walk(DATASET_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_image_grayscale(path):
    """OpenCV가 한글/Unicode 경로에서 실패할 때 안전하게 이미지를 로드합니다."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
    return img


def save_image(path, img):
    """Unicode 경로에서도 안전하게 이미지를 저장합니다."""
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
    except Exception as e:
        print(f"❌ 이미지 저장 실패: {e}")
        return False

# --- 1. NAFSMF 알고리즘 코어 (Numba 가속) ---
@njit
def get_adaptive_median(img, mask, r, c, rows, cols):
    """주변 유효 픽셀을 찾아 3x3~7x7까지 확장하며 중간값을 반환"""
    for s in range(1, 4): # 3x3 (s=1), 5x5 (s=2), 7x7 (s=3)
        r0, r1 = max(0, r-s), min(rows, r+s+1)
        c0, c1 = max(0, c-s), min(cols, c+s+1)
        
        valid_vals = []
        for i in range(r0, r1):
            for j in range(c0, c1):
                if mask[i, j] == 1: # 0(잡음)이 아닌 1(정상) 픽셀만 수집
                    valid_vals.append(img[i, j])
        
        if len(valid_vals) > 0:
            return np.median(np.array(valid_vals))
    
    return img[r, c] # 끝까지 못 찾으면 현재 값 유지

@njit
def nafsm_core(img, mask, rows, cols):
    """NAFSMF 핵심 연산 루프"""
    output = img.copy()
    for r in range(rows):
        for c in range(cols):
            # 잡음(mask=0)인 경우에만 복원 수행
            if mask[r, c] == 0:
                output[r, c] = get_adaptive_median(img, mask, r, c, rows, cols)
    return output

# --- 2. 메인 복원 실행 함수 ---
def run_restoration():
    # (1) 경로 탐색 알고리즘 실행
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)

    if orig_path is None or noisy_path is None:
        print("❌ 파일을 찾을 수 없습니다. dataset 폴더 구조를 확인하세요.")
        print(f"탐색 기준 경로: {DATASET_DIR}")
        return

    # (2) 이미지 로드 (Grayscale)
    orig = load_image_grayscale(orig_path)
    nois = load_image_grayscale(noisy_path)

    if orig is None or nois is None:
        print("❌ 이미지 로드 실패! 파일 손상 여부를 확인하세요.")
        return

    print(f"\n🚀 [{FILTER_NAME}] 복원 시작: {os.path.basename(noisy_path)}")
    
    # 시간 측정 (ACWMF와 동일하게 perf_counter 사용)
    start_time = time.perf_counter()
    rows, cols = nois.shape
    
    # 노이즈 마스크 생성 (0: 잡음, 1: 정상 픽셀)
    # Salt & Pepper(0, 255)가 아닌 값들만 정상으로 간주
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[(nois > 0) & (nois < 255)] = 1

    # (3) 알고리즘 실행
    res_raw = nafsm_core(nois.astype(np.float64), mask, rows, cols)
    restored = np.clip(res_raw, 0, 255).astype(np.uint8)
    
    elapsed_time = time.perf_counter() - start_time

    # (4) 정량 지표 계산
    psnr_val = psnr(orig, restored, data_range=255)
    ssim_val = ssim(orig, restored, data_range=255)

    # (5) 결과 저장 경로 설정
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    # ACWMF와 동일한 파일명 형식: (노이즈명)_res_(필터)_(PSNR)_(SSIM)_(시간).bmp
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
    print(f"⏱️ 실행시간 : {elapsed_time:.4f} s")
    print("-" * 50)

if __name__ == "__main__":
    run_restoration()