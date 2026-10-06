import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import njit


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path, image):
    if image is None:
        return False
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
    result, buf = cv2.imencode(ext, image)
    if not result or buf is None:
        return False
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False

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

ORIGINAL_FILE = "pepper.bmp"           # 원본 이미지 파일명
NOISY_FILE    = "pepper_80noise.bmp"   # 노이즈 이미지 파일명
# ACWMF 결과 전용 저장 경로 (dataset/ACWMF/restore)
SAVE_DIR      = os.path.join(DATASET_DIR, "ACWMF", "restore") 
FILTER_NAME   = "ACWMF"
# ==========================================================

def find_image_path(filename, subdirs=None):
    """dataset 폴더 내부에서 이미지 파일을 재귀적으로 찾습니다."""
    if subdirs is None:
        subdirs = ["", "1_original", "2_noise", "ACWMF"]

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

# --- 1. ACWMF 알고리즘 코어 (Numba 가속) ---
@njit
def acwmf_core(img, rows, cols):
    """ACWMF 핵심 연산 루프"""
    pad = 1
    # 테두리 처리를 위한 패딩 (3x3 윈도우 기준)
    padded = np.zeros((rows + 2, cols + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = img
    output = img.copy()

    # 적응형 임계값 파라미터 (일반적으로 논문에서 권장되는 설정)
    # [0, 10, 20, 30] 등 상황에 따라 조절 가능
    thresholds = np.array([0, 10, 20, 30], dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            # 3x3 윈도우 추출
            r, c = i + 1, j + 1
            window = padded[r-1:r+2, c-1:c+2].flatten()
            center_val = img[i, j]
            
            # 중간값 및 MAD(Median Absolute Deviation) 계산
            med = np.median(window)
            diff_list = np.abs(window - med)
            mad = np.median(diff_list)
            
            # 중앙 픽셀과 중간값의 차이 계산
            d = np.abs(center_val - med)
            
            # [Adaptive Weighting Logic]
            # 차이가 작으면 디테일로 판단하여 원본 가중치를 높임
            if d <= mad:
                output[i, j] = center_val
            else:
                # 차이가 크면 노이즈로 판단하여 가중치를 조절한 중간값 적용
                # 가중치(Weight) 3 적용 (중앙 픽셀을 3번 포함하여 정렬)
                weighted_window = np.zeros(11, dtype=np.float32)
                weighted_window[0:9] = window
                weighted_window[9] = center_val
                weighted_window[10] = center_val
                output[i, j] = np.median(weighted_window)
                
    return output

# --- 2. 메인 복원 실행 함수 ---
def run_restoration():
    # (1) 경로 탐색 알고리즘 실행
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)

    if orig_path is None or noisy_path is None:
        print("❌ 파일을 찾을 수 없습니다. dataset 폴더 구조를 확인하세요.")
        return

    # (2) 이미지 로드
    orig = imread_unicode(orig_path, cv2.IMREAD_GRAYSCALE)
    nois = imread_unicode(noisy_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or nois is None:
        print("❌ 이미지 로드 실패!")
        print(f"  원본 경로: {orig_path} (존재: {os.path.exists(orig_path)})")
        print(f"  노이즈 경로: {noisy_path} (존재: {os.path.exists(noisy_path)})")
        return

    print(f"\n🚀 [{FILTER_NAME}] 복원 시작: {os.path.basename(noisy_path)}")
    
    # 시간 측정 (perf_counter 사용)
    start_time = time.perf_counter()
    
    # (3) 알고리즘 실행
    res_raw = acwmf_core(nois.astype(np.float32), nois.shape[0], nois.shape[1])
    restored = np.clip(res_raw, 0, 255).astype(np.uint8)
    
    elapsed_time = time.perf_counter() - start_time

    # (4) 정량 지표 계산
    psnr_val = psnr(orig, restored, data_range=255)
    ssim_val = ssim(orig, restored, data_range=255)

    # (5) 결과 저장 경로 설정
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    # 파일명 형식 일치: (이름)_res_(필터)_(PSNR)_(SSIM)_(시간).bmp
    save_filename = f"{noise_filename}_res_{FILTER_NAME}_{psnr_val:.2f}dB_{ssim_val:.4f}_{elapsed_time:.2f}s.bmp"
    save_full_path = os.path.join(SAVE_DIR, save_filename)
    
    if not imwrite_unicode(save_full_path, restored):
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