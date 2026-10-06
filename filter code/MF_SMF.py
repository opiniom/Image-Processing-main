import cv2
import numpy as np
import time
import os
import sys
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import njit

# ==========================================================
# [사용자 수정 부분] (파일명만 수정하고 실행하면 됨)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 현재 스크립트 위치에서 상위 두 단계의 'dataset' 폴더를 기준으로 설정
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "dataset"))
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset"))

ORIGINAL_FILE = "pepper.bmp"           # 원본 이미지 파일명
NOISY_FILE    = "pepper_80noise.bmp"   # 노이즈 이미지 파일명
# 결과 저장 경로 (dataset/Mean_SMF/restore)
SAVE_DIR      = os.path.join(DATASET_DIR, "Mean_SMF", "restore") 
# ==========================================================

# --- 1. 유니코드 경로 지원 입출력 ---
def load_image_grayscale(path):
    """한글 경로에서도 안전하게 이미지를 로드합니다."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None

def save_image_unicode(path, img):
    """한글 경로에서도 안전하게 이미지를 저장합니다."""
    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
        return True
    return False

# --- 2. 경로 및 원본 찾기 로직 ---
def find_image_path(filename):
    """dataset 폴더 내부에서 파일을 재귀적으로 찾습니다."""
    for root, _, files in os.walk(DATASET_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_reference_image(image_path):
    """노이즈 파일명을 분석하여 자동으로 원본(Reference) 파일을 찾습니다."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    # 접미사 제거 (예: _90noise, _10noise 등)
    prefix = re.split(r'[_\-]', base)[0]
    search_roots = [os.path.join(DATASET_DIR, "1_original"), DATASET_DIR]
    
    for root in search_roots:
        if not os.path.exists(root): continue
        for file in os.listdir(root):
            name, ext = os.path.splitext(file)
            if name == prefix and ext.lower() in ('.bmp', '.png', '.jpg'):
                return os.path.join(root, file)
    return None

# --- 3. 알고리즘 코어 (SMF는 Numba 가속 적용) ---
@njit
def smf_core(img, rows, cols, k_size=3):
    """Switching Median Filter 핵심 연산 루프"""
    output = img.copy()
    pad = k_size // 2
    # Numba 내부에서는 직접적인 패딩보다 인덱스 제어가 유리함
    for r in range(rows):
        for c in range(cols):
            pixel = img[r, c]
            # 잡음 검출 (0: Pepper, 255: Salt)
            if pixel == 0 or pixel == 255:
                r0, r1 = max(0, r-pad), min(rows, r+pad+1)
                c0, c1 = max(0, c-pad), min(cols, c+pad+1)
                window = img[r0:r1, c0:c1].flatten()
                output[r, c] = np.median(window)
    return output

def apply_mean_filter(image, k_size=3):
    """Mean Filter (OpenCV 최적화 함수 사용)"""
    return cv2.blur(image, (k_size, k_size))

# --- 4. 메인 복원 실행 함수 ---
def run_restoration(filter_type="SMF"):
    # (1) 경로 탐색
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)

    if noisy_path is None:
        print(f"❌ 노이즈 파일을 찾을 수 없습니다: {NOISY_FILE}")
        return

    # (2) 이미지 로드
    nois = load_image_grayscale(noisy_path)
    orig = load_image_grayscale(orig_path) if orig_path else None
    
    if orig is None:
        # 파일명 기반으로 자동 탐색 시도
        ref_path = find_reference_image(noisy_path)
        if ref_path: orig = load_image_grayscale(ref_path)

    print(f"\n🚀 [{filter_type}] 처리 시작: {os.path.basename(noisy_path)}")
    
    start_time = time.perf_counter()
    
    # (3) 알고리즘 실행
    if filter_type == "SMF":
        res_raw = smf_core(nois.astype(np.float32), nois.shape[0], nois.shape[1], 3)
    else: # Mean
        res_raw = apply_mean_filter(nois, 3)
        
    restored = np.clip(res_raw, 0, 255).astype(np.uint8)
    elapsed_time = time.perf_counter() - start_time

    # (4) 지표 계산 (원본이 있을 경우만)
    metrics_str = ""
    if orig is not None and orig.shape == restored.shape:
        psnr_val = psnr(orig, restored, data_range=255)
        ssim_val = ssim(orig, restored, data_range=255)
        metrics_str = f"_{psnr_val:.2f}dB_{ssim_val:.4f}"
        print(f"📊 PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
    else:
        print("⚠️ 원본 이미지가 없어 지표를 계산할 수 없습니다.")

    # (5) 결과 저장
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    save_name = f"{noise_filename}_res_{filter_type}{metrics_str}_{elapsed_time:.4f}s.png"
    save_full_path = os.path.join(SAVE_DIR, save_name)
    
    if save_image_unicode(save_full_path, restored):
        print(f"✅ 저장 완료: {save_full_path}")
        print(f"⏱️ 소요 시간: {elapsed_time:.4f} 초")
    print("-" * 50)

if __name__ == "__main__":
    # 두 가지 필터를 각각 실행하여 비교
    run_restoration("Mean")
    run_restoration("SMF")