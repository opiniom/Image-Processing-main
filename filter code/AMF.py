import cv2
import numpy as np
import time
import os
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import njit

# ==========================================================
# [사용자 수정 부분] (파일명만 수정하고 실행하면 됨)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 현재 스크립트 위치에서 상위 폴더 또는 상위 두 단계에 있는 'dataset' 폴더를 기준으로 설정
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
# AMF 결과 전용 저장 경로 (dataset/AMF/restore)
SAVE_DIR      = os.path.join(DATASET_DIR, "AMF", "restore") 
FILTER_NAME   = "AMF"
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
    """파일명 분석을 통한 자동 원본(Reference) 탐색"""
    base = os.path.splitext(os.path.basename(image_path))[0]
    prefix = re.split(r'[_\-]', base)[0]
    search_roots = [os.path.join(DATASET_DIR, "1_original"), DATASET_DIR]
    
    for root in search_roots:
        if not os.path.exists(root): continue
        for file in os.listdir(root):
            name, ext = os.path.splitext(file)
            if name == prefix and ext.lower() in ('.bmp', '.png', '.jpg'):
                return os.path.join(root, file)
    return None

# --- 3. AMF 알고리즘 코어 (Numba 가속 적용) ---
@njit
def amf_core(img, rows, cols, max_window_size=7):
    """Adaptive Median Filter 핵심 연산 루프"""
    output = np.copy(img)
    pad = max_window_size // 2
    # 패딩된 이미지 생성 (Numba 호환을 위해 0으로 초기화 후 복사)
    padded_img = np.zeros((rows + pad*2, cols + pad*2), dtype=np.float32)
    padded_img[pad:-pad, pad:-pad] = img.astype(np.float32)
    
    for y in range(rows):
        for x in range(cols):
            window_size = 3
            z_xy = img[y, x]
            
            while window_size <= max_window_size:
                w_pad = window_size // 2
                # 현재 위치 (y+pad, x+pad) 기준으로 윈도우 추출
                r, c = y + pad, x + pad
                window = padded_img[r - w_pad : r + w_pad + 1, 
                                    c - w_pad : c + w_pad + 1].flatten()
                
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                
                # Level A: 중간값이 잡음인지 확인
                if z_min < z_med < z_max:
                    # Level B: 현재 픽셀이 잡음인지 확인
                    if z_min < z_xy < z_max:
                        output[y, x] = z_xy
                    else:
                        output[y, x] = z_med
                    break # 현재 픽셀 처리 완료
                else:
                    window_size += 2 # 창 크기 확장
            
            # 최대 창 크기에서도 중간값을 못 찾은 경우
            if window_size > max_window_size:
                output[y, x] = z_med
                
    return output

# --- 4. 메인 복원 실행 함수 ---
def run_restoration():
    # (1) 경로 탐색
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)

    if noisy_path is None:
        print(f"❌ 파일을 찾을 수 없습니다: {NOISY_FILE}")
        return

    # (2) 이미지 로드
    nois = load_image_grayscale(noisy_path)
    orig = load_image_grayscale(orig_path) if orig_path else None
    
    if orig is None:
        ref_path = find_reference_image(noisy_path)
        if ref_path: orig = load_image_grayscale(ref_path)

    print(f"\n🚀 [{FILTER_NAME}] 복원 시작: {os.path.basename(noisy_path)}")
    
    # 시간 측정 (알고리즘 연산 구간만)
    start_time = time.perf_counter()
    
    # (3) 알고리즘 실행
    res_raw = amf_core(nois.astype(np.float32), nois.shape[0], nois.shape[1], 7)
    restored = np.clip(res_raw, 0, 255).astype(np.uint8)
    
    elapsed_time = time.perf_counter() - start_time

    # (4) 정량 지표 계산
    metrics_str = ""
    if orig is not None and orig.shape == restored.shape:
        psnr_val = psnr(orig, restored, data_range=255)
        ssim_val = ssim(orig, restored, data_range=255)
        metrics_str = f"_{psnr_val:.2f}dB_{ssim_val:.4f}"
        print(f"📊 PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
    else:
        print("⚠️ 원본 이미지가 없거나 크기가 달라 지표를 계산할 수 없습니다.")

    # (5) 결과 저장
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    save_name = f"{noise_filename}_res_{FILTER_NAME}{metrics_str}_{elapsed_time:.4f}s.bmp"
    save_full_path = os.path.join(SAVE_DIR, save_name)
    
    if save_image_unicode(save_full_path, restored):
        print(f"✅ 저장 완료: {save_full_path}")
        print(f"⏱️ 알고리즘 연산 시간 : {elapsed_time:.4f} 초")
    print("-" * 50)

if __name__ == "__main__":
    run_restoration()