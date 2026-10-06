import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import njit


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
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
# INLM 결과 전용 저장 경로
SAVE_DIR      = os.path.join(DATASET_DIR, "INLM", "restore") 
FILTER_NAME   = "INLM"
# ==========================================================

def find_image_path(filename, subdirs=None):
    """dataset 폴더 내부에서 이미지 파일을 재귀적으로 찾습니다."""
    if subdirs is None:
        subdirs = ["", "1_original", "1_original/source", "2_Noise_Image", "2_Noise_Image/barbara", "INLM"]

    # 기본 후보 경로 확인
    for subdir in subdirs:
        candidate = os.path.join(DATASET_DIR, subdir, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # 전체 재귀 탐색
    for root, _, files in os.walk(DATASET_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None

@njit
def fast_guide_filter(padded, img_shape, kernel_size):
    """Numba 가속 기반 가이드 이미지 생성 함수"""
    rows, cols = img_shape
    out = np.zeros((rows, cols), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            roi = padded[r:r+kernel_size, c:c+kernel_size]
            
            valid_list = []
            v_sum = 0.0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    val = roi[i, j]
                    if 0 < val < 255:
                        valid_list.append(val)
                        v_sum += val
            
            if len(valid_list) > 0:
                valid_pixels = np.array(valid_list)
                v_median = np.median(valid_pixels)
                v_mean = v_sum / len(valid_list)
                # 중간값 70% + 평균값 30% 혼합으로 가이드 픽셀 생성
                out[r, c] = np.uint8(v_median * 0.7 + v_mean * 0.3)
            else:
                # 유효 픽셀이 없는 경우 인접 픽셀 복사
                if c > 0: out[r, c] = out[r, c-1]
                elif r > 0: out[r, c] = out[r-1, c]
    return out

def run_restoration():
    # 1. 경로 탐색
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)

    if orig_path is None or noisy_path is None:
        print("❌ 파일을 찾을 수 없습니다. dataset 폴더 구조를 확인하세요.")
        return

    # 2. 이미지 로드
    original = imread_unicode(orig_path, cv2.IMREAD_COLOR)
    noisy = imread_unicode(noisy_path, cv2.IMREAD_COLOR)
    if original is None or noisy is None:
        print("❌ 이미지 로드 실패!")
        print(f"  원본 경로: {orig_path} (존재: {os.path.exists(orig_path)})")
        print(f"  노이즈 경로: {noisy_path} (존재: {os.path.exists(noisy_path)})")
        return

    print(f"\n🚀 [{FILTER_NAME}] 복원 시작: {os.path.basename(noisy_path)}")
    
    # 시간 측정 (perf_counter 사용)
    start_time = time.perf_counter()
    
    # 3. 복원 프로세스
    gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    
    # [Step 1] 가이드 이미지 생성 (11x11 커널)
    pad = 11 // 2
    padded = cv2.copyMakeBorder(gray_noisy, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    guide_gray = fast_guide_filter(padded, gray_noisy.shape, 11)
    
    # [Step 2] NLM 필터링 (컬러 기반)
    guide_bgr = cv2.cvtColor(guide_gray, cv2.COLOR_GRAY2BGR)
    # h: 필터 강도, hColor: 컬러 필터 강도, templateWindowSize: 7, searchWindowSize: 21
    restored = cv2.fastNlMeansDenoisingColored(guide_bgr, None, 8, 8, 5, 21)
    
    # [Step 3] 비노이즈 픽셀 주입 (데이터 무결성 보존)
    mask = (noisy > 0) & (noisy < 255)
    restored[mask] = noisy[mask]

    elapsed_time = time.perf_counter() - start_time

    # 4. 정량 지표 계산
    psnr_val = psnr(original, restored, data_range=255)
    ssim_val = ssim(original, restored, channel_axis=2, data_range=255)

    # 5. 결과 저장 및 출력
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    save_name = f"{noise_filename}_res_{FILTER_NAME}_{psnr_val:.2f}dB_{ssim_val:.4f}_{elapsed_time:.2f}s.bmp"
    save_full_path = os.path.join(SAVE_DIR, save_name)
    
    if not imwrite_unicode(save_full_path, restored):
        print(f"❌ 결과 저장 실패: {save_full_path}")
        return

    print("-" * 50)
    print(f"✅ 결과 저장 완료: {save_full_path}")
    print(f"📊 PSNR : {psnr_val:.2f} dB")
    print(f"📊 SSIM : {ssim_val:.4f}")
    print(f"⏱️ 실행시간 : {elapsed_time:.4f} s")
    print("-" * 50)

if __name__ == "__main__":
    run_restoration()