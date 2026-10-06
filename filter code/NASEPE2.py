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


def safe_imwrite(path, image):
    try:
        ext = os.path.splitext(path)[1]
        result, buf = cv2.imencode(ext if ext else '.png', image)
        if not result:
            return False
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False

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
# 결과 저장 경로 (dataset/NASEPF/restore)
SAVE_DIR      = os.path.join(DATASET_DIR, "NASEPF", "restore") 
FILTER_NAME   = "NASEPF"
# ==========================================================

def find_image_path(filename, subdirs=None):
    """dataset 폴더 내부에서 이미지 파일을 재귀적으로 찾습니다."""
    if subdirs is None:
        subdirs = ["", "1_original", "1_original/source", "2_Noise_Image", "2_Noise_Image/barbara", "NASEPF"]

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

# --- 1. NASEPF 알고리즘 코어 (Numba 가속) ---
@njit
def nasepf_core(noisy_img, rows, cols):
    pad = 7
    # 연산 정밀도를 위해 float32 사용
    padded = np.zeros((rows + pad*2, cols + pad*2), dtype=np.float32)
    padded[pad:-pad, pad:-pad] = noisy_img
    restored = noisy_img.copy().astype(np.float32)
    
    for i in range(rows):
        for j in range(cols):
            # 잡음 검출 (0: Pepper, 255: Salt)
            if noisy_img[i, j] == 0 or noisy_img[i, j] == 255:
                r, c = i + pad, j + pad
                found = False
                # 적응형 창 확장 (3x3 ~ 13x13)
                for w in range(3, 15, 2):
                    offset = w // 2
                    win = padded[r-offset : r+offset+1, c-offset : c+offset+1]
                    
                    valid_list = []
                    for wr in range(w):
                        for wc in range(w):
                            # 잡음이 아닌 정상 픽셀만 수집
                            if 0 < win[wr, wc] < 255:
                                valid_list.append(win[wr, wc])
                    
                    if len(valid_list) >= 1:
                        valid = np.array(valid_list)
                        # [NASEPF 핵심 로직] 유효 픽셀 수와 표준편차에 따른 처리
                        if len(valid_list) < 4:
                            restored[i, j] = np.median(valid)
                        else:
                            # 국부 표준편차 기반 적응형 처리
                            if np.std(valid) < 20:
                                restored[i, j] = np.mean(valid)
                            else:
                                restored[i, j] = np.median(valid)
                        found = True
                        break
                
                # 끝까지 유효 픽셀을 못 찾으면 인접 픽셀 복사
                if not found:
                    restored[i, j] = restored[max(0, i-1), j]
    return restored

# --- 2. 메인 복원 실행 함수 ---
def run_restoration():
    # (1) 경로 탐색 알고리즘 실행
    orig_path = find_image_path(ORIGINAL_FILE)
    noisy_path = find_image_path(NOISY_FILE)

    if orig_path is None or noisy_path is None:
        print("❌ 파일을 찾을 수 없습니다. 파일명이 dataset 폴더 내에 있는지 확인하세요.")
        print(f"탐색 기준 경로: {DATASET_DIR}")
        return

    # (2) 이미지 로드
    orig = imread_unicode(orig_path, cv2.IMREAD_GRAYSCALE)
    nois = imread_unicode(noisy_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or nois is None:
        print("❌ 이미지 로드 실패! 파일 손상 여부를 확인하세요.")
        print(f"  원본 경로: {orig_path} (존재: {os.path.exists(orig_path)})")
        print(f"  노이즈 경로: {noisy_path} (존재: {os.path.exists(noisy_path)})")
        return

    print(f"\n🚀 [{FILTER_NAME}] 복원 시작: {os.path.basename(noisy_path)}")
    
    # 시간 측정 (perf_counter 사용)
    start_time = time.perf_counter()
    
    # (3) 알고리즘 실행
    res_raw = nasepf_core(nois.astype(np.float32), nois.shape[0], nois.shape[1])
    restored = np.clip(res_raw, 0, 255).astype(np.uint8)
    
    elapsed_time = time.perf_counter() - start_time

    # (4) 정량 지표 계산
    psnr_val = psnr(orig, restored, data_range=255)
    ssim_val = ssim(orig, restored, data_range=255)

    # (5) 결과 저장 경로 설정
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    noise_filename = os.path.splitext(os.path.basename(noisy_path))[0]
    # 파일명 형식: (노이즈이름)_res_(필터이름)_(PSNR)_(SSIM)_(시간).bmp
    save_filename = f"{noise_filename}_res_{FILTER_NAME}_{psnr_val:.2f}dB_{ssim_val:.4f}_{elapsed_time:.2f}s.bmp"
    save_full_path = os.path.join(SAVE_DIR, save_filename)
    
    if not safe_imwrite(save_full_path, restored):
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