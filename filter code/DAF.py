import cv2
import numpy as np
import time
import os
import sys
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================================
# [사용자 수정 부분] (파일명만 수정하고 실행하면 됨)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 현재 스크립트 위치에서 위로 올라가며 'dataset' 폴더를 찾습니다.
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

# 테스트할 특정 파일명을 지정하거나, 폴더 전체 처리를 위해 아래 main 루프를 사용하세요.
# 여기서는 ACWMF 스타일 템플릿 유지를 위해 변수만 선언합니다.
FILTER_NAME   = "DAF" 
# ==========================================================


def load_image_grayscale(path):
    """한글/Unicode 경로에서도 안전하게 이미지를 로드합니다."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
    return img


def save_image(path, img):
    """한글/Unicode 경로에서도 안전하게 이미지를 저장합니다."""
    ext = os.path.splitext(path)[1]
    if ext == "":
        print(f"❌ 이미지 저장 실패: 확장자가 없습니다. {path}")
        return False
    success, encoded = cv2.imencode(ext, img)
    if not success:
        print(f"❌ 이미지 인코딩 실패: {path}")
        return False
    try:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
        return True
    except Exception as e:
        print(f"❌ 이미지 저장 실패: {e}")
        return False

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def find_reference_image(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    search_roots = [os.path.join(DATASET_DIR, "1_original"), DATASET_DIR]

    for root in search_roots:
        for ext in IMAGE_EXTENSIONS:
            candidate = os.path.join(root, base + ext)
            if os.path.exists(candidate):
                if os.path.abspath(candidate) == os.path.abspath(image_path):
                    continue
                return candidate

    prefix = re.split(r'[_\-]', base)[0]
    if prefix != base:
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for file in os.listdir(root):
                name, ext = os.path.splitext(file)
                if ext.lower() in IMAGE_EXTENSIONS and name == prefix:
                    return os.path.join(root, file)

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for file in os.listdir(root):
            name, ext = os.path.splitext(file)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            if base.startswith(name + "_") or name.startswith(base + "_"):
                return os.path.join(root, file)

    return None


def compute_metrics(reference, target):
    if reference is None or target is None:
        return None
    if reference.shape != target.shape:
        return None
    try:
        return psnr(reference, target, data_range=255), ssim(reference, target, data_range=255)
    except Exception:
        return None

# --- 1. Detail-Aware (Rolling Guidance) 필터 알고리즘 정의 ---
def apply_detail_aware_filter(image, sigma_s=3.0, sigma_r=25.0, num_of_iter=4):
    """
    Rolling Guidance Filter (RGF)를 적용하여 구조는 보존하고 미세 질감(노이즈)을 제거하는 디테일 인지 필터.
    Args:
        image (np.ndarray): 입력 이미지 (Grayscale)
        sigma_s (float): 공간 표준편차 (숫자가 클수록 더 넓은 범위를 참고하여 '구조'를 단순화함)
        sigma_r (float): 색상 표준편차 (숫자가 클수록 색 차이를 무시하고 뭉갬 -> 질감 제거 강도)
        num_of_iter (int): 반복 횟수 (보통 2~5 사이)
    Returns:
        np.ndarray: 필터링된 결과 이미지
    """
    # opencv-contrib-python의 ximgproc 모듈 사용
    # d=-1로 설정하면 sigma_s에 따라 자동으로 지름이 결정되어 더 정밀한 처리가 가능합니다.
    try:
        result = cv2.ximgproc.rollingGuidanceFilter(image, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s, numOfIter=num_of_iter)
        return result
    except AttributeError:
        print("❌ 오류: OpenCV ximgproc 모듈을 사용할 수 없습니다. opencv-contrib-python 설치를 확인하세요.")
        return None

# --- 2. 메인 복원 실행 함수 (배치 처리용으로 수정) ---
def run_restoration(image_path, save_root):
    """
    이미지 경로를 받아 DAF 복원을 수행하고 지표를 출력/저장합니다. (ACWMF 스타일)
    """
    # (1) 이미지 로드 (Unicode-safe Grayscale 로드)
    src = load_image_grayscale(image_path)
    
    if src is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return

    file_basename = os.path.basename(image_path)
    print(f"\n🚀 [{FILTER_NAME}] 처리 시작: {file_basename}")
    
    # (2) 시간 측정 시작 (고정밀 perf_counter 사용)
    start_time = time.perf_counter()
    
    # 알고리즘 실행 (기존 파라미터 유지)
    filtered_res = apply_detail_aware_filter(src, sigma_s=3.0, sigma_r=25.0, num_of_iter=4)
    
    if filtered_res is None: return
    
    elapsed_time = time.perf_counter() - start_time

    # (3) 결과 저장 경로 조립 (dataset/DAF/restore)
    restore_dir = os.path.join(save_root, FILTER_NAME, "restore")
    if not os.path.exists(restore_dir):
        os.makedirs(restore_dir)
        
    raw_name = os.path.splitext(file_basename)[0]
    # ACWMF 스타일 파일명: (노이즈명)_res_(필터)_(시간).bmp
    # ※ 원본이 없는 배치 처리이므로 PSNR/SSIM 수치는 제외하고 시간만 기록
    save_name = f"{raw_name}_res_{FILTER_NAME}_{elapsed_time:.4f}s.png"
    save_full_path = os.path.join(restore_dir, save_name)
    
    # 결과 저장 (Unicode-safe 저장)
    if not save_image(save_full_path, filtered_res):
        print(f"❌ 저장 실패: {save_full_path}")
        return

    ref_path = find_reference_image(image_path)
    if ref_path is not None:
        ref_img = load_image_grayscale(ref_path)
        metrics = compute_metrics(ref_img, filtered_res)
    else:
        metrics = None

    # (4) 결과 출력 (ACWMF 스타일 포맷)
    print("-" * 50)
    print(f"✅ 저장 완료: {save_full_path}")
    if metrics is not None:
        psnr_val, ssim_val = metrics
        print(f"📊 PSNR : {psnr_val:.2f} dB")
        print(f"📊 SSIM : {ssim_val:.4f}")
    else:
        print("📊 PSNR/SSIM 평가 불가: 원본 이미지가 없거나 크기가 다릅니다.")
    print(f"⏱️ 소요 시간: {elapsed_time:.4f} 초")
    print("-" * 50)

# --- 실행부 ---
if __name__ == "__main__":
    print(f"현재 프로젝트 기준 경로: {DATASET_DIR}\n")
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ 오류: {DATASET_DIR} 디렉토리를 찾을 수 없습니다.")
    else:
        # 지원하는 이미지 확장자
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        # 폴더 내 파일 목록 가져오기
        try:
            all_files = os.listdir(DATASET_DIR)
        except FileNotFoundError:
            print("❌ 오류: Dataset 디렉토리를 읽을 수 없습니다.")
            sys.exit()

        image_files = [f for f in all_files 
                       if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(DATASET_DIR, f))]
        
        if not image_files:
            print(f"❌ 오류: {DATASET_DIR}에서 이미지 파일을 찾을 수 없습니다.")
        else:
            print(f"📂 발견된 이미지 파일: {len(image_files)}개 처리 시작...")
            
            # 각 이미지 파일을 순차적으로 처리 (배치 프로세싱)
            for img_file in image_files:
                image_full_path = os.path.join(DATASET_DIR, img_file)
                run_restoration(image_full_path, DATASET_DIR)
                
            print("\n🎉 모든 이미지 처리가 완료되었습니다.")