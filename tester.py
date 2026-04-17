import cv2
import numpy as np
import sys

# skimage 임포트를 시도해보고, 없다면 사용자에게 설치 안내를 하기 위한 try-except 구문
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("\n[오류] skimage 라이브러리가 설치되어 있지 않습니다.")
    print("터미널 창에 아래 명령어를 입력해서 설치해 주세요:")
    print("python -m pip install scikit-image\n")
    sys.exit(1)

def show_resized(title, img, size=600):
    """어떤 사진이든 강제로 꽉 차는 정사각형 모양(600x600)으로 변형해 출력합니다."""
    display_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, display_img)

def calculate_metrics(img1, img2):
    """원본 이미지와 복원된 이미지 간의 PSNR과 SSIM을 계산합니다."""
    # 크기 검증
    if img1.shape != img2.shape:
        raise ValueError("두 이미지의 크기가 다릅니다.")
    
    psnr_score = cv2.PSNR(img1, img2)
    # 이미지 윈도우 크기에 문제가 안 생기도록 처리
    win_size = min(img1.shape[0], img1.shape[1], 7) 
    if win_size % 2 == 0:
        win_size -= 1
        
    ssim_score = ssim(img1, img2, data_range=255, win_size=win_size)
    
    return psnr_score, ssim_score

def run_test(source_img, noise_img, base_filtered, f1_filtered, f2_filtered, hf_filtered=None):
    print("="*50)
    print("   원본 이미지 vs 필터별 복원 차이 분석   ")
    print("="*50)
    
    print("[1] 전달받은 이미지 데이터 화면 출력...")
    if base_filtered is not None:
        show_resized("3. Base Filter Result", base_filtered)
        show_resized("4. Progressive Median", f1_filtered)
        show_resized("5. Group Filter", f2_filtered)
        
    if hf_filtered is not None:
        show_resized("6. Hybrid Filter (HF)", hf_filtered)
    
    # 노이즈가 추가된 직후의 상태 측정
    # psnr_noisy, ssim_noisy = calculate_metrics(source_img, noise_img)
    
    # 최종 복원 성능 평가 부분 삭제됨
    print("\n테스트가 완료되었습니다. 이미지 창에서 [아무 키]나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()