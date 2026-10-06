import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numba import njit

# ==========================================================
# [사용자 수정 부분] (여기만 수정하고 실행하면 됨)
# ==========================================================
ORIGINAL_PATH = r"C:\Users\dmsrb\Desktop\MedLab\image\1_original\pepper.bmp"            #원본 이미지 경로
NOISY_PATH    = r"C:\Users\dmsrb\Desktop\MedLab\image\2_noise\pepper_90noise.bmp"         #노이즈 이미지 경로  
SAVE_DIR      = r"C:\Users\dmsrb\Desktop\MedLab\image\3_restore_NASEPF"         #저장 폴더 경로
FILTER_NAME   = "INLM"                                                                    #필터 이름(수정 할 필요 없음)
# ==========================================================

@njit
def fast_guide_filter(padded, img_shape, kernel_size):
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
                out[r, c] = np.uint8(v_median * 0.7 + v_mean * 0.3)
            else:
                if c > 0: out[r, c] = out[r, c-1]
                elif r > 0: out[r, c] = out[r-1, c]
    return out

def run_restoration():
    original = cv2.imread(ORIGINAL_PATH)
    noisy = cv2.imread(NOISY_PATH)
    if original is None or noisy is None: return

    start_time = time.time()
    gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    
    # 1. 가이드 이미지 생성
    pad = 11 // 2
    padded = cv2.copyMakeBorder(gray_noisy, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    guide_gray = fast_guide_filter(padded, gray_noisy.shape, 11)
    
    # 2. NLM 필터링
    guide_bgr = cv2.cvtColor(guide_gray, cv2.COLOR_GRAY2BGR)
    restored = cv2.fastNlMeansDenoisingColored(guide_bgr, None, 8, 8, 5, 21)
    
    # 3. 비노이즈 픽셀 주입
    mask = (noisy > 0) & (noisy < 255)
    restored[mask] = noisy[mask]

    elapsed_time = time.time() - start_time
    psnr_val = psnr(original, restored)
    ssim_val = ssim(original, restored, channel_axis=2, data_range=255)

    noise_filename = os.path.splitext(os.path.basename(NOISY_PATH))[0]
    save_name = f"{noise_filename}_restore_{FILTER_NAME}_{psnr_val:.2f}_{ssim_val:.4f}_{elapsed_time:.2f}s.bmp"
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    cv2.imwrite(os.path.join(SAVE_DIR, save_name), restored)

    print(f"[{FILTER_NAME} 복원 완료]")
    print(f"PSNR : {psnr_val:.2f}dB")
    print(f"SSIM : {ssim_val:.4f}")
    print(f"실행시간 : {elapsed_time:.2f}s")

if __name__ == "__main__":
    run_restoration()