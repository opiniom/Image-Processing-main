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
FILTER_NAME   = "NASEPF"                                                                    #필터 이름(수정 할 필요 없음)
# ==========================================================

@njit
def nasepf_core(noisy_img, rows, cols):
    pad = 7
    padded = np.zeros((rows + pad*2, cols + pad*2), dtype=np.float32)
    padded[pad:-pad, pad:-pad] = noisy_img
    restored = noisy_img.copy().astype(np.float32)
    
    for i in range(rows):
        for j in range(cols):
            if noisy_img[i, j] == 0 or noisy_img[i, j] == 255:
                r, c = i + pad, j + pad
                found = False
                for w in range(3, 15, 2):
                    offset = w // 2
                    win = padded[r-offset : r+offset+1, c-offset : c+offset+1]
                    
                    valid_list = []
                    for wr in range(w):
                        for wc in range(w):
                            if 0 < win[wr, wc] < 255:
                                valid_list.append(win[wr, wc])
                    
                    if len(valid_list) >= 1:
                        valid = np.array(valid_list)
                        if len(valid_list) < 4:
                            restored[i, j] = np.median(valid)
                        else:
                            if np.std(valid) < 20:
                                restored[i, j] = np.mean(valid)
                            else:
                                restored[i, j] = np.median(valid)
                        found = True
                        break
                if not found:
                    restored[i, j] = restored[max(0, i-1), j]
    return restored

def run_restoration():
    orig = cv2.imread(ORIGINAL_PATH, 0)
    nois = cv2.imread(NOISY_PATH, 0)
    if orig is None or nois is None: return

    start_time = time.time()
    res = nasepf_core(nois.astype(np.float32), nois.shape[0], nois.shape[1])
    restored = np.clip(res, 0, 255).astype(np.uint8)
    
    elapsed_time = time.time() - start_time
    psnr_val = psnr(orig, restored)
    ssim_val = ssim(orig, restored)

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