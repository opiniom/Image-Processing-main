import cv2
import numpy as np
import time
from Noise_Filters import myImNoise
from Filtering_Methods import myImFilter
from tester import calculate_metrics

def headless_test():
    # 1. 원본 이미지 읽기 및 리사이즈 (테스트 속도 향상)
    image_full = cv2.imread("resources/testimg.png", cv2.IMREAD_GRAYSCALE)
    if image_full is None:
        print("Failed to read testimg.png")
        return
    image = cv2.resize(image_full, (256, 256))
    
    print("Image loaded:", image.shape)
    
    # 2. 노이즈 추가 (50%)
    print("Applying 50% Salt and Pepper noise...")
    start_time = time.time()
    noise_img = myImNoise(image, param="saltandpepper", noise_percent=50)
    print(f"Noise added in {time.time() - start_time:.2f} seconds")
    
    # 측정
    psnr_noisy, ssim_noisy = calculate_metrics(image, noise_img)
    print(f"Noisy PSNR: {psnr_noisy:.2f}, SSIM: {ssim_noisy:.4f}")
    
    # 3. 필터 적용
    print("\nApplying Progressive Median Filter...")
    start_time = time.time()
    filtered_img = myImFilter(noise_img, param="mean")
    print(f"Filter finished in {time.time() - start_time:.2f} seconds")
    
    # 결과 측정
    psnr_filtered, ssim_filtered = calculate_metrics(image, filtered_img)
    print(f"\nFiltered PSNR: {psnr_filtered:.2f}, SSIM: {ssim_filtered:.4f}")
    
    cv2.imwrite("Results/headless_filtered.png", filtered_img)
    print("Done. Image saved to Results/headless_filtered.png")

if __name__ == "__main__":
    headless_test()
