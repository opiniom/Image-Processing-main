import cv2
import glob
import os
import csv
import numpy as np
from datetime import datetime
import platform

# --- 시각화(Matplotlib) 관련 임포트 (사용하지 않으므로 주석 처리) ---
# try:
#     import matplotlib.pyplot as plt
#     from matplotlib import rc
#     HAS_MATPLOTLIB = True
# except ImportError:
#     HAS_MATPLOTLIB = False

# --- 기존 필터들 (사용하지 않으므로 주석 처리) ---
# from Filtering_Methods import progressive_mean_filter, progressive_median_filter, group_filter, myImFilter

from Hybrid_Filter import hybrid_filter
from AMSHF import amshf_filter
from AHF import ahf_filter
from tester import calculate_metrics

def main():
    print("="*60)
    print("     초고속 벤치마크 데이터 추출기 (AMSHF CSV 전용)     ")
    print("="*60)
    
    # 한 번에 모든 이미지셋을 처리하도록 리스트화
    target_folders = ['baboon', 'barbara', 'cameraman', 'lena', 'pepper']
    
    for target_folder in target_folders:
        print(f"\n[{target_folder}] 데이터셋 처리를 시작합니다...")
        base_folder_path = os.path.join("imageset", target_folder)
        
        noise_images = glob.glob(f"{base_folder_path}/*noise*.*")
        if not noise_images:
            noise_images = glob.glob(f"{base_folder_path}/*.bmp") + glob.glob(f"{base_folder_path}/*.jpg") + glob.glob(f"{base_folder_path}/*.png")
            if not noise_images:
                print(f"에러: 'imageset/{target_folder}' 폴더 내에 노이즈 이미지를 찾을 수 없습니다.")
                continue

        # 복원된 이미지를 저장할 폴더 생성 (Restored 폴더 내부)
        restored_folder = os.path.join("Restored", target_folder)
        os.makedirs(restored_folder, exist_ok=True)
        
        # 원본 이미지 로드 로직 (메트릭 계산용)
        orig_path = os.path.join(base_folder_path, f"{target_folder}.bmp")
        original_image = cv2.imread(orig_path)
        calc_metrics = True
        
        if original_image is None:
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(base_folder_path, f"{target_folder}{ext}")
                if cv2.imread(test_path) is not None:
                    orig_path = test_path
                    original_image = cv2.imread(orig_path)
                    break

        if original_image is None:
            print(f"[경고] 원본 이미지('{target_folder}')를 찾을 수 없어 PSNR/SSIM 평가를 생략합니다.")
            calc_metrics = False
        else:
            imgGray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        os.makedirs("data_table", exist_ok=True)
        
        # 저장 파일명을 AMSHF_{노이즈 이름}.csv 로 고정
        csv_filename = f"data_table/AMSHF_{target_folder}.csv"
        
        with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "Filter Name", "PSNR (dB)", "SSIM", "Route Count"])
            
            for img_path in noise_images:
                img_name = os.path.basename(img_path)
                image = cv2.imread(img_path)
                SPnoise_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # --- 기존 3가지 필터 실행부 (사용 안 함, 주석 처리) ---
                # base_res, base_routes = myImFilter(SPnoise_img, 'progressive_mean', return_route=True)
                # f1_res, f1_routes = progressive_median_filter(SPnoise_img, return_route=True)
                # grp_res, grp_routes = group_filter(SPnoise_img, return_route=True)
                
                # 4. Hybrid Filter (HF) 실행 (주석 처리)
                # hf_res, hf_routes = hybrid_filter(SPnoise_img, return_route=True, verbose=False)
                
                # 5. AMSHF (Experimental) 실행 (주석 처리)
                # am_res, am_routes = amshf_filter(SPnoise_img, return_route=True, verbose=False)

                # 6. AHF (Algorithm.txt) 실행
                ah_res, ah_routes = ahf_filter(SPnoise_img, return_route=True, verbose=False)
                
                # 복원된 이미지 저장 (AHF 저장)
                restored_path = os.path.join(restored_folder, f"restored_{img_name}")
                cv2.imwrite(restored_path, ah_res)
                
                if calc_metrics:
                    # p_hf, s_hf = calculate_metrics(imgGray_original, hf_res)
                    # p_am, s_am = calculate_metrics(imgGray_original, am_res)
                    p_ah, s_ah = calculate_metrics(imgGray_original, ah_res)
                    # writer.writerow([img_name, "Hybrid Filter (HF)", round(p_hf, 2), round(s_hf, 4), hf_routes])
                    # writer.writerow([img_name, "AMSHF (Experiment)", round(p_am, 2), round(s_am, 4), am_routes])
                    writer.writerow([img_name, "AHF (Proposed)", round(p_ah, 2), round(s_ah, 4), ah_routes])
                else:
                    # writer.writerow([img_name, "Hybrid Filter (HF)", "-", "-", hf_routes])
                    # writer.writerow([img_name, "AMSHF (Experiment)", "-", "-", am_routes])
                    writer.writerow([img_name, "AHF (Proposed)", "-", "-", ah_routes])

                
        print(f" -> 성공적으로 데이터 추출 완료: {csv_filename}")

    # --- 시각화 (Matplotlib) 코드 (사용 안 함, 주석 처리) ---
    """
    if HAS_MATPLOTLIB and calc_metrics:
        print("\n -> 통계 차트 그래픽 렌더링 중...")
        # ...그래프 생성 코드 생략...
    """

    print("\n" + "="*60)
    print("모든 처리 및 CSV 파일(AMSHF_*) 저장이 완료되었습니다!")
    print("="*60)

if __name__ == "__main__":
    main()
