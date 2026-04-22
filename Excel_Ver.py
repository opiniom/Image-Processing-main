import cv2
import glob
import os
import csv
import numpy as np
from datetime import datetime
import platform


from AMSHF import amshf_filter
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
                

                # 1. AMSHF 실행
                ah_res, ah_routes = amshf_filter(SPnoise_img, return_route=True, verbose=False)
                
                # 복원된 이미지 저장 (AHF 저장)
                restored_path = os.path.join(restored_folder, f"restored_{img_name}")
                cv2.imwrite(restored_path, ah_res)
                
                if calc_metrics:
                    p_res, s_res = calculate_metrics(imgGray_original, ah_res)
                    writer.writerow([img_name, "AMSHF (Final)", round(p_res, 2), round(s_res, 4), ah_routes])
                else:
                    writer.writerow([img_name, "AMSHF (Final)", "-", "-", ah_routes])

                
        print(f" -> 성공적으로 데이터 추출 완료: {csv_filename}")


    print("\n" + "="*60)
    print("모든 처리 및 CSV 파일(AMSHF_*) 저장이 완료되었습니다!")
    print("="*60)

if __name__ == "__main__":
    main()
