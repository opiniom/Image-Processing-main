import numpy as np
import cv2
import random
from Utils import *
from Noise_Filters import *
from Filtering_Methods import *
from Hybrid_Filter import hybrid_filter
import os

RUN_ALL_FILTERS = False  # 전체 필터 비교 시 True로 변경

def main():
    # ============================================#
    #        디스크에서 이미지 읽어오기           #
    # ============================================#
    
    import glob
    import os
    
    target_folder = input("노이즈 이미지가 있는 폴더명을 입력하세요 (예: lena, baboon 등): ").strip()
    
    base_folder_path = os.path.join("imageset", target_folder)
    
    noise_images = glob.glob(f"{base_folder_path}/*noise*.*")
    if not noise_images:
        noise_images = glob.glob(f"{base_folder_path}/*.bmp") + glob.glob(f"{base_folder_path}/*.jpg") + glob.glob(f"{base_folder_path}/*.png")
        if not noise_images:
            print(f"에러: 'imageset/{target_folder}' 폴더 내에 노이즈 이미지를 찾을 수 없습니다.")
            return

    # 원본(Ground Truth) 이미지 로드
    orig_path = os.path.join(base_folder_path, f"{target_folder}.bmp")
    original_image = cv2.imread(orig_path)
    has_original = True
    if original_image is None:
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = os.path.join(base_folder_path, f"{target_folder}{ext}")
            if cv2.imread(test_path) is not None:
                orig_path = test_path
                original_image = cv2.imread(orig_path)
                break
                
    if original_image is None:
        print(f"\n[경고] 원본 이미지('{target_folder}.bmp' 등)를 찾을 수 없어 벤치마크 평가를 생략합니다.")
        print("[알림] 오직 복원된 이미지만 화면에 띄웁니다.\n")
        has_original = False
    else:
        imgGray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    import tester

    for img_path in noise_images:
        print(f"\n{'='*50}")
        print(f"[알림] 처리 중인 노이즈 이미지: {os.path.basename(img_path)}")
        print(f"{'='*50}")

        image = cv2.imread(img_path)
        if image is None:
            continue

        # 원본 이미지 출력 코드 주석 처리
        # cv2.imshow("Original Image", image)
        # cv2.waitKey(0)

        # ===========================================#
        #   이미지를 흑백(GrayScale)으로 변환하기    #
        # ===========================================#

        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print("Gray scale image: ", imgGray.shape,len(imgGray.shape))
        # cv2.imshow("Gray Image", imgGray)
        # cv2.waitKey(0)
        # cv2.imwrite("Results/Gray_Scale_Image.png", imgGray)

        # ==========================================#
        # 이미지에 점잡음(Salt and Pepper noise) 추가하기 #
        # ==========================================#
        # 노이즈를 추가하는 코드 주석처리 (이미 노이즈가 추가된 이미지를 사용하므로)
        # SPnoise_img = myImNoise(imgGray, param="saltandpepper", noise_percent=50)
        # cv2.imshow('Salt And Pepper Noise Image', SPnoise_img)
        # cv2.waitKey(0)
        # cv2.imwrite('Results/Salt_And_Pepper_Noise_Image.png', SPnoise_img)
        
        # 현재 읽어온 이미지가 이미 노이즈 이미지입니다.
        SPnoise_img = imgGray

        # =======================================#
        #   과거 필터들을 옵션 플래그로 깔끔하게 정리  #
        # =======================================#
        Base_Filtered, Filter1_Result, Filter2_Result = None, None, None
        if RUN_ALL_FILTERS:
            Base_Filtered = myImFilter(SPnoise_img, param="mean")
            Filter1_Result = progressive_median_filter(SPnoise_img)
            Filter2_Result = group_filter(SPnoise_img)
            
        # =======================================#
        #   최종 개발 모델: 하이브리드 필터 (HF)    #
        # =======================================#
        Filter3_Result = hybrid_filter(SPnoise_img)
        
        # =======================================#
        #  메인 함수의 처리 결과를 tester로 전달 #
        # =======================================#
        
        if has_original:
            print(f"\n[알림] {os.path.basename(img_path)} 의 영상 처리가 완료되었습니다. 테스터(tester.py) 벤치마크를 시작합니다...")
            tester.run_test(imgGray_original, SPnoise_img, Base_Filtered, Filter1_Result, Filter2_Result, Filter3_Result)
        else:
            print(f"\n[알림] {os.path.basename(img_path)} 처리가 완료되었습니다. 시각적 확인 팝업을 띄웁니다...")
            print("창을 닫으려면 [아무 키]나 누르세요.")
            
            # 비율 유지하며 크기 600x600 고정 후 이미지 띄우기
            display_noisy = cv2.resize(SPnoise_img, (512, 512), interpolation=cv2.INTER_AREA)
            display_hf = cv2.resize(Filter3_Result, (512, 512), interpolation=cv2.INTER_AREA)
            
            cv2.imshow("1. Corrupted Noise Image", display_noisy)
            cv2.imshow("2. Hybrid Filter (HF) Restored", display_hf)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
