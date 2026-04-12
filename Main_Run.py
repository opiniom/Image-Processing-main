import numpy as np
import cv2
import random
from Utils import *
from Noise_Filters import *
from Filtering_Methods import *
from Hybrid_Filter import hybrid_filter

def main():
    # ============================================#
    #        디스크에서 이미지 읽어오기           #
    # ============================================#
    
    # 원본(Ground Truth) 이미지 로드
    orig_path = "Results/lena.bmp"
    original_image = cv2.imread(orig_path)
    if original_image is None:
        print(f"에러: '{orig_path}' 파일을 찾을 수 없습니다.")
        return
    imgGray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    import glob
    import os
    
    # Results 폴더 내의 노이즈가 추가된 영상들을 가져옵니다.
    noise_images = glob.glob("Results/*noise*.bmp")
    
    if not noise_images:
        print("에러: 'Results' 폴더 내에 노이즈 이미지를 찾을 수 없습니다.")
        return

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

        # ============================================#
        #    최적화 테스트를 위해 다른 필터는 주석 처리   #
        # ============================================#
        # Base_Filtered = myImFilter(SPnoise_img, 'progressive_mean')
        # cv2.imwrite(f'Results/Base_Filtered_{os.path.basename(img_path)}.png', Base_Filtered)

        # cv2.waitKey(0)
        # cv2.imwrite(f'Results/Filter2_Result_{os.path.basename(img_path)}.png', Filter2_Result)

        # =======================================#
        #   제안 필터 3: 하이브리드 필터 (HF)
        # =======================================#
        Filter3_Result = hybrid_filter(SPnoise_img)
        
        # =======================================#
        #  메인 함수의 처리 결과를 tester로 전달 #
        # =======================================#
        print(f"\n[알림] {os.path.basename(img_path)} 의 모든 필터 이미지 처리가 완료되었습니다. 자동으로 테스터(tester.py) 벤치마크를 시작합니다...")
        
        # 최적화 진행중: HF 필터만 벤치마크 테스트 진행
        tester.run_test(imgGray_original, SPnoise_img, None, None, None, Filter3_Result)

if __name__ == "__main__":
    main()
