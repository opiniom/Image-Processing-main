import numpy as np
import cv2
import random
from Utils import *
from Noise_Filters import *
from Filtering_Methods import *

def main():
    # ============================================#
    #        디스크에서 이미지 읽어오기           #
    # ============================================#

    image = cv2.imread("resources/testimg.png")
    if image is None:
        print("에러: 'resources/testimg.png' 파일을 찾을 수 없거나 열 수 없습니다. 파일이 동일한 폴더에 있는지 확인해주세요.")
        return
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)


    # ===========================================#
    #   이미지를 흑백(GrayScale)으로 변환하기    #
    # ===========================================#

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Gray scale image: ", imgGray.shape,len(imgGray.shape))
    cv2.imshow("Gray Image", imgGray)
    cv2.waitKey(0)
    cv2.imwrite("Results/Gray_Scale_Image.png", imgGray)

    # ==========================================#
    # 이미지에 점잡음(Salt and Pepper noise) 추가하기 #
    # ==========================================#
    # 여기서 10(%)을 원하는 퍼센트로 조절하세요.
    SPnoise_img = myImNoise(imgGray, param="saltandpepper", noise_percent=90)
    cv2.imshow('Salt And Pepper Noise Image', SPnoise_img)
    cv2.waitKey(0)
    cv2.imwrite('Results/Salt_And_Pepper_Noise_Image.png', SPnoise_img)

    # =======================================#
    #   점잡음 이미지에 평균 필터 적용       #
    # =======================================#
    Mean_SP = myImFilter(SPnoise_img, param="mean")
    cv2.imshow('Mean filtered SP Image ', Mean_SP)
    cv2.waitKey(0)
    cv2.imwrite('Results/Mean_filtered_SP_Image.png', Mean_SP)

    # =======================================#
    #  (변경됨) 메인 함수의 처리 결과를 tester로 전달 #
    # =======================================#
    print("\n[알림] 기존 필터 이미지 출력이 완료되었습니다. 자동으로 테스터(tester.py) 벤치마크를 시작합니다...")
    import tester
    cv2.destroyAllWindows() # 기존에 열려있던 창 치우기 (선택사항)
    tester.run_test(imgGray, SPnoise_img, Mean_SP)

if __name__ == "__main__":
    main()
