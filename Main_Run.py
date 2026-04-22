import numpy as np
import cv2
from AMSHF import amshf_filter
import os
import tester


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
        #   최종 개발 모델: AMSHF (Adaptive)      #
        # =======================================#
        result_img, stats = amshf_filter(SPnoise_img, return_stats=True)
        
        # 필터 사용 통계 출력
        print("\n" + "="*55)
        print(" [ 최종 AMSHF 필터 복원 통계 ]")
        print("="*55)
        print(f"{'구분':<15} | {'복원 픽셀 수':>12}")
        print("-" * 55)
        print(f"{'1. Median':<15} | {stats['median']:>12} px")
        print(f"{'2. Group':<15} | {stats['group']:>12} px")
        print(f"{'3. Mean':<15} | {stats['mean']:>12} px")
        print("-" * 55)
        total_restored = sum(stats.values())
        print(f"{'총 복원 합계':<15} | {total_restored:>12} px")
        print("="*55)

        # 성능 비교 출력 부분 삭제됨
        
        # =======================================#
        #  메인 함수의 처리 결과를 tester로 전달 #
        # =======================================#
        
        if has_original:
            print(f"\n[알림] {os.path.basename(img_path)} 의 영상 처리가 완료되었습니다. 테스터(tester.py) 벤치마크를 시작합니다...")
            # tester에는 이전 인자 구조가 남아있을 수 있으므로 필요한 부분만 전달
            tester.run_test(imgGray_original, SPnoise_img, None, None, None, result_img)
        else:
            print(f"\n[알림] {os.path.basename(img_path)} 처리가 완료되었습니다. 시각적 확인 팝업을 띄웁니다...")
            print("창을 닫으려면 [아무 키]나 누르세요.")
            
            display_noisy = cv2.resize(SPnoise_img, (512, 512), interpolation=cv2.INTER_AREA)
            display_res = cv2.resize(result_img, (512, 512), interpolation=cv2.INTER_AREA)
            
            cv2.imshow("1. Corrupted Noise Image", display_noisy)
            cv2.imshow("2. AMSHF Restored", display_res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
