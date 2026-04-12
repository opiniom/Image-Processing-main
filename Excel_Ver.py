import cv2
import glob
import os
import csv
import numpy as np
from datetime import datetime
import platform

# matplotlib은 설치가 진행 중이므로 임포트를 방어적으로 처리
try:
    import matplotlib.pyplot as plt
    from matplotlib import rc
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from Filtering_Methods import progressive_mean_filter, progressive_median_filter, group_filter, myImFilter
from tester import calculate_metrics

def main():
    print("="*60)
    print("     초고속 벤치마크 데이터 추출 및 그래프 생성기     ")
    print("="*60)
    
    orig_path = "Results/lena.bmp"
    original_image = cv2.imread(orig_path)
    if original_image is None:
        print(f"에러: '{orig_path}' 파일을 찾을 수 없습니다.")
        return
        
    imgGray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    noise_images = glob.glob("Results/*noise*.bmp")
    if not noise_images:
        print("에러: 'Results' 폴더 내에 노이즈 이미지를 찾을 수 없습니다.")
        return

    os.makedirs("data_table", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"data_table/Benchmark_Results_{timestamp}.csv"
    
    # ------------------
    # 시각화용 데이터 수집
    # ------------------
    image_names = []
    filters = ["Base Filter (Mean)", "Progressive Median", "Group Filter"]
    psnr_data = {f: [] for f in filters}
    ssim_data = {f: [] for f in filters}
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Filter Name", "PSNR (dB)", "SSIM", "Route Count"])
        
        for img_path in noise_images:
            img_name = os.path.basename(img_path)
            image_names.append(img_name)
            print(f"\n[{img_name}] 데이터를 분석 중입니다. 잠시만 기다려주세요...")
            
            image = cv2.imread(img_path)
            SPnoise_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Base Filter (Mean)
            base_res, base_routes = myImFilter(SPnoise_img, param="mean", return_route=True)
            p, s = calculate_metrics(imgGray_original, base_res)
            writer.writerow([img_name, "Base Filter (Mean)", round(p, 2), round(s, 4), base_routes])
            psnr_data["Base Filter (Mean)"].append(p)
            ssim_data["Base Filter (Mean)"].append(s)
            
            # 2. Progressive Median
            med_res, med_routes = progressive_median_filter(SPnoise_img, return_route=True)
            p, s = calculate_metrics(imgGray_original, med_res)
            writer.writerow([img_name, "Progressive Median", round(p, 2), round(s, 4), med_routes])
            psnr_data["Progressive Median"].append(p)
            ssim_data["Progressive Median"].append(s)
            
            # 3. Group Filter
            grp_res, grp_routes = group_filter(SPnoise_img, return_route=True)
            p, s = calculate_metrics(imgGray_original, grp_res)
            writer.writerow([img_name, "Group Filter", round(p, 2), round(s, 4), grp_routes])
            psnr_data["Group Filter"].append(p)
            ssim_data["Group Filter"].append(s)
            
            print(f" -> 성공적으로 데이터 추출 완료!")

    # ------------------
    # 시각화 (Matplotlib)
    # ------------------
    if HAS_MATPLOTLIB:
        print("\n -> 통계 차트 그래픽 렌더링 중...")
        # 윈도우 한글 폰트 설정
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        x = np.arange(len(image_names)) 
        width = 0.2
        offsets = [-1, 0, 1]
        colors = ['#4C72B0', '#DD8452', '#55A868'] # 파랑, 주황, 초록
        
        # --- PSNR 플롯 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, f_name in enumerate(filters):
            bars = ax.bar(x + offsets[idx]*width, psnr_data[f_name], width, label=f_name, color=colors[idx])
            # 막대 위에 값 적어주기
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                            
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('테스트 이미지별 PSNR 성능 비교 차트')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names)
        # 범례를 그래프 밖으로 이동
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        psnr_png = f"data_table/PSNR_Comparison_{timestamp}.png"
        fig.savefig(psnr_png, dpi=150)
        plt.close(fig)
        
        # --- SSIM 플롯 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, f_name in enumerate(filters):
            bars = ax.bar(x + offsets[idx]*width, ssim_data[f_name], width, label=f_name, color=colors[idx])
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                            
        ax.set_ylabel('SSIM 점수')
        ax.set_title('테스트 이미지별 SSIM 성능 비교 차트')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        ssim_png = f"data_table/SSIM_Comparison_{timestamp}.png"
        fig.savefig(ssim_png, dpi=150)
        plt.close(fig)
        
        print("\n" + "="*60)
        print(f"모든 데이터 추출 및 그래프 렌더링이 완료되었습니다!")
        print(f"1. 표 형식 데이터 (CSV)    -> data_table/{os.path.basename(csv_filename)}")
        print(f"2. PSNR 그래프 사진 (.png) -> data_table/{os.path.basename(psnr_png)}")
        print(f"3. SSIM 그래프 사진 (.png) -> data_table/{os.path.basename(ssim_png)}")
        print("="*60)
    else:
        print("\n[경고] matplotlib이 설치 실패하여 이미지를 생성하지 못했습니다.")
        print(f"표 형식 데이터 (엑셀/CSV) 파일 하나만 저장되었습니다 -> {csv_filename}")

if __name__ == "__main__":
    main()
