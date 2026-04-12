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
from Hybrid_Filter import hybrid_filter
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
    filters = ["Hybrid Filter (HF)"] # 최적화를 위해 HF만 남김
    # filters = ["Base Filter (Mean)", "Progressive Median", "Group Filter", "Hybrid Filter (HF)"]
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
            
            # 최적화를 위해 다른 필터들은 임시로 비활성화
            # # 1. Base Filter (Mean)
            # base_res, base_routes = myImFilter(SPnoise_img, 'progressive_mean', return_route=True)
            # p, s = calculate_metrics(imgGray_original, base_res)
            # writer.writerow([img_name, "Base Filter (Mean)", round(p, 2), round(s, 4), base_routes])
            # psnr_data["Base Filter (Mean)"].append(p)
            # ssim_data["Base Filter (Mean)"].append(s)
            
            # # 2. Progressive Median
            # f1_res, f1_routes = progressive_median_filter(SPnoise_img, return_route=True)
            # p, s = calculate_metrics(imgGray_original, f1_res)
            # writer.writerow([img_name, "Progressive Median", round(p, 2), round(s, 4), f1_routes])
            # psnr_data["Progressive Median"].append(p)
            # ssim_data["Progressive Median"].append(s)
            
            # # 3. Group Filter
            # grp_res, grp_routes = group_filter(SPnoise_img, return_route=True)
            # p, s = calculate_metrics(imgGray_original, grp_res)
            # writer.writerow([img_name, "Group Filter", round(p, 2), round(s, 4), grp_routes])
            # psnr_data["Group Filter"].append(p)
            # ssim_data["Group Filter"].append(s)
            
            # 4. Hybrid Filter (HF)
            hf_res, hf_routes = hybrid_filter(SPnoise_img, return_route=True)
            p, s = calculate_metrics(imgGray_original, hf_res)
            writer.writerow([img_name, "Hybrid Filter (HF)", round(p, 2), round(s, 4), hf_routes])
            psnr_data["Hybrid Filter (HF)"].append(p)
            ssim_data["Hybrid Filter (HF)"].append(s)
            
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
        width = 0.4
        offsets = [0]
        colors = ['#C44E52'] # 빨강 (HF 단일)
        
        # --- PSNR 플롯 ---
        fig, ax = plt.subplots(figsize=(10, 8))
        for idx, f_name in enumerate(filters):
            bars = ax.barh(x + offsets[idx]*width, psnr_data[f_name], height=width, label=f_name, color=colors[idx])
            for bar in bars:
                width_val = bar.get_width()
                ax.annotate(f'{width_val:.2f}',
                            xy=(width_val, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),  # 3 points horizontal offset
                            textcoords="offset points",
                            ha='left', va='center', fontsize=8)
                            
        ax.set_xlabel('PSNR (dB)')
        ax.set_title('테스트 이미지별 PSNR 성능 비교 차트')
        ax.set_yticks(x)
        ax.set_yticklabels(image_names)
        
        # 텍스트가 잘리지 않도록 X축(PSNR 점수축)을 +2 정도 여유있게 늘려줌
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + 2)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        psnr_png = f"data_table/PSNR_Comparison_{timestamp}.png"
        fig.savefig(psnr_png, dpi=150)
        plt.close(fig)
        
        # --- SSIM 플롯 ---
        fig, ax = plt.subplots(figsize=(10, 8))
        for idx, f_name in enumerate(filters):
            bars = ax.barh(x + offsets[idx]*width, ssim_data[f_name], height=width, label=f_name, color=colors[idx])
            for bar in bars:
                width_val = bar.get_width()
                ax.annotate(f'{width_val:.4f}',
                            xy=(width_val, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=8)
                            
        ax.set_xlabel('SSIM 점수')
        ax.set_title('테스트 이미지별 SSIM 성능 비교 차트')
        ax.set_yticks(x)
        ax.set_yticklabels(image_names)
        
        # SSIM 점수축은 최대 스케일이 1이므로 비율에 맞춰 +0.1 여유를 줌 (약 2글자 공간)
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + 0.1)
        
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
