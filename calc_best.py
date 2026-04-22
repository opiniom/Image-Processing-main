import os
import glob
import csv
from collections import defaultdict

def main():
    files = glob.glob('data_table/Tuning_*.csv')
    if not files:
        print("No Tuning CSV files found.")
        return

    # combo -> {'psnr': sum, 'ssim': sum, 'count': n}
    combo_stats = defaultdict(lambda: {'psnr': 0.0, 'ssim': 0.0, 'count': 0})
    
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                c1 = int(row['cond1_thresh'])
                c2 = int(row['cond2_thresh'])
                c3 = int(row['cond3_thresh'])
                psnr = float(row['Avg PSNR (dB)'])
                ssim = float(row['Avg SSIM'])
                
                combo_stats[(c1, c2, c3)]['psnr'] += psnr
                combo_stats[(c1, c2, c3)]['ssim'] += ssim
                combo_stats[(c1, c2, c3)]['count'] += 1

    results = []
    for (c1, c2, c3), stats in combo_stats.items():
        if stats['count'] == len(files):
            avg_psnr = stats['psnr'] / stats['count']
            avg_ssim = stats['ssim'] / stats['count']
            results.append(((c1, c2, c3), avg_psnr, avg_ssim))

    if not results:
        print("No common combinations across all files.")
        return

    results_psnr = sorted(results, key=lambda x: x[1], reverse=True)
    results_ssim = sorted(results, key=lambda x: x[2], reverse=True)

    print(f"Total analyzed files: {len(files)}")
    print([os.path.basename(f) for f in files])
    print("\n--- Top 5 by Average PSNR ---")
    for combo, psnr, ssim in results_psnr[:5]:
        print(f"cond1={combo[0]}, cond2={combo[1]}, cond3={combo[2]} -> PSNR: {psnr:.3f}, SSIM: {ssim:.4f}")

    print("\n--- Top 5 by Average SSIM ---")
    for combo, psnr, ssim in results_ssim[:5]:
        print(f"cond1={combo[0]}, cond2={combo[1]}, cond3={combo[2]} -> PSNR: {psnr:.3f}, SSIM: {ssim:.4f}")

if __name__ == '__main__':
    main()
