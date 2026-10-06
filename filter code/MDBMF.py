import cv2
import numpy as np
import os

def apply_mdbmf(image, k_size=3):
    """
    Modified Decision-Based Median Filter (MDBMF)
    솔트 앤 페퍼 잡음 제거의 강력한 벤치마크 알고리즘
    """
    output = np.copy(image)
    pad = k_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x]
            
            # [Detection] 0 또는 255인 잡음일 때만 동작
            if pixel == 0 or pixel == 255:
                window = padded[y:y+k_size, x:x+k_size]
                
                # 윈도우 내에서 잡음이 아닌 정상 픽셀만 골라냄
                valid_pixels = window[(window > 0) & (window < 255)]
                
                if len(valid_pixels) > 0:
                    # 정상 픽셀들의 중간값으로 교체
                    output[y, x] = np.median(valid_pixels)
                else:
                    # 모든 윈도우가 잡음이면? 이전 픽셀 값(혹은 주변 평균) 사용
                    # 이것이 MDBMF가 고밀도 잡음에 강한 이유입니다.
                    output[y, x] = output[y, x-1] if x > 0 else pixel
            
    return output

# --- 메인 파이프라인 (로드 -> 처리 -> 저장) ---

def run_benchmark(input_path):
    if not os.path.exists(input_path):
        print("파일을 찾을 수 없습니다.")
        return

    # 1. 이미지 가져오기
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. 필터 적용
    print("MDBMF 필터 적용 중 (솔트 앤 페퍼 특화)...")
    res_mdbmf = apply_mdbmf(src)

    # 3. 저장 및 폴더 관리
    out_dir = "MDBMF"
    os.makedirs(out_dir, exist_ok=True)
    
    base = os.path.basename(input_path).split('.')[0]
    save_path = os.path.join(out_dir, f"{base}_mdbmf.png")
    
    cv2.imwrite(save_path, res_mdbmf)
    print(f"결과 저장 완료: {save_path}")

    # 4. 시각화
    cv2.imshow("MDBMF Result", np.hstack((src, res_mdbmf)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행
run_benchmark("LenaNZ90.png")