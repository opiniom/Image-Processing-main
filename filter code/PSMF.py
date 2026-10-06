import cv2
import numpy as np
import os

def apply_psmf(image, max_iter=5, k_size=3):
    """
    Progressive Switching Median Filter (PSMF)
    점진적으로 잡음을 제거하여 고밀도 임펄스 노이즈에 특화된 필터
    """
    height, width = image.shape
    curr_img = np.copy(image).astype(np.float32)
    
    # 1. 초기 잡음 맵 생성 (0 또는 255이면 잡음 후보 1, 아니면 0)
    # flag_map: 1이면 잡음(미복원), 0이면 정상(또는 복원완료)
    flag_map = np.zeros((height, width), dtype=np.uint8)
    flag_map[(image == 0) | (image == 255)] = 1
    
    pad = k_size // 2
    
    for iteration in range(max_iter):
        new_img = np.copy(curr_img)
        new_flag_map = np.copy(flag_map)
        changed = False
        
        # 패딩 처리
        padded_img = cv2.copyMakeBorder(curr_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        padded_flag = cv2.copyMakeBorder(flag_map, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=1)
        
        for y in range(height):
            for x in range(width):
                if flag_map[y, x] == 1: # 잡음인 픽셀만 처리
                    # 현재 윈도우 내의 픽셀과 플래그 추출
                    window = padded_img[y:y+k_size, x:x+k_size]
                    flag_window = padded_flag[y:y+k_size, x:x+k_size]
                    
                    # '정상 픽셀(flag == 0)'만 골라내기
                    valid_pixels = window[flag_window == 0]
                    
                    if len(valid_pixels) > 0:
                        # 정상 픽셀들의 중간값으로 교체하고 복원 완료 표시
                        new_img[y, x] = np.median(valid_pixels)
                        new_flag_map[y, x] = 0
                        changed = True
        
        curr_img = new_img
        flag_map = new_flag_map
        
        # 더 이상 복원할 픽셀이 없으면 조기 종료
        if not changed or np.sum(flag_map) == 0:
            break
            
    return np.clip(curr_img, 0, 255).astype(np.uint8)

# --- 메인 실행 파이프라인 ---

def run_psmf_experiment(input_path):
    if not os.path.exists(input_path):
        print(f"오류: {input_path} 파일을 찾을 수 없습니다.")
        return

    # 1. 이미지 가져오기
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None: return

    # 2. PSMF 적용
    print("PSMF 알고리즘 가동 중 (점진적 복원)...")
    res_psmf = apply_psmf(src, max_iter=10, k_size=3)

    # 3. 결과 저장 및 폴더 관리
    output_dir = "PSMF"
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = os.path.basename(input_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{file_name}_psmf.png")
    cv2.imwrite(save_path, res_psmf)
    
    print(f"복원 완료! 결과가 저장되었습니다: {save_path}")

    # 4. 시각적 확인 (Numpy hstack으로 비교)
    comparison = np.hstack((src, res_psmf))
    cv2.imshow("Original Noisy vs PSMF Result", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행 예시
run_psmf_experiment("LenaNZ90.png")