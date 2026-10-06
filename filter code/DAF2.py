import cv2
import numpy as np
import os
import sys
import time

# --- 유니코드 경로를 지원하는 이미지 로드 ---
def imread_unicode(input_path, flags=cv2.IMREAD_GRAYSCALE):
    if not os.path.exists(input_path):
        return None
    try:
        data = np.fromfile(input_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception as e:
        print(f"이미지 로드 예외: {e}")
        return None


def safe_imwrite(output_path, image, desc="이미지"):
    try:
        result = cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"{desc} 저장 중 예외 발생: {e}")
        return False
    if not result:
        print(f"{desc} 저장 실패: {output_path}")
    return result

# --- 1. Detail-Aware (Rolling Guidance) 필터 알고리즘 정의 ---
def apply_detail_aware_filter(image, sigma_s=3.0, sigma_r=25.0, num_of_iter=4):
    """
    Rolling Guidance Filter (RGF)를 적용하여 구조는 보존하고 미세 질감(노이즈)을 제거하는 디테일 인지 필터.
    Args:
        image (np.ndarray): 입력 이미지 (Grayscale 권장)
        sigma_s (float): 공간 표준편차 (숫자가 클수록 더 넓은 범위를 참고하여 '구조'를 단순화함)
        sigma_r (float): 색상 표준편차 (숫자가 클수록 색 차이를 무시하고 뭉갬 -> 질감 제거 강도)
        num_of_iter (int): 반복 횟수 (디테일 복원의 정밀도를 결정. 보통 2~5 사이)
    Returns:
        np.ndarray: 필터링된 결과 이미지
    """
    # opencv-contrib-python의 ximgproc 모듈 사용
    # d=-1로 설정하면 sigma_s에 따라 자동으로 지름이 결정되어 더 정밀한 처리가 가능합니다.
    result = cv2.ximgproc.rollingGuidanceFilter(image, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s, numOfIter=num_of_iter)
    return result

# --- 2. 메인 실험 파이프라인 (로드 -> 처리 -> 저장 -> 시각화) ---
def run_detail_aware_experiment(input_path):
    # (1) 파일 존재 여부 확인 (방어적 프로그래밍)
    if not os.path.exists(input_path):
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # (2) 이미지 가져오기 (의료 영상 판독 표준인 Grayscale로 로드)
    src = imread_unicode(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        print(f"오류: 이미지 파일을 읽어오지 못했습니다. 파일 경로 또는 형식을 확인하세요. ({input_path})")
        return

    print("Detail-Aware 필터(Rolling Guidance) 작동 중... (시간이 다소 걸릴 수 있습니다)")
    
    # (3) 디테일 인지 필터 적용 (파라미터 설정)
    # sigma_s=3.0: 적당한 범위의 구조 유지
    # sigma_r=25.0: 미세 질감(노이즈)을 부드럽게 정돈
    # num_of_iter=4: 정밀한 반복 복원
    start_time = time.perf_counter()
    try:
        filtered_res = apply_detail_aware_filter(src, sigma_s=3.0, sigma_r=25.0, num_of_iter=4)
    except AttributeError as e:
        print("오류: OpenCV ximgproc 모듈을 사용할 수 없습니다. opencv-contrib-python이 설치되어 있는지 확인하세요.")
        print(f"세부 오류: {e}")
        return
    except Exception as e:
        print(f"필터 적용 중 오류가 발생했습니다: {e}")
        return
    filter_elapsed = time.perf_counter() - start_time

    if filtered_res is None:
        print("오류: 필터 결과가 생성되지 않았습니다.")
        return

    print(f"필터 실행 시간: {filter_elapsed:.6f} 초")
    print("필터링 완료. 결과를 저장하고 시각화합니다.")

    # (4) 결과 저장 폴더 생성 및 저장
    output_dir = "DAF"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(input_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{base_name}_detail_aware90.png")
    if not safe_imwrite(save_path, filtered_res, desc="Detail-Aware 결과 이미지"):
        return
    print(f"결과 저장 완료: {save_path}")

    # (5) 시각적 비교 (원본 vs 디테일 인지 결과)
    comparison = np.hstack((src, filtered_res))
    
    # GUI 창을 띄우지 않고 이미지로만 저장 (배치 처리용)
    # 비교 이미지 저장
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    if not safe_imwrite(comparison_path, comparison, desc="비교 이미지"):
        return
    print(f"비교 이미지 저장: {comparison_path}")
    
    # GUI 창 표시 (원한다면 주석을 풀 수 있습니다)
    # cv2.imshow("Original vs Detail-Aware (Rolling Guidance)", comparison)
    # print("아무 키나 누르면 프로그램이 종료됩니다.")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# --- 실행부 ---
# Dataset 디렉토리에서 이미지 파일들을 자동으로 찾아서 처리
if __name__ == "__main__":
    # 현재 디렉토리 기준으로 Dataset 폴더 경로
    current_dir = os.getcwd()
    print(f"현재 작업 디렉토리: {current_dir}")
    
    # dataset 폴더가 현재 디렉토리에 있으면 사용, 아니면 부모 디렉토리에서 찾기
    if os.path.exists("dataset"):
        dataset_dir = "dataset"
    elif os.path.exists("../dataset"):
        dataset_dir = "../dataset"
    else:
        # 절대 경로로 직접 지정
        dataset_dir = r"G:\내 드라이브\재품개발학기제\dataset"

    if not os.path.isdir(dataset_dir):
        print("경고: 기본 dataset 경로를 찾을 수 없습니다. 현재 작업 디렉토리를 기준으로 다시 검색합니다.")
        # 현재 디렉토리부터 부모 디렉토리까지 dataset 폴더 재탐색
        current = os.getcwd()
        for _ in range(5):
            candidate = os.path.join(current, "dataset")
            if os.path.isdir(candidate):
                dataset_dir = candidate
                break
            current = os.path.dirname(current)
    
    print(f"Dataset 경로: {os.path.abspath(dataset_dir)}\n")
    
    if not os.path.exists(dataset_dir):
        print(f"오류: {dataset_dir} 디렉토리를 찾을 수 없습니다.")
    else:
        # 지원하는 이미지 확장자
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(dataset_dir) 
                      if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(dataset_dir, f))]
        
        if not image_files:
            print(f"오류: {dataset_dir}에서 이미지 파일을 찾을 수 없습니다.")
        else:
            print(f"발견된 이미지 파일 ({len(image_files)}개):")
            for i, img_file in enumerate(image_files):
                print(f"  {i+1}. {img_file}")
            
            # 각 이미지 파일을 처리
            for img_file in image_files:
                print(f"\n{'='*50}")
                print(f"처리 중: {img_file}")
                print(f"{'='*50}")
                image_path = os.path.join(dataset_dir, img_file)
                run_detail_aware_experiment(image_path)