import numpy as np
from Utils import *


def mean_filter(img,size):
    # 커널 행렬을 0으로 채웁니다.
    B = np.zeros((size, size),dtype=np.float32)
    # 행렬의 모든 지점에 주어진 값 1/size^2를 적용합니다.
    B.fill(1 / (size) ** 2)
    img_filt = myConv2D(img,B,strides=1,param="same")

    return (img_filt.astype(np.uint8))


"""이미지를 순회합니다. 모든 3X3 영역에 대해,
     픽셀들의 중간값을 찾고
     중앙 픽셀을 그 중간값으로 대체합니다."""
def median_filter(A):
    m, n = A.shape
    img_filt = A.copy()
    
    # 1루트 기준값은 6으로 시작합니다.
    k = 6 
    route_count = 0
    
    while True:
        route_count += 1
        
        # 현재 영상에 남아있는 노이즈(0 또는 255) 마스크 생성
        noise_mask = (img_filt == 0) | (img_filt == 255)
        
        # Q4 반영: 노이즈가 아예 없다면 반복을 완전히 종료합니다.
        if not np.any(noise_mask):
            print(f"[Median Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        # 진행 상황을 알 수 있도록 터미널에 현재 상태 출력 (flush=True로 즉시 출력)
        print(f"[Median Filter] 진행 중: {route_count}번째 루트 탐색 시작 (현재 정상 픽셀 기준 k={k})", flush=True)
            
        # Q3 반영: Padding 처리 (가장자리 픽셀 복제)
        P = np.pad(img_filt, 1, mode='edge')
        
        # 넘파이(NumPy) 벡터화를 통한 100배 이상의 연산 속도 향상(Sliding Window View 이용)
        # 패딩된 이미지에서 모든 가장자리를 활용해 3x3 배열 생성 (M, N, 3, 3) -> 평탄화 -> (M, N, 9)
        windows = np.lib.stride_tricks.sliding_window_view(P, (3, 3)).reshape(m, n, 9)
        
        # 중앙 픽셀(인덱스 4) 제거 (M, N, 8)
        surrounding = np.delete(windows, 4, axis=2)
        
        # 노이즈가 아닌 정상 픽셀 마스크
        normal_mask = (surrounding != 0) & (surrounding != 255)
        
        # 타겟 조건: 현재 픽셀이 노이즈이면서 정상 픽셀의 수가 k개 이상인 위치 검색
        update_mask = noise_mask & (np.sum(normal_mask, axis=2) >= k)
        
        if np.any(update_mask):
            # 조건을 만족시키는 픽셀의 8개 이웃 픽셀들 추출 -> 배열 (X, 8)
            surr_to_update = surrounding[update_mask].astype(np.float32)
            mask_to_update = normal_mask[update_mask]
            
            # 노이즈인 곳을 계산에서 배제하기 위해 NaN으로 처리
            surr_to_update[~mask_to_update] = np.nan
            
            # Runtime Warning 방지 및 안전한 정상 픽셀들의 중앙값 도출
            with np.errstate(all='ignore'):
                med_vals = np.nanmedian(surr_to_update, axis=1)
            
            # 복구된 중앙값을 대입
            img_filt[update_mask] = med_vals.astype(np.uint8)
            
            # 추가 변경이 발생했으므로 k 값은 깎지 않고 유지합니다.
        else:
            if k > 1:
                k -= 1
            else:
                # k가 1까지 내려왔음에도 더 이상 바뀔 노이즈가 없다면 무한루프 방지
                print(f"[Median Filter] 경고: 주변에 0/255만 있어 더이상 노이즈를 완전히 지울 수 없습니다. {route_count}루트에서 조기 종료합니다.")
                break

    return img_filt.astype(np.uint8)




# 필터링 유형을 결정하는 함수입니다.
def myImFilter(B,param):
    # 'mean'을 선택한 경우 평균 필터를 적용합니다.
    # (사용자 요청으로 인해 파라미터에 상관없이 무조건 mean 필터가 적용되도록 수정)
    print("\n[알림] 새로운 점진적 적응형 미디언 필터를 시작합니다...")
    C = median_filter(B)
    return C
