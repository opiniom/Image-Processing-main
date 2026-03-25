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
    
    # 기준값 k를 4로 고정합니다. (테스트용)
    k = 4 
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
            
            # Runtime Warning 방지 및 안전한 정상 픽셀들의 평균값 도출
            with np.errstate(all='ignore'):
                med_vals = np.nanmean(surr_to_update, axis=1)
            
            # 복구된 평균값을 대입
            img_filt[update_mask] = med_vals.astype(np.uint8)
            
            # 추가 변경이 발생했으므로 동일한 k(4) 값으로 다음 루트 진행
        else:
            # k=4 고정이므로, 이번 루트에서 변화가 없었다면 더 이상 살릴 수 있는 픽셀이 없으므로 종료합니다.
            print(f"[Median Filter] 알림: k={k} 조건에서 복구할 수 있는 노이즈가 더 이상 없어 {route_count}루트에서 종료합니다.")
            break

    return img_filt.astype(np.uint8)

def deviation_filter(A):
    print("\n[알림] 제안 필터 1: 편차 필터(Deviation Filter)를 시작합니다...")
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    
    while True:
        route_count += 1
        noise_mask = (img_filt == 0) | (img_filt == 255)
        
        if not np.any(noise_mask):
            print(f"[Deviation Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        print(f"[Deviation Filter] 진행 중: {route_count}번째 탐색 시작", flush=True)
        P = np.pad(img_filt, 1, mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(P, (3, 3)).reshape(m, n, 9)
        
        # 4개 그룹 생성 (M, N, 4, 3)
        groups = np.zeros((m, n, 4, 3), dtype=np.uint8)
        groups[:, :, 0, :] = windows[:, :, [0, 1, 3]] # 좌상단 {1, 2, 4}
        groups[:, :, 1, :] = windows[:, :, [1, 2, 5]] # 우상단 {2, 3, 6}
        groups[:, :, 2, :] = windows[:, :, [3, 6, 7]] # 좌하단 {4, 7, 8}
        groups[:, :, 3, :] = windows[:, :, [5, 7, 8]] # 우하단 {6, 8, 9}
        
        # 유효(정상) 픽셀 마스크
        valid_mask = (groups > 0) & (groups < 255)
        n_valid = np.sum(valid_mask, axis=3) # (M, N, 4)
        
        # 노이즈를 NaN으로 치환하여 편차 계산
        groups_float = groups.astype(np.float32)
        groups_float[~valid_mask] = np.nan
        
        with np.errstate(all='ignore'):
            g_max = np.nanmax(groups_float, axis=3)
            g_min = np.nanmin(groups_float, axis=3)
            # 노이즈가 없는 정상 픽셀들 사이의 편차 (C - A 와 동일)
            variance = g_max - g_min 
            
        # 정상 픽셀이 0개인 그룹은 편차가 999 (최악의 점수)
        variance = np.nan_to_num(variance, nan=999)
        
        # 페널티: 정상 픽셀 1개당 -1000점 (정상픽셀 수가 절대적 우선순위)
        penalty = -n_valid * 1000 + variance
        
        best_group_idx = np.argmin(penalty, axis=2)
        
        idx_m, idx_n = np.indices((m, n))
        best_groups = groups_float[idx_m, idx_n, best_group_idx]
        
        with np.errstate(all='ignore'):
            best_medians = np.nanmedian(best_groups, axis=2)
            
        # 정상 픽셀 중앙값이 존재하는 경우만 업데이트
        valid_medians_mask = ~np.isnan(best_medians)
        update_mask = noise_mask & valid_medians_mask
        
        prev_img = img_filt.copy()
        img_filt[update_mask] = best_medians[update_mask].astype(np.uint8)
        
        if np.array_equal(prev_img, img_filt) or route_count >= 30:
            print(f"[Deviation Filter] 알림: 더 이상 변화가 없거나 최대 루프에 도달하여 {route_count}루트에서 종료합니다.")
            break
            
    return img_filt.astype(np.uint8)


def group_filter(A):
    print("\n[알림] 제안 필터 2: 업그레이드된 그룹 필터(Group Filter)를 시작합니다...")
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    max_k = 7 # 3x3 탐색 실패 시 5x5, 7x7까지 확장
    
    while True:
        route_count += 1
        noise_mask = (img_filt == 0) | (img_filt == 255)
        
        if not np.any(noise_mask):
            print(f"[Group Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        print(f"[Group Filter] 진행 중: {route_count}번째 탐색 시작 (적응형 윈도우 3~7 적용)", flush=True)
        
        unresolved_mask = noise_mask.copy()
        new_values = np.zeros_like(img_filt, dtype=np.float32)
        new_values[:] = np.nan
        
        # 3, 5, 7로 반경을 확장해가며 풀리지 않은 노이즈 픽셀에만 적용
        for k in range(3, max_k + 2, 2):
            if not np.any(unresolved_mask):
                break
                
            pad_w = k // 2
            P = np.pad(img_filt, pad_w, mode='edge')
            
            windows = np.lib.stride_tricks.sliding_window_view(P, (k, k))
            windows = windows.reshape(m, n, k*k)
            
            c = k // 2
            idx_1d = np.arange(k*k)
            i_idx = idx_1d // k
            j_idx = idx_1d % k
            
            # WxW 모양의 4방향 그룹 마스킹 알고리즘
            up_mask = (i_idx <= c) & (idx_1d != c*k + c)
            right_mask = (j_idx >= c) & (idx_1d != c*k + c)
            down_mask = (i_idx >= c) & (idx_1d != c*k + c)
            left_mask = (j_idx <= c) & (idx_1d != c*k + c)
            dir_masks = [up_mask, right_mask, down_mask, left_mask]
            
            # 자원 절약을 위해 풀리지 않은 픽셀들만 추출
            unresolved_idx = np.nonzero(unresolved_mask)
            u_windows = windows[unresolved_idx]
            
            best_dir_medians = np.zeros(len(u_windows), dtype=np.float32)
            best_dir_medians[:] = np.nan
            min_noise_counts = np.full(len(u_windows), 99999, dtype=np.int32)
            
            for d_mask in dir_masks:
                d_wind = u_windows[:, d_mask]
                d_is_noise = (d_wind == 0) | (d_wind == 255)
                d_noise_count = np.sum(d_is_noise, axis=1)
                
                d_wind_float = d_wind.astype(np.float32)
                d_wind_float[d_is_noise] = np.nan
                with np.errstate(all='ignore'):
                    d_med = np.nanmedian(d_wind_float, axis=1)
                
                # 노이즈가 더 적고 최소 1개의 정상 픽셀이 있는 방향 채택
                better_mask = (d_noise_count < min_noise_counts) & ~np.isnan(d_med)
                min_noise_counts[better_mask] = d_noise_count[better_mask]
                best_dir_medians[better_mask] = d_med[better_mask]
                
            valid_found = ~np.isnan(best_dir_medians)
            resolved_m = unresolved_idx[0][valid_found]
            resolved_n = unresolved_idx[1][valid_found]
            new_values[resolved_m, resolved_n] = best_dir_medians[valid_found]
            unresolved_mask[resolved_m, resolved_n] = False
            
        update_mask = noise_mask & ~np.isnan(new_values)
        prev_img = img_filt.copy()
        img_filt[update_mask] = new_values[update_mask].astype(np.uint8)
        
        if np.array_equal(prev_img, img_filt) or route_count >= 30:
            print(f"[Group Filter] 알림: 변화가 없거나 최대 루프에 도달하여 {route_count}루트에서 종료합니다.")
            break
            
    return img_filt.astype(np.uint8)


# 필터링 유형을 결정하는 함수입니다.
def myImFilter(B,param):
    # 'mean'을 선택한 경우 평균 필터를 적용합니다.
    # (사용자 요청으로 인해 파라미터에 상관없이 무조건 mean 필터가 적용되도록 수정)
    print("\n[알림] 새로운 점진적 적응형 미디언 필터를 시작합니다...")
    C = median_filter(B)
    return C
