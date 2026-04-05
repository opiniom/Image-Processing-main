import numpy as np
from Utils import *


def mean_filter(img,size):
    # 커널 행렬을 0으로 채웁니다.
    B = np.zeros((size, size),dtype=np.float32)
    # 행렬의 모든 지점에 주어진 값 1/size^2를 적용합니다.
    B.fill(1 / (size) ** 2)
    img_filt = myConv2D(img,B,strides=1,param="same")

    return (img_filt.astype(np.uint8))


"""노이즈 픽셀에 대해 3X3 영역의 정상 픽셀들의 평균값을 계산해 대체하고,
   정상 픽셀이 없으면 5X5 영역으로 탐색 범위를 확장하는 점진적 평균 필터입니다."""
def progressive_mean_filter(A):
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    max_k = 5 # 3x3, 노이즈가 많으면 5x5로 확장
    
    while True:
        route_count += 1
        noise_mask = (img_filt == 0) | (img_filt == 255)
        
        if not np.any(noise_mask):
            print(f"[Progressive Mean Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        print(f"[Progressive Mean Filter] 진행 중: {route_count}번째 탐색 시작 (적응형 윈도우 3~5 적용)", flush=True)
        
        unresolved_mask = noise_mask.copy()
        new_values = np.zeros_like(img_filt, dtype=np.float32)
        new_values[:] = np.nan
        
        # 3x3에서 정상픽셀이 없으면 5x5로 탐색 확장
        for k in range(3, max_k + 2, 2):
            if not np.any(unresolved_mask):
                break
                
            pad_w = k // 2
            P = np.pad(img_filt, pad_w, mode='edge')
            
            windows = np.lib.stride_tricks.sliding_window_view(P, (k, k))
            windows = windows.reshape(m, n, k*k)
            
            unresolved_idx = np.nonzero(unresolved_mask)
            u_windows = windows[unresolved_idx]
            
            is_noise = (u_windows == 0) | (u_windows == 255)
            u_windows_float = u_windows.astype(np.float32)
            u_windows_float[is_noise] = np.nan
            
            req = max(6 - route_count, 3) if k == 3 else max(19 - route_count, 12)
            normal_counts = np.sum(~is_noise, axis=1)
            not_enough_normals = normal_counts < req
            
            with np.errstate(all='ignore'):
                k_mean = np.nanmean(u_windows_float, axis=1)
                
            k_mean[not_enough_normals] = np.nan
            valid_found = ~np.isnan(k_mean)
            
            resolved_m = unresolved_idx[0][valid_found]
            resolved_n = unresolved_idx[1][valid_found]
            new_values[resolved_m, resolved_n] = k_mean[valid_found]
            unresolved_mask[resolved_m, resolved_n] = False
            
        update_mask = noise_mask & ~np.isnan(new_values)
        prev_img = img_filt.copy()
        img_filt[update_mask] = new_values[update_mask].astype(np.uint8)
        
        if np.array_equal(prev_img, img_filt) or route_count >= 30:
            print(f"[Progressive Mean Filter] 알림: 변화가 없거나 최대 루프에 도달하여 {route_count}루트에서 종료합니다.")
            break
            
    return img_filt.astype(np.uint8)


"""십자 방향(상/하/좌/우) 가장 인접한 픽셀의 정상값을 최우선 탐색하고,
   없을 시 3x3 전체 반경(8방향)으로 확장 탐색하는 공간 최적화 점진적 미디안 필터입니다."""
def progressive_median_filter(A):
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    
    while True:
        route_count += 1
        noise_mask = (img_filt == 0) | (img_filt == 255)
        
        if not np.any(noise_mask):
            print(f"[Cross-to-3x3 Median Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        print(f"[Cross-to-3x3 Median Filter] 진행 중: {route_count}번째 탐색 시작 (우선순위: 십자 -> 3x3)", flush=True)
        
        P = np.pad(img_filt, 1, mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(P, (3, 3)).reshape(m, n, 9)
        
        unresolved_idx = np.nonzero(noise_mask)
        u_windows = windows[unresolved_idx] # (N, 9)
        
        # 십자 모양 픽셀 마스크: 인덱스 1(상), 3(좌), 5(우), 7(하)
        cross_indices = [1, 3, 5, 7]
        # 전체 3x3 픽셀 마스크: 인덱스 4를 제외한 이웃들
        all_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        
        # 단계 1: 십자 모양 검사
        cross_windows = u_windows[:, cross_indices]
        is_noise_cross = (cross_windows == 0) | (cross_windows == 255)
        cross_float = cross_windows.astype(np.float32)
        cross_float[is_noise_cross] = np.nan
        
        with np.errstate(all='ignore'):
            cross_median = np.nanmedian(cross_float, axis=1)
            
        # 단계 2: 3x3 전체 검사 (십자가 실패한 곳)
        all_windows = u_windows[:, all_indices]
        is_noise_all = (all_windows == 0) | (all_windows == 255)
        all_float = all_windows.astype(np.float32)
        all_float[is_noise_all] = np.nan
        
        with np.errstate(all='ignore'):
            all_median = np.nanmedian(all_float, axis=1)
            
        # 값 할당 로직: 십자 모양에서 정상값이 하나라도 있으면 cross_median, 없으면 all_median
        valid_cross = ~np.isnan(cross_median)
        
        # 십자(Cross)를 최우선 적용
        final_medians = np.where(valid_cross, cross_median, all_median)
        valid_found = ~np.isnan(final_medians)
        
        resolved_m = unresolved_idx[0][valid_found]
        resolved_n = unresolved_idx[1][valid_found]
        
        prev_img = img_filt.copy()
        img_filt[resolved_m, resolved_n] = final_medians[valid_found].astype(np.uint8)
        
        if np.array_equal(prev_img, img_filt) or route_count >= 30:
            print(f"[Cross-to-3x3 Median Filter] 알림: 변화가 없거나 최대 루프에 도달하여 {route_count}루트에서 종료합니다.")
            break
            
    return img_filt.astype(np.uint8)

def group_filter(A):
    print("\n[알림] 제안 필터 2: 업그레이드된 방향성 지향 그룹 필터(Directional Group Filter)를 시작합니다...")
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    
    while True:
        route_count += 1
        noise_mask = (img_filt == 0) | (img_filt == 255)
        
        if not np.any(noise_mask):
            print(f"[Group Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        print(f"[Group Filter] 진행 중: {route_count}번째 탐색 시작 (상하좌우 방향성 그룹 검사)", flush=True)
        
        P = np.pad(img_filt, 1, mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(P, (3, 3)).reshape(m, n, 9)
        
        unresolved_idx = np.nonzero(noise_mask)
        u_windows = windows[unresolved_idx] # (N, 9)
        
        # 3x3 그룹 마스크
        # 0 1 2
        # 3 4 5
        # 6 7 8
        up_idx    = [0, 1, 2, 3, 5]
        right_idx = [1, 2, 5, 7, 8]
        down_idx  = [3, 5, 6, 7, 8]
        left_idx  = [0, 1, 3, 6, 7]
        
        dir_indices = [up_idx, right_idx, down_idx, left_idx]
        
        best_dir_medians = np.zeros(len(u_windows), dtype=np.float32)
        best_dir_medians[:] = np.nan
        min_noise_counts = np.full(len(u_windows), 99999, dtype=np.int32)
        
        for d_idx in dir_indices:
            d_wind = u_windows[:, d_idx]
            d_is_noise = (d_wind == 0) | (d_wind == 255)
            d_noise_count = np.sum(d_is_noise, axis=1) # 노이즈 개수 계산 (적을수록 정상값이 많음을 의미)
            
            d_wind_float = d_wind.astype(np.float32)
            d_wind_float[d_is_noise] = np.nan
            
            with np.errstate(all='ignore'):
                d_med = np.nanmedian(d_wind_float, axis=1)
                
            # 정상값이 더 많은 방향(노이즈가 더 적은 방향)으로 채택
            better_mask = (d_noise_count < min_noise_counts) & ~np.isnan(d_med)
            min_noise_counts[better_mask] = d_noise_count[better_mask]
            best_dir_medians[better_mask] = d_med[better_mask]
            
        valid_found = ~np.isnan(best_dir_medians)
        resolved_m = unresolved_idx[0][valid_found]
        resolved_n = unresolved_idx[1][valid_found]
        
        prev_img = img_filt.copy()
        img_filt[resolved_m, resolved_n] = best_dir_medians[valid_found].astype(np.uint8)
        
        if np.array_equal(prev_img, img_filt) or route_count >= 30:
            print(f"[Group Filter] 알림: 변화가 없거나 최대 루프에 도달하여 {route_count}루트에서 종료합니다.")
            break
            
    return img_filt.astype(np.uint8)


# 필터링 유형을 결정하는 함수입니다.
def myImFilter(B,param):
    print("\n[알림] 점진적 평균 필터(Progressive Mean Filter)를 Base Filter로 시작합니다...")
    C = progressive_mean_filter(B)
    return C
