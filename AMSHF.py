import numpy as np

def _apply_close_median(u_wind_float, indices):
    """중접(Close) Median 필터: 주어진 인덱스 영역 내 정상값들의 중앙값 산출"""
    vals = np.full(len(u_wind_float), np.nan, dtype=np.float32)
    with np.errstate(all='ignore'):
        # 각 행별로 특정 인덱스 영역의 중앙값 계산
        vals = np.nanmedian(u_wind_float[:, indices], axis=1)
    return vals

def _apply_group_mean(u_windows, indices, dir_indices):
    """방향성 그룹(Group Mean) 필터: 윈도우 평균과 가장 유사한 그룹 평균 선택"""
    # 1. 중심 윈도우의 전체 정상값 평균 산출
    is_n = (u_windows != 0) & (u_windows != 255)
    u_wind_f = u_windows.astype(np.float32)
    u_wind_f[~is_n] = np.nan
    with np.errstate(all='ignore'):
        win_mean = np.nanmean(u_wind_f[:, indices], axis=1)
    
    best_means = np.full(len(u_windows), np.nan, dtype=np.float32)
    min_diff = np.full(len(u_windows), np.inf, dtype=np.float32)
    
    for d_idx in dir_indices:
        d_w = u_windows[:, d_idx]
        d_is_n = (d_w != 0) & (d_w != 255)
        d_w_f = d_w.astype(np.float32)
        d_w_f[~d_is_n] = np.nan
        
        with np.errstate(all='ignore'):
            d_m = np.nanmean(d_w_f, axis=1)
            
        valid = ~np.isnan(d_m)
        diff = np.abs(d_m - win_mean)
        update = valid & (diff < min_diff)
        
        min_diff[update] = diff[update]
        best_means[update] = d_m[update]
        
    return best_means

def _apply_mean(u_wind_float, indices):
    """평균(Mean) 필터: 주어진 인덱스 영역 내 정상값들의 산술 평균 산출"""
    with np.errstate(all='ignore'):
        vals = np.nanmean(u_wind_float[:, indices], axis=1)
    return vals

def amshf_filter(A, return_route=False, return_stats=False, verbose=True, 
                 k_neu_th=2, k_moo_th=5, k_mean_th=2):
    """
    AMSHF (Adaptive Multi-stage Hybrid Filter) - Final Version
    Incorporates the Adaptive Hybrid Filter (AHF) logic.
    """
    if verbose:
        print(f"\n[알림] 필터 시작 (k_neu_th={k_neu_th}, k_moo_th={k_moo_th}, k_mean_th={k_mean_th})")
    
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    
    # 복원 통계용 카운터
    stats = {'median': 0, 'group': 0, 'mean': 0}
    
    # --- 인덱스 정의 ---
    # r=1 (3x3 window, center=4)
    neumann1 = [1, 3, 5, 7]
    moore1 = [0, 1, 2, 3, 5, 6, 7, 8]
    dir_indices_3 = [
        [0, 1, 2], # Top
        [2, 5, 8], # Right
        [6, 7, 8], # Bottom
        [0, 3, 6]  # Left
    ]
    
    # r=2 (5x5 window, center=12)
    neumann2 = [2, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 22]
    moore2 = [i for i in range(25) if i != 12]
    dir_indices_5 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],             # Up
        [2, 3, 4, 7, 8, 9, 13, 14, 17, 18, 19, 22, 23, 24],          # Right
        [10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],   # Down
        [0, 1, 2, 5, 6, 7, 10, 11, 15, 16, 17, 20, 21, 22]          # Left
    ]

    while True:
        route_count += 1
        noise_mask = (img_filt == 0) | (img_filt == 255)
        if not np.any(noise_mask):
            if verbose:
                print(f"[필터] 모든 노이즈 제거 완료. (총 루트 수: {route_count - 1})")
            break
            
        if verbose:
            print(f"[필터] 루트 {route_count} 시작 (반경 r=1~2 순차적 적용)", flush=True)
            
        restored_this_route = 0
        
        # r=1, 2 루프
        for r in [1, 2]:
            current_noise_mask = (img_filt == 0) | (img_filt == 255)
            if not np.any(current_noise_mask):
                break
                
            # 슬라이딩 윈도우 준비
            pad_w = r
            P = np.pad(img_filt, pad_w, mode='edge')
            k_size = 2 * r + 1
            windows = np.lib.stride_tricks.sliding_window_view(P, (k_size, k_size)).reshape(m, n, k_size*k_size)
            
            u_idx = np.nonzero(current_noise_mask)
            u_windows = windows[u_idx]
            u_wind_float = u_windows.astype(np.float32)
            is_noise = (u_wind_float == 0) | (u_wind_float == 255)
            u_wind_float[is_noise] = np.nan
            
            # 반경별 마스크 선택
            neu_idx = neumann1 if r == 1 else neumann2
            moo_idx = moore1 if r == 1 else moore2
            group_idx = dir_indices_3 if r == 1 else dir_indices_5
            
            # 샘플 개수(k) 계산
            neu_normals = np.sum(~is_noise[:, neu_idx], axis=1)
            moo_normals = np.sum(~is_noise[:, moo_idx], axis=1)
            
            new_vals = np.full(len(u_windows), np.nan, dtype=np.float32)
            
            # 1. Neumann Condition (k >= k_neu_th) -> Median
            cond1 = (neu_normals >= k_neu_th)
            if np.any(cond1):
                new_vals[cond1] = _apply_close_median(u_wind_float[cond1], neu_idx)
                stats['median'] += np.sum(~np.isnan(new_vals[cond1]))
            
            # 2. Moore Condition (k >= k_moo_th) -> Group Mean
            cond2 = ~cond1 & (moo_normals >= k_moo_th)
            if np.any(cond2):
                new_vals[cond2] = _apply_group_mean(u_windows[cond2], moo_idx, group_idx)
                stats['group'] += np.sum(~np.isnan(new_vals[cond2]))
                
            # 3. Else -> Mean (k >= k_mean_th)
            cond3 = ~cond1 & ~cond2 & (moo_normals >= k_mean_th)
            if np.any(cond3):
                new_vals[cond3] = _apply_mean(u_wind_float[cond3], moo_idx)
                stats['mean'] += np.sum(~np.isnan(new_vals[cond3]))
                
            # 값 적용
            valid_mask = ~np.isnan(new_vals)
            res_m = u_idx[0][valid_mask]
            res_n = u_idx[1][valid_mask]
            
            img_filt[res_m, res_n] = new_vals[valid_mask].astype(np.uint8)
            restored_this_route += np.sum(valid_mask)

        if restored_this_route == 0 or route_count >= 10:
            if verbose:
                print(f"[필터] 조기 종료 혹은 최대 루프 도달 (루트 {route_count})")
            break

    remains = np.any((img_filt == 0) | (img_filt == 255))
    final_routes = route_count if remains else route_count - 1
    
    if return_route and return_stats:
        return img_filt.astype(np.uint8), final_routes, stats
    elif return_route:
        return img_filt.astype(np.uint8), final_routes
    elif return_stats:
        return img_filt.astype(np.uint8), stats
    
    return img_filt.astype(np.uint8)
