import numpy as np

def hybrid_filter(A, return_route=False, return_stats=False):
    print("\n[알림] 제안 필터 3: 하이브리드 필터(Hybrid Filter - HF)를 시작합니다...")
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    
    # 복원 통계용 카운터 (필터별 복원 픽셀 수 기록)
    stats = {'median': 0, 'mean3x3': 0, 'group3x3': 0, 'group5x5': 0}
    
    # 마스크 인덱스
    cross_idx = [1, 3, 5, 7]
    all_idx = [0, 1, 2, 3, 5, 6, 7, 8]
    dir_indices_3 = [
        [0, 1, 2, 3, 5],
        [1, 2, 5, 7, 8],
        [3, 5, 6, 7, 8],
        [0, 1, 3, 6, 7]
    ]
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
            print(f"[Hybrid Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        print(f"[Hybrid Filter] 진행 중: {route_count}번째 융합 탐색 시작", flush=True)
        
        unresolved_mask = noise_mask.copy()
        new_values = np.zeros_like(img_filt, dtype=np.float32)
        new_values[:] = np.nan
        
        # 1. 3x3 범위 추출 및 평가
        P3 = np.pad(img_filt, 1, mode='edge')
        windows3 = np.lib.stride_tricks.sliding_window_view(P3, (3, 3)).reshape(m, n, 9)
        unresolved_idx = np.nonzero(unresolved_mask)
        u_windows3 = windows3[unresolved_idx]
        
        is_noise3 = (u_windows3 == 0) | (u_windows3 == 255)
        u_wind3_float = u_windows3.astype(np.float32)
        u_wind3_float[is_noise3] = np.nan
        
        cross_normals = np.sum(~is_noise3[:, cross_idx], axis=1)
        total_normals = np.sum(~is_noise3[:, all_idx], axis=1)
        
        best_vals = np.zeros(len(u_windows3), dtype=np.float32)
        best_vals[:] = np.nan
        
        # 조건 1. 십자 정상값 >= 3 -> 십자 중앙값(Median) 연산
        cond1 = (cross_normals >= 3)
        if np.any(cond1):
            with np.errstate(all='ignore'):
                best_vals[cond1] = np.nanmedian(u_wind3_float[cond1][:, cross_idx], axis=1)
                
        # 조건 2. 전체 정상값 >= 3 -> 3x3 전체 평균(Mean) 연산
        cond2 = ~cond1 & (total_normals >= 3)
        if np.any(cond2):
            with np.errstate(all='ignore'):
                best_vals[cond2] = np.nanmean(u_wind3_float[cond2][:, all_idx], axis=1)
                
        # 조건 3. 전체 정상값 1~3 -> 3x3 방향성 그룹(Group Mean) 필터 연산
        cond3 = ~cond1 & ~cond2 & (total_normals >= 1)
        if np.any(cond3):
            c3_windows = u_windows3[cond3]
            c3_best_means = np.zeros(len(c3_windows), dtype=np.float32)
            c3_best_means[:] = np.nan
            c3_min_noise = np.full(len(c3_windows), 99999, dtype=np.int32)
            
            for d_idx in dir_indices_3:
                d_w = c3_windows[:, d_idx]
                d_is_n = (d_w == 0) | (d_w == 255)
                d_noise_c = np.sum(d_is_n, axis=1)
                d_w_f = d_w.astype(np.float32)
                d_w_f[d_is_n] = np.nan
                
                with np.errstate(all='ignore'):
                    d_m = np.nanmean(d_w_f, axis=1)
                
                better = (d_noise_c < c3_min_noise) & ~np.isnan(d_m)
                c3_min_noise[better] = d_noise_c[better]
                c3_best_means[better] = d_m[better]
                
            best_vals[cond3] = c3_best_means
            
        valid_found3 = ~np.isnan(best_vals)
        
        # 통계 누적: 실제로 값이 복원되는 픽셀들에 대해 어떤 조건이었는지 기록
        if return_stats:
            stats['median'] += int(np.sum(cond1 & valid_found3))
            stats['mean3x3'] += int(np.sum(cond2 & valid_found3))
            stats['group3x3'] += int(np.sum(cond3 & valid_found3))
            
        res_m3 = unresolved_idx[0][valid_found3]
        res_n3 = unresolved_idx[1][valid_found3]
        new_values[res_m3, res_n3] = best_vals[valid_found3]
        unresolved_mask[res_m3, res_n3] = False
        
        # 2. 모든 3x3 조건 실패(전체 정상값 == 0 등) -> 5x5 방향성 그룹 필터로 무조건 확장
        if np.any(unresolved_mask):
            P5 = np.pad(img_filt, 2, mode='edge')
            windows5 = np.lib.stride_tricks.sliding_window_view(P5, (5, 5)).reshape(m, n, 25)
            
            u_idx5 = np.nonzero(unresolved_mask)
            u_windows5 = windows5[u_idx5]
            
            c4_best_means = np.zeros(len(u_windows5), dtype=np.float32)
            c4_best_means[:] = np.nan
            c4_min_noise = np.full(len(u_windows5), 99999, dtype=np.int32)
            
            for d_idx in dir_indices_5:
                d_w = u_windows5[:, d_idx]
                d_is_n = (d_w == 0) | (d_w == 255)
                d_noise_c = np.sum(d_is_n, axis=1)
                d_w_f = d_w.astype(np.float32)
                d_w_f[d_is_n] = np.nan
                
                with np.errstate(all='ignore'):
                    d_m = np.nanmean(d_w_f, axis=1)
                
                better = (d_noise_c < c4_min_noise) & ~np.isnan(d_m)
                c4_min_noise[better] = d_noise_c[better]
                c4_best_means[better] = d_m[better]
                
            valid_found5 = ~np.isnan(c4_best_means)
            if return_stats:
                stats['group5x5'] += int(np.sum(valid_found5))
                
            res_m5 = u_idx5[0][valid_found5]
            res_n5 = u_idx5[1][valid_found5]
            new_values[res_m5, res_n5] = c4_best_means[valid_found5]
        
        # 이미지에 동시 적용
        update_mask = noise_mask & ~np.isnan(new_values)
        prev_img = img_filt.copy()
        img_filt[update_mask] = new_values[update_mask].astype(np.uint8)
        
        if np.array_equal(prev_img, img_filt) or route_count >= 30:
            print(f"[Hybrid Filter] 알림: 변화가 없거나 최대 루프에 도달하여 {route_count}루트에서 종료합니다.")
            break

    remains = np.any((img_filt == 0) | (img_filt == 255))
    final_routes = route_count if remains else route_count - 1
    
    # 반환 형식 조합
    if return_route and return_stats:
        return img_filt.astype(np.uint8), final_routes, stats
    elif return_route:
        return img_filt.astype(np.uint8), final_routes
    elif return_stats:
        return img_filt.astype(np.uint8), stats
        
    return img_filt.astype(np.uint8)
