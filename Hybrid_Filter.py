import numpy as np

def hybrid_filter(A, return_route=False, return_stats=False,
                  cond1_thresh=2,   # 십자 Median 발동 기준 (십자 4칸 중 정상값 수, 권장: 2~4)
                  cond2_thresh=5,   # Group Mean 발동 기준 (3x3 전체 영역 내 정상값 수, 권장: 4~6)
                  cond3_thresh=1,   # 전체 Mean 발동 기준 (전체 8칸 중 정상값 수, 권장: 1~3)
                  verbose=True):    # False시 루트별 진행 로그를 출력하지 않음
    if verbose:
        print("\n[알림] 제안 필터 3: 하이브리드 필터(Hybrid Filter - HF)를 시작합니다...")
    m, n = A.shape
    img_filt = A.copy()
    route_count = 0
    
    # 복원 통계용 카운터
    stats = {'median': 0, 'group3x3': 0, 'mean': 0, 'mean5x5': 0}
    
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
            if verbose:
                print(f"[Hybrid Filter] 모든 노이즈 제거 완료. (총 가동된 루트 수: {route_count - 1})")
            break
            
        if verbose:
            print(f"[Hybrid Filter] 진행 중: {route_count}번째 융합 탐색 시작", flush=True)
        prev_noise_count = int(np.sum(noise_mask))
        
        unresolved_mask = noise_mask.copy()
        new_values = np.full((m, n), np.nan, dtype=np.float32)  # 단순화: np.full() 사용
        
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
        
        # 그룹별 정상 픽셀 수 계산 (사용자 요청: 한 그룹이라도 정상 픽셀이 4개 이상이면 발동)
        group_normals = np.stack([np.sum(~is_noise3[:, d_idx], axis=1) for d_idx in dir_indices_3], axis=0)
        max_group_normal = np.max(group_normals, axis=0)
        
        best_vals = np.full(len(u_windows3), np.nan, dtype=np.float32)  # 단순화: np.full() 사용
        
        # 조건 1. 십자 정상값 >= cond1_thresh -> 십자 중앙값(Median) 연산
        cond1 = (cross_normals >= cond1_thresh)
        if np.any(cond1):
            with np.errstate(all='ignore'):
                best_vals[cond1] = np.nanmedian(u_wind3_float[cond1][:, cross_idx], axis=1)
                
        # 조건 2. 전체 정상값 >= cond2_thresh -> 3x3 방향성 그룹(Group Mean) 필터 연산
        cond2 = ~cond1 & (total_normals >= cond2_thresh)
        if np.any(cond2):
            c2_windows = u_windows3[cond2]
            c2_best_means = np.full(len(c2_windows), np.nan, dtype=np.float32)
            c2_min_noise = np.full(len(c2_windows), 99999, dtype=np.int32)
            c2_best_var = np.full(len(c2_windows), np.inf, dtype=np.float32)  # 타이브레이커: 정상값 분산
            
            for d_idx in dir_indices_3:
                d_w = c2_windows[:, d_idx]
                d_is_n = (d_w == 0) | (d_w == 255)
                d_noise_c = np.sum(d_is_n, axis=1)
                d_w_f = d_w.astype(np.float32)
                d_w_f[d_is_n] = np.nan
                
                with np.errstate(all='ignore'):
                    d_m = np.nanmean(d_w_f, axis=1)
                    d_var = np.nanvar(d_w_f, axis=1)  # 정상값들의 분산
                
                valid = ~np.isnan(d_m)
                # 1순위: 노이즈 수가 더 적은 방향
                strictly_better = valid & (d_noise_c < c2_min_noise)
                # 2순위(타이): 노이즈 수 동일 + 정상값 분산이 더 낮은 방향
                tie_better = valid & (d_noise_c == c2_min_noise) & (d_var < c2_best_var)
                update = strictly_better | tie_better
                c2_min_noise[update] = d_noise_c[update]
                c2_best_means[update] = d_m[update]
                c2_best_var[update] = d_var[update]
                
            best_vals[cond2] = c2_best_means
                
        # 조건 3. 전체 정상값 >= cond3_thresh -> 3x3 전체 평균(Mean) 연산
        cond3 = ~cond1 & ~cond2 & (total_normals >= cond3_thresh)
        if np.any(cond3):
            with np.errstate(all='ignore'):
                best_vals[cond3] = np.nanmean(u_wind3_float[cond3][:, all_idx], axis=1)
                
        valid_found3 = ~np.isnan(best_vals)
        
        # 통계 누적: 실제로 값이 복원되는 픽셀들에 대해 어떤 조건이었는지 기록
        if return_stats:
            stats['median'] += int(np.sum(cond1 & valid_found3))
            stats['group3x3'] += int(np.sum(cond2 & valid_found3))
            stats['mean'] += int(np.sum(cond3 & valid_found3))

        res_m3 = unresolved_idx[0][valid_found3]
        res_n3 = unresolved_idx[1][valid_found3]
        new_values[res_m3, res_n3] = best_vals[valid_found3]
        unresolved_mask[res_m3, res_n3] = False
        
        # 2. 모든 3x3 조건 실패 -> 5x5 전체 평균(Mean) 필터로 확장
        if np.any(unresolved_mask):
            P5 = np.pad(img_filt, 2, mode='edge')
            windows5 = np.lib.stride_tricks.sliding_window_view(P5, (5, 5)).reshape(m, n, 25)
            
            u_idx5 = np.nonzero(unresolved_mask)
            u_windows5 = windows5[u_idx5]
            
            # 5x5 영역의 부동소수점 변환 및 노이즈 마스킹
            u_wind5_float = u_windows5.astype(np.float32)
            is_noise5 = (u_wind5_float == 0) | (u_wind5_float == 255)
            u_wind5_float[is_noise5] = np.nan
            
            with np.errstate(all='ignore'):
                c4_mean_vals = np.nanmean(u_wind5_float, axis=1)
            
            valid_found5 = ~np.isnan(c4_mean_vals)
            if return_stats:
                stats['mean5x5'] += int(np.sum(valid_found5))
                
            res_m5 = u_idx5[0][valid_found5]
            res_n5 = u_idx5[1][valid_found5]
            new_values[res_m5, res_n5] = c4_mean_vals[valid_found5]
        
        # 이미지에 동시 적용
        update_mask = noise_mask & ~np.isnan(new_values)
        prev_img = img_filt.copy()
        img_filt[update_mask] = new_values[update_mask].astype(np.uint8)
        
        # 복원된 픽셀 수 계산
        curr_noise_count = int(np.sum((img_filt == 0) | (img_filt == 255)))
        restored_count = prev_noise_count - curr_noise_count
        
        if np.array_equal(prev_img, img_filt) or route_count >= 30:
            if verbose:
                print(f"[Hybrid Filter] 알림: 변화가 없거나 최대 루프에 도달하여 {route_count}루트에서 종료합니다.")
            break
        
        # 조기 종료: 복원된 픽셀 수가 임계값(10개) 이하이면 수렴 판정
        if restored_count <= 10:
            if verbose:
                print(f"[Hybrid Filter] 조기 종료: {route_count}루트에서 복원 픽셀 수({restored_count}개)가 임계값(10개) 이하입니다.")
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
