import os
import sys
import streamlit as st
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from PIL import Image

# AMSHF 모듈을 불러오기 위한 경로 설정
sys.path.insert(0, os.path.dirname(__file__))
try:
    from AMSHF import amshf_filter 
except ImportError:
    st.error("AMSHF.py 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

# --- 1. 페이지 설정 및 다크 테마 ---
st.set_page_config(page_title="AMSHF Restoration Platform", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 사이드바: 이미지 소스 선택 ---
with st.sidebar:
    st.title("📁 Image Loader")
    source_option = st.radio(
        "이미지 소스를 선택하세요:",
        ("Upload Image (파일 업로드)", "Webcam (실시간)", "Sample Set")
    )
    
    st.markdown("---")
    st.info("이미지를 선택한 후, 메인 대시보드에서 AMSHF 필터 파라미터를 조절하세요.")

# --- 3. 이미지 로딩 로직 ---
input_image = None

if source_option == "Upload Image (파일 업로드)":
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=['jpg', 'png', 'bmp', 'jpeg'])
    if uploaded_file is not None:
        # 업로드된 파일을 OpenCV 형식으로 변환
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

elif source_option == "Webcam (실시간)":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()

elif source_option == "Sample Set":
    # 샘플 이미지: 기본 회색 이미지 생성 (실제 샘플 파일이 없으므로)
    input_image = np.full((256, 256), 128, dtype=np.uint8)  # 중간 회색 이미지
    st.info("샘플 이미지: 기본 256x256 회색 이미지 사용")

# --- 4. 메인 대시보드 화면 ---
st.title("🛡️ AMSHF Factory Image Restoration")

if input_image is not None:
    k_neu = 2
    k_moo = 5

    # 1단계: 처리 속도를 위해 리사이징 (선택 사항)
    img_resized = cv2.resize(input_image, (512, 512))
    
    # 입력 이미지의 노이즈 비율 계산 (0 또는 255인 픽셀)
    total_pixels = img_resized.size
    noise_pixels = np.sum((img_resized == 0) | (img_resized == 255))
    input_noise_density = (noise_pixels / total_pixels) * 100
    st.write(f"**Input Image Noise Density:** {input_noise_density:.2f}%")
    
    # 2단계: AMSHF 필터 적용 (입력 이미지의 노이즈를 그대로 복원)
    start_t = time.time()
    restored_img, routes, stats = amshf_filter(
        img_resized,  # 노이즈 추가 없이 원본 리사이즈 이미지 사용
        k_neu_th=k_neu, 
        k_moo_th=k_moo,
        k_mean_th=2,
        return_route=True, 
        return_stats=True,
        verbose=True  # 디버깅을 위해 로그 출력
    )
    proc_time = (time.time() - start_t) * 1000


    
    # 역산된 노이즈 비율 (복원된 픽셀 수 기반)
    corrected_pixels = sum(stats.values())
    estimated_noise_density = (corrected_pixels / total_pixels) * 100
    st.write(f"**Estimated Noise Density (from restoration):** {estimated_noise_density:.2f}%")

    # 4단계: 시각화 레이아웃
    col_img, col_metrics = st.columns([1.5, 1])

    with col_img:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Visual Comparison")
        c1, c2 = st.columns(2)
        c1.image(img_resized, caption=f"Input Image ({input_noise_density:.1f}% noise)", use_container_width=True)
        c2.image(restored_img, caption=f"AMSHF Restored ({proc_time:.1f}ms)", use_container_width=True)
        st.write(f"**Total Processing Routes (Convergence):** {routes}")
        st.markdown('</div>', unsafe_allow_html=True)



    with col_metrics:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Reliability Analysis")
        
        # 70% 비율로 축소하기 위해 내부 컬럼 사용
        c1, c2, c3 = st.columns([0.15, 0.7, 0.15])
        
        with c2:
            # 알고리즘 내부 로직 사용량 파이 차트 (stats 변수 활용)
            logic_df = pd.DataFrame({
                'Logic': ['Neumann (Median)', 'Moore (Group Mean)', 'Fallback (Mean)'],
                'Count': [stats.get('median', 0), stats.get('group', 0), stats.get('mean', 0)]
            })
            # 밝은 배경에서도 잘 보이도록 색상 명도/채도 조절 (#00ffcc -> #12b886)
            fig_pie = px.pie(logic_df, values='Count', names='Logic', hole=0.5,
                             color_discrete_sequence=['#12b886', '#3A7BD5', '#7F00FF'])
            fig_pie.update_layout(margin=dict(l=40,r=40,t=30,b=10), height=190, 
                                 paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

            # 노이즈 밀도 추산 (복원된 픽셀 수 기반)
            fig_gauge = go.Figure()
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=estimated_noise_density,
                title={"text": "Estimated Noise Density (%)", "font": {"size": 14}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#12b886"}},
                number={"font": {"size": 24}, "valueformat": ".1f"}
            ))
            fig_gauge.update_layout(height=160, margin=dict(l=30,r=30,t=60,b=10), 
                                    paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("왼쪽 사이드바에서 이미지를 업로드하거나 소스를 선택하여 필터를 테스트하세요.")