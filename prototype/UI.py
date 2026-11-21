import os
import sys
import json
import hashlib
from datetime import datetime
import glob
import base64
import streamlit as st
from st_clickable_images import clickable_images
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
    st.error("AMSHF.py 파일을 찾을 수 없습니다. 경로를 확인해주세요. / Cannot find AMSHF.py. Please check the path.")

# --- 1. 페이지 설정 및 다크 테마 ---
st.set_page_config(page_title="AMSHF Restoration Platform", layout="wide")

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_base64_image(filepath):
    with open(filepath, "rb") as f:
        data = f.read()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"

@st.cache_data(show_spinner=False)
def cached_amshf_filter(img, k_neu, k_moo, k_mean):
    return amshf_filter(
        img,
        k_neu_th=k_neu, 
        k_moo_th=k_moo,
        k_mean_th=k_mean,
        return_route=True, 
        return_stats=True,
        verbose=True
    )

def get_image_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()

def save_to_history(input_img, restored_img, meta_data):
    img_hash = get_image_hash(input_img)
    if st.session_state.get('last_saved_hash') == img_hash:
        return # 이미 저장됨

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = f"{timestamp}_{img_hash[:6]}"
    
    cv2.imwrite(os.path.join(HISTORY_DIR, f"{uid}_input.png"), input_img)
    cv2.imwrite(os.path.join(HISTORY_DIR, f"{uid}_restored.png"), restored_img)
    
    with open(os.path.join(HISTORY_DIR, f"{uid}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)
        
    st.session_state['last_saved_hash'] = img_hash

def load_history_items():
    items = []
    # 파일 이름 역순 정렬 (최신순)
    for meta_file in sorted(glob.glob(os.path.join(HISTORY_DIR, "*_meta.json")), reverse=True):
        uid = os.path.basename(meta_file).replace("_meta.json", "")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        items.append({
            "uid": uid,
            "meta": meta,
            "input_path": os.path.join(HISTORY_DIR, f"{uid}_input.png"),
            "restored_path": os.path.join(HISTORY_DIR, f"{uid}_restored.png")
        })
    return items

# --- Session State 초기화 ---
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'live'
if 'last_saved_hash' not in st.session_state:
    st.session_state['last_saved_hash'] = None
if 'scroll_to_top' not in st.session_state:
    st.session_state['scroll_to_top'] = False


if st.session_state['scroll_to_top']:
    st.components.v1.html("""
        <script>
            setTimeout(() => {
                const main = window.parent.document.querySelector('.main');
                if (main) main.scrollTo({top: 0, behavior: 'smooth'});
                window.parent.scrollTo({top: 0, behavior: 'smooth'});
            }, 100);
        </script>
    """, height=0)
    st.session_state['scroll_to_top'] = False

def reset_to_live():
    st.session_state['current_view'] = 'live'

def set_history_view(uid):
    st.session_state['current_view'] = uid

# --- Language Dictionary ---
LANG = {
    "KR": {
        "sidebar_title": "📁 이미지 로더",
        "lang_select": "언어 선택 (Language)",
        "source_select": "이미지 소스를 선택하세요:",
        "source_upload": "Upload Image (파일 업로드)",
        "source_webcam": "Webcam (실시간)",
        "source_sample": "Sample Set (샘플)",
        "upload_prompt": "이미지 파일을 선택하세요",
        "sample_info": "샘플 이미지: 기본 256x256 회색 이미지 사용",
        "main_title": "🛡️ AMSHF Factory Image Restoration",
        "input_noise": "**입력 이미지 노이즈 밀도:**",
        "est_noise": "**추정 노이즈 밀도 (복원 기반):**",
        "visual_comp": "시각적 비교 (Visual Comparison)",
        "input_caption": "입력 이미지",
        "restored_caption": "AMSHF 복원 완료",
        "routes": "**총 처리 루트 (Convergence):**",
        "reliability": "신뢰성 분석 (Reliability Analysis)",
        "desc_pie": "💡 복원 과정에서 각 필터 단계(Median, Group Mean, Mean)가 사용된 비율을 나타냅니다.",
        "desc_gauge": "💡 복원된 픽셀 수를 기반으로 역산한 이미지의 실제 노이즈 밀도 추정치입니다.",
        "waiting": "왼쪽 사이드바에서 이미지를 업로드하거나 소스를 선택하여 필터를 테스트하세요.",
        "history_title": "📜 히스토리 갤러리",
        "history_empty": "저장된 히스토리가 없습니다.",
        "load_btn": "불러오기"
    },
    "EN": {
        "sidebar_title": "📁 Image Loader",
        "lang_select": "Language",
        "source_select": "Select image source:",
        "source_upload": "Upload Image",
        "source_webcam": "Webcam",
        "source_sample": "Sample Set",
        "upload_prompt": "Select an image file",
        "sample_info": "Sample image: Using default 256x256 gray image",
        "main_title": "🛡️ AMSHF Factory Image Restoration",
        "input_noise": "**Input Image Noise Density:**",
        "est_noise": "**Estimated Noise Density (from restoration):**",
        "visual_comp": "Visual Comparison",
        "input_caption": "Input Image",
        "restored_caption": "AMSHF Restored",
        "routes": "**Total Processing Routes (Convergence):**",
        "reliability": "Reliability Analysis",
        "desc_pie": "💡 Represents the proportion of each filter stage (Median, Group Mean, Mean) utilized during restoration.",
        "desc_gauge": "💡 The estimated actual noise density of the image, calculated based on the number of restored pixels.",
        "waiting": "Upload an image or select a source from the left sidebar to test the filter.",
        "history_title": "📜 History Gallery",
        "history_empty": "No history available.",
        "load_btn": "Load"
    }
}

st.markdown("""
    <style>
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .history-card {
        text-align: center;
        padding: 10px;
        border: 1px solid var(--secondary-background-color);
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .history-card img {
        border-radius: 5px;
        margin-bottom: 10px;
    }
    

    </style>
    """, unsafe_allow_html=True)

# --- 2. 사이드바: 이미지 소스 선택 ---
with st.sidebar:
    # Language Toggle
    lang_choice = st.selectbox("🌐 Language / 언어", ["한국어 (KR)", "English (EN)"])
    lang_code = "KR" if "KR" in lang_choice else "EN"
    t = LANG[lang_code]

    st.title(t["sidebar_title"])
    source_option = st.radio(
        t["source_select"],
        ("upload", "webcam", "sample"),
        format_func=lambda x: t[f"source_{x}"],
        key="source_radio",
        on_change=reset_to_live
    )

# --- 3. 이미지 로딩 로직 ---
input_image = None

if source_option == "upload":
    uploaded_file = st.file_uploader(t["upload_prompt"], type=['jpg', 'png', 'bmp', 'jpeg'], key="file_uploader", on_change=reset_to_live)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

elif source_option == "webcam":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()

elif source_option == "sample":
    input_image = np.full((256, 256), 128, dtype=np.uint8)
    st.info(t["sample_info"])

# --- 4. 메인 대시보드 화면 ---
st.title(t["main_title"])

render_input_img = None
render_restored_img = None
render_input_noise = 0
render_est_noise = 0
render_routes = 0
render_stats = None
render_time = 0
has_data_to_render = False

if st.session_state['current_view'] == 'live':
    if input_image is not None:
        k_neu = 2
        k_moo = 5

        img_resized = cv2.resize(input_image, (512, 512))
        
        total_pixels = img_resized.size
        noise_pixels = np.sum((img_resized == 0) | (img_resized == 255))
        input_noise_density = (noise_pixels / total_pixels) * 100
        
        start_t = time.time()
        restored_img, routes, stats = cached_amshf_filter(img_resized, k_neu, k_moo, 2)
        proc_time = (time.time() - start_t) * 1000
        
        corrected_pixels = sum(stats.values())
        estimated_noise_density = (corrected_pixels / total_pixels) * 100

        # Save to history
        meta_data = {
            "input_noise_density": float(input_noise_density),
            "estimated_noise_density": float(estimated_noise_density),
            "routes": int(routes),
            "stats": {k: int(v) for k, v in stats.items()},
            "proc_time": float(proc_time),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_to_history(img_resized, restored_img, meta_data)

        render_input_img = img_resized
        render_restored_img = restored_img
        render_input_noise = input_noise_density
        render_est_noise = estimated_noise_density
        render_routes = routes
        render_stats = stats
        render_time = proc_time
        has_data_to_render = True

else:
    # History mode
    uid = st.session_state['current_view']
    meta_path = os.path.join(HISTORY_DIR, f"{uid}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        input_path = os.path.join(HISTORY_DIR, f"{uid}_input.png")
        restored_path = os.path.join(HISTORY_DIR, f"{uid}_restored.png")
        
        render_input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        render_restored_img = cv2.imread(restored_path, cv2.IMREAD_GRAYSCALE)
        render_input_noise = meta["input_noise_density"]
        render_est_noise = meta["estimated_noise_density"]
        render_routes = meta["routes"]
        render_stats = meta["stats"]
        render_time = meta["proc_time"]
        has_data_to_render = True

# --- 시각화 렌더링 ---
if has_data_to_render:
    st.write(f"{t['input_noise']} {render_input_noise:.2f}%")
    st.write(f"{t['est_noise']} {render_est_noise:.2f}%")

    col_img, col_metrics = st.columns([1.5, 1])

    with col_img:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(t["visual_comp"])
        c1, c2 = st.columns(2)
        c1.image(render_input_img, caption=f"{t['input_caption']} ({render_input_noise:.1f}% noise)", width="stretch")
        c2.image(render_restored_img, caption=f"{t['restored_caption']} ({render_time:.1f}ms)", width="stretch")
        st.write(f"{t['routes']} {render_routes}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(t["reliability"])
        
        c1, c2, c3 = st.columns([0.15, 0.7, 0.15])
        
        with c2:
            logic_df = pd.DataFrame({
                'Logic': ['Neumann (Median)', 'Moore (Group Mean)', 'Fallback (Mean)'],
                'Count': [render_stats.get('median', 0), render_stats.get('group', 0), render_stats.get('mean', 0)]
            })
            fig_pie = px.pie(logic_df, values='Count', names='Logic', hole=0.5,
                             color_discrete_sequence=['#12b886', '#3A7BD5', '#7F00FF'])
            fig_pie.update_layout(margin=dict(l=40,r=40,t=30,b=10), height=190, 
                                 paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, width="stretch")
            st.caption(t["desc_pie"])
            st.markdown("<br>", unsafe_allow_html=True)

            fig_gauge = go.Figure()
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=render_est_noise,
                title={"text": "Noise Density (%)", "font": {"size": 14}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#12b886"}},
                number={"font": {"size": 24}, "valueformat": ".1f"}
            ))
            fig_gauge.update_layout(height=160, margin=dict(l=30,r=30,t=60,b=10), 
                                    paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, width="stretch")
            st.caption(t["desc_gauge"])
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state['current_view'] == 'live':
    st.info(t["waiting"])

# --- 5. 히스토리 갤러리 ---
st.markdown("---")
st.subheader(t["history_title"])

history_items = load_history_items()
if not history_items:
    st.write(t["history_empty"])
else:
    images = []
    titles = []
    valid_items = []
    for item in history_items:
        if os.path.exists(item["restored_path"]):
            images.append(get_base64_image(item["restored_path"]))
            titles.append(item["meta"]["timestamp"])
            valid_items.append(item)
            
    if images:
        st.markdown("<p style='text-align: center; color: gray;'>※ 썸네일 이미지를 클릭하면 해당 시점의 분석 결과로 즉시 이동합니다.</p>", unsafe_allow_html=True)
        
        clicked = clickable_images(
            images,
            titles=titles,
            div_style={
                "display": "flex", 
                "justify-content": "flex-start", 
                "flex-wrap": "nowrap", 
                "gap": "10px", 
                "overflow-x": "auto",
                "padding-bottom": "15px"
            },
            img_style={
                "height": "110px", 
                "flex-shrink": "0",
                "border-radius": "8px", 
                "cursor": "pointer", 
                "transition": "transform 0.2s", 
                "box-shadow": "0 4px 6px rgba(0,0,0,0.3)"
            }
        )
        
        if clicked > -1:
            set_history_view(valid_items[clicked]["uid"])
            st.session_state['scroll_to_top'] = True
            st.rerun()