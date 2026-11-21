# AMSHF Restoration Platform (Prototype)

AMSHF(Adaptive Multi-Stage Hybrid Filter) 알고리즘 기반의 소금-후추 노이즈(SPN) 영상 복원 Streamlit 웹 프로토타입입니다.

## 구성 파일
- `AMSHF.py`: AMSHF 필터 핵심 알고리즘 구현 (3단계 하이브리드 복원, 반경 확장 r=1~2, 반복 루트)
- `UI.py`: Streamlit 기반 인터랙티브 영상 복원 및 분석 UI
- `run_ui.bat`: Windows 환경 원클릭 실행 스크립트

## 실행 방법
```bash
python -m streamlit run UI.py
```
또는 Windows에서 `run_ui.bat`을 실행합니다.
