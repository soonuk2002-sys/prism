# CMP AI Decision Platform

반도체 산화막 CMP 공정 의사결정 지원 Streamlit 웹사이트입니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV 필수 컬럼

아래 컬럼이 필요합니다.

- Pressure
- Pad Speed
- Carrier Speed
- Slurry Flow Rate
- Polishing Time
- MRR

단위 예시:

- Pressure: psi
- Pad Speed: rpm
- Carrier Speed: rpm
- Slurry Flow Rate: mL/min
- Polishing Time: sec
- MRR: nm/min

## 지원 기능

- 논문 CSV 여러 개 업로드
- 데이터 자동 병합
- RandomForest 실제 학습
- R2, MAE, RMSE 평가
- Feature Importance
- SHAP 기반 XAI
- 목표 MRR 기반 공정 조건 추천
