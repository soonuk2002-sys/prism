# app.py
# ============================================================
# CMP(Chemical Mechanical Polishing) 공정 의사결정 지원 플랫폼
# ------------------------------------------------------------
# 기능 요약
# 1) 논문 CSV 여러 개 업로드
# 2) 데이터 자동 병합 및 컬럼명 표준화
# 3) EDA 대시보드
# 4) 실제 RandomForest 모델 학습
# 5) 모델 성능 평가: R2, MAE, RMSE
# 6) Feature Importance 분석
# 7) SHAP 기반 XAI 분석
# 8) 목표 MRR 기반 공정 조건 최적화
# 9) 반도체 공정 엔지니어용 UI
#
# 실행 방법
#   pip install -r requirements.txt
#   streamlit run app.py
# ============================================================

import streamlit as st
from google import genai

import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore")




# ============================================================
# 1. Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="CMP AI Decision Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    client = None
    
# ============================================================
# 2. 커스텀 CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #F5F7FA;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: clamp(1.6rem, 3vw, 2.3rem);
    font-weight: 800;
    color: #172033;
    margin-bottom: 0.2rem;
    line-height: 1.25;
    white-space: normal;
    word-break: keep-all;
    }

    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #5F6B7A;
        margin-bottom: 1.2rem;
    }

    .section-card {
        background-color: white;
        border: 1px solid #E6EAF0;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(17, 24, 39, 0.04);
        margin-bottom: 1rem;
    }

    .small-caption {
        color: #667085;
        font-size: 0.85rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 800;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #475467;
    }

    .warning-box {
        background-color: #FFF7E6;
        color: #8A5200;
        border-left: 5px solid #F59E0B;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 8px;
    }

    .success-box {
        background-color: #ECFDF3;
        color: #027A48;
        border-left: 5px solid #12B76A;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 8px;
    }

    .info-box {
        background-color: #EFF8FF;
        color: #175CD3;
        border-left: 5px solid #2E90FA;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. 전역 설정
# ============================================================

FEATURE_COLUMNS = [
    "Pressure",
    "Pad Speed",
    "Carrier Speed",
    "Slurry Flow Rate",
]

TARGET_COLUMN = "MRR"

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

UNIT_MAP = {
    "Pressure": "psi",
    "Pad Speed": "rpm",
    "Carrier Speed": "rpm",
    "Slurry Flow Rate": "mL/min",
    "MRR": "nm/min"
}


# ============================================================
# 4. 컬럼명 표준화 함수
# ============================================================

def normalize_column_name(col):
    """
    논문마다 컬럼명이 다르기 때문에 다양한 표현을 표준 컬럼명으로 통합한다.
    예:
    Pressure (psi), Down Force, Downforce -> Pressure
    Platen Speed, Pad Speed (rpm) -> Pad Speed
    Removal Rate, Material Removal Rate -> MRR
    """
    original = str(col).strip()
    key = original.lower().replace("_", " ").replace("-", " ").replace("(", " ").replace(")", " ")
    key = " ".join(key.split())

    mapping = {
        "Pressure": [
            "pressure", "pressure psi", "down force", "downforce",
            "applied pressure", "polishing pressure", "load", "wafer pressure"
        ],
        "Pad Speed": [
            "pad speed", "pad speed rpm", "platen speed", "platen speed rpm",
            "table speed", "table speed rpm", "rotation speed", "rotational speed"
        ],
        "Carrier Speed": [
            "carrier speed", "carrier speed rpm", "head speed", "head speed rpm",
            "wafer speed", "wafer speed rpm", "carrier rotation", "head rotation"
        ],
        "Slurry Flow Rate": [
            "slurry flow rate", "slurry flow rate ml min", "slurry flow",
            "flow rate", "flow rate ml min", "slurry rate", "slurry supply"
        ],
        "Polishing Time": [
            "polishing time", "polishing time sec", "polishing time s",
            "time", "time sec", "process time", "duration"
        ],
        "MRR": [
            "mrr", "mrr nm min", "material removal rate",
            "removal rate", "polishing rate", "oxide removal rate"
        ],
        "Paper ID": [
            "paper id", "paper", "source", "reference", "citation", "doi"
        ],
        "Table/Figure": [
            "table", "figure", "source table", "source figure", "table figure",
            "source table or figure"
        ]
    }

    for standard_name, candidates in mapping.items():
        if key in candidates:
            return standard_name

    return original


def normalize_dataframe_columns(df):
    """
    전체 데이터프레임 컬럼명을 표준화한다.
    """
    renamed = {}
    for col in df.columns:
        renamed[col] = normalize_column_name(col)
    return df.rename(columns=renamed)


# ============================================================
# 5. 가상 CMP 데이터 생성
# ============================================================

@st.cache_data
def generate_virtual_cmp_data(n_rows=120, seed=42):
    """
    논문 데이터가 없을 때도 웹사이트가 작동하도록 가상 데이터를 생성한다.
    실제 CMP 경향성:
    - Pressure 증가: 일반적으로 MRR 증가
    - Pad Speed 증가: 상대속도 증가로 MRR 증가
    - Slurry Flow 증가: 화학 반응 및 입자 공급 증가
    - 너무 높은 조건은 비선형성/포화 효과를 가정
    """
    rng = np.random.default_rng(seed)

    pressure = rng.uniform(3.0, 7.0, n_rows)
    pad_speed = rng.uniform(40.0, 120.0, n_rows)
    carrier_speed = rng.uniform(30.0, 100.0, n_rows)
    slurry_flow = rng.uniform(100.0, 300.0, n_rows)

    noise = rng.normal(0, 7.0, n_rows)

    mrr = (
        65
        + 20.0 * pressure
        + 0.42 * pad_speed
        + 0.20 * carrier_speed
        + 0.075 * slurry_flow
        - 1.40 * pressure ** 2
        - 0.0009 * (pad_speed - 95) ** 2
        + 4.0 * np.sin(slurry_flow / 55)
        + noise
    )

    df = pd.DataFrame({
        "Pressure": pressure.round(3),
        "Pad Speed": pad_speed.round(3),
        "Carrier Speed": carrier_speed.round(3),
        "Slurry Flow Rate": slurry_flow.round(3),
        "MRR": mrr.round(3),
        "Paper ID": "Virtual Dataset",
        "Table/Figure": "Generated"
    })

    return df


# ============================================================
# 6. 데이터 검증 및 전처리
# ============================================================

def validate_and_clean_data(df):
    """
    업로드된 데이터가 모델 학습에 적합한지 확인한다.
    - 필수 컬럼 존재 여부 확인
    - 숫자형 변환
    - 결측값 제거
    """
    df = normalize_dataframe_columns(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        return False, df, missing

    cleaned = df.copy()

    for col in REQUIRED_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    before_rows = len(cleaned)
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    after_rows = len(cleaned)

    if "Paper ID" not in cleaned.columns:
        cleaned["Paper ID"] = "Uploaded Paper"

    if "Table/Figure" not in cleaned.columns:
        cleaned["Table/Figure"] = "Not specified"

    if after_rows == 0:
        return False, cleaned, ["숫자로 변환 가능한 유효 행이 없습니다."]

    cleaned.attrs["dropped_rows"] = before_rows - after_rows
    return True, cleaned, []


def merge_uploaded_files(uploaded_files):
    """
    여러 CSV 파일을 읽어서 자동 병합한다.
    """
    merged_list = []
    error_logs = []

    for file in uploaded_files:
        try:
            raw = pd.read_csv(file)
            raw = normalize_dataframe_columns(raw)

            if "Paper ID" not in raw.columns:
                raw["Paper ID"] = file.name

            is_valid, clean_df, errors = validate_and_clean_data(raw)

            if is_valid:
                clean_df["Uploaded File"] = file.name
                merged_list.append(clean_df)
            else:
                error_logs.append({
                    "file": file.name,
                    "error": ", ".join(errors)
                })

        except Exception as e:
            error_logs.append({
                "file": file.name,
                "error": str(e)
            })

    if len(merged_list) == 0:
        return None, error_logs

    merged_df = pd.concat(merged_list, ignore_index=True)
    return merged_df, error_logs


# ============================================================
# 7. 모델 학습 함수
# ============================================================

@st.cache_resource
def train_random_forest_model(df, n_estimators, max_depth, random_state):
    """
    RandomForestRegressor를 실제로 학습한다.
    데이터 수가 너무 적은 경우에도 작동하도록 test_size를 조정한다.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if len(df) < 10:
        # 데이터가 너무 적으면 전체 데이터 학습만 수행
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        metrics = {
            "R2": r2_score(y, y_pred) if len(y) > 1 else np.nan,
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "Train Size": len(df),
            "Test Size": 0
        }

        return model, X, X, y, y, y_pred, metrics

    test_size = 0.25 if len(df) >= 30 else 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Train Size": len(X_train),
        "Test Size": len(X_test)
    }

    return model, X_train, X_test, y_train, y_test, y_pred, metrics


# ============================================================
# 8. SHAP 분석 함수
# ============================================================

def compute_shap_values(model, X_background, input_row):
    """
    SHAP 라이브러리가 설치되어 있으면 실제 SHAP 값을 계산한다.
    설치되어 있지 않으면 RandomForest feature importance 기반의 대체 기여도를 사용한다.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_row)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        values = np.array(shap_values).reshape(-1)
        base_value = float(explainer.expected_value)

        method = "Actual SHAP"
        return values, base_value, method

    except Exception:
        # SHAP이 설치되지 않았거나 실행 오류가 있을 때 대체 로직
        importances = model.feature_importances_
        mean_values = X_background.mean().values
        current_values = input_row.values.reshape(-1)
        diff = current_values - mean_values

        scaled = diff * importances * 0.25

        base_value = float(model.predict(X_background).mean())
        method = "Fallback Importance Contribution"

        return scaled, base_value, method


# ============================================================
# 9. 최적화 함수
# ============================================================

def optimize_process_conditions(
    model,
    target_mrr,
    pressure_range,
    pad_speed_range,
    carrier_speed_range,
    slurry_flow_range,
    polishing_time_range,
    n_candidates,
    random_state=42
):
    """
    목표 MRR에 가까운 조건 조합을 랜덤 서치로 탐색한다.
    실제 연구에서는 Bayesian Optimization, Genetic Algorithm 등으로 확장 가능.
    """
    rng = np.random.default_rng(random_state)

    candidates = pd.DataFrame({
        "Pressure": rng.uniform(pressure_range[0], pressure_range[1], n_candidates),
        "Pad Speed": rng.uniform(pad_speed_range[0], pad_speed_range[1], n_candidates),
        "Carrier Speed": rng.uniform(carrier_speed_range[0], carrier_speed_range[1], n_candidates),
        "Slurry Flow Rate": rng.uniform(slurry_flow_range[0], slurry_flow_range[1], n_candidates)
    })

    predictions = model.predict(candidates[FEATURE_COLUMNS])
    candidates["Predicted MRR"] = predictions
    candidates["Target Error"] = np.abs(candidates["Predicted MRR"] - target_mrr)

    # 엔지니어 관점에서 과도한 조건을 피하기 위한 간단한 안정성 점수
    candidates["Process Stress Score"] = (
        (candidates["Pressure"] - pressure_range[0]) / (pressure_range[1] - pressure_range[0]) * 0.35
        + (candidates["Pad Speed"] - pad_speed_range[0]) / (pad_speed_range[1] - pad_speed_range[0]) * 0.25
        + (candidates["Carrier Speed"] - carrier_speed_range[0]) / (carrier_speed_range[1] - carrier_speed_range[0]) * 0.15
        + (candidates["Slurry Flow Rate"] - slurry_flow_range[0]) / (slurry_flow_range[1] - slurry_flow_range[0]) * 0.15
    )

    candidates["Recommendation Score"] = (
        candidates["Target Error"] + 7.5 * candidates["Process Stress Score"]
    )

    result = candidates.sort_values("Recommendation Score").head(10)

    return result.round(3)


# ============================================================
# 10. 다운로드 함수
# ============================================================

def dataframe_to_csv_download(df):
    """
    데이터프레임을 CSV 다운로드용 bytes로 변환한다.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")


# ============================================================
# 11. 사이드바: 데이터 업로드 및 모델 설정
# ============================================================

st.sidebar.title("⚙️ PRISM")

st.sidebar.markdown("### 1. 논문 CSV 데이터")
uploaded_files = st.sidebar.file_uploader(
    "논문에서 정리한 CSV 파일을 여러 개 업로드",
    type=["csv"],
    accept_multiple_files=True,
    help="필수 컬럼: Pressure, Pad Speed, Carrier Speed, Slurry Flow Rate, MRR"
)

use_virtual_when_empty = st.sidebar.checkbox(
    "업로드 데이터가 없으면 가상 데이터 사용",
    value=True
)

if uploaded_files:
    merged_df, upload_errors = merge_uploaded_files(uploaded_files)

    if merged_df is not None:
        df = merged_df.copy()
        data_status = "논문 CSV 업로드 데이터"
    else:
        df = generate_virtual_cmp_data()
        data_status = "가상 데이터"
        st.sidebar.error("업로드 파일을 읽지 못해 가상 데이터로 실행합니다.")

    if upload_errors:
        with st.sidebar.expander("업로드 오류 로그"):
            for log in upload_errors:
                st.write(f"- {log['file']}: {log['error']}")

else:
    if use_virtual_when_empty:
        df = generate_virtual_cmp_data()
        data_status = "가상 데이터"
    else:
        st.warning("CSV 파일을 업로드해야 분석을 시작할 수 있습니다.")
        st.stop()


st.sidebar.divider()

st.sidebar.markdown("### 2. RandomForest 설정")
n_estimators = st.sidebar.slider(
    "Tree 개수",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

max_depth_option = st.sidebar.selectbox(
    "Max Depth",
    options=["None", 3, 5, 7, 10, 15],
    index=0
)

max_depth = None if max_depth_option == "None" else int(max_depth_option)

random_state = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

st.sidebar.divider()

st.sidebar.markdown("### 3. 공정 변수 입력")

pressure_input = st.sidebar.slider(
    "Pressure [psi]",
    3.0,
    7.0,
    5.0,
    0.1
)

pad_speed_input = st.sidebar.slider(
    "Pad Speed [rpm]",
    40.0,
    120.0,
    80.0,
    1.0
)

carrier_speed_input = st.sidebar.slider(
    "Carrier Speed [rpm]",
    30.0,
    100.0,
    65.0,
    1.0
)

slurry_flow_input = st.sidebar.slider(
    "Slurry Flow Rate [mL/min]",
    100.0,
    300.0,
    200.0,
    5.0
)




# ============================================================
# 12. 모델 학습
# ============================================================

model, X_train, X_test, y_train, y_test, y_pred, metrics = train_random_forest_model(
    df,
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=int(random_state)
)


current_input_df = pd.DataFrame([{
    "Pressure": pressure_input,
    "Pad Speed": pad_speed_input,
    "Carrier Speed": carrier_speed_input,
    "Slurry Flow Rate": slurry_flow_input
}])

current_prediction = float(model.predict(current_input_df)[0])


# ============================================================
# 13. 메인 헤더
# ============================================================

st.markdown("## ⚙️ PRISM CMP 공정 의사결정 지원 플랫폼")
st.caption(
    "Oxide CMP Process Data Analytics · RandomForest MRR Prediction · "
    "SHAP Explainable AI · Process Optimization"
)

top1, top2, top3, top4 = st.columns(4)
top1.metric("데이터 소스", data_status)
top2.metric("데이터 행 수", f"{len(df)} rows")
top3.metric("입력 조건 예측 MRR", f"{current_prediction:.2f} nm/min")
top4.metric("모델 R²", f"{metrics['R2']:.3f}" if not np.isnan(metrics["R2"]) else "N/A")


# ============================================================
# 14. 탭 구성
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "① EDA & 데이터 대시보드",
    "② AI MRR 예측 모델",
    "③ XAI / SHAP 분석",
    "④ 공정 최적화"
])


# ============================================================
# TAB 1. EDA
# ============================================================

with tab1:
    st.subheader("📊 CMP 공정 데이터 대시보드")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("#### 병합된 공정 데이터")
        st.dataframe(df, use_container_width=True, height=360)

    with col_b:
        st.markdown("#### 데이터 요약")

        summary_df = df[REQUIRED_COLUMNS].describe().T
        summary_df = summary_df[["mean", "std", "min", "max"]].round(3)
        st.dataframe(summary_df, use_container_width=True)

        csv_bytes = dataframe_to_csv_download(df)
        st.download_button(
            "병합 데이터 CSV 다운로드",
            data=csv_bytes,
            file_name=f"merged_cmp_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    st.divider()

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("평균 MRR", f"{df['MRR'].mean():.2f}")
    metric2.metric("MRR 표준편차", f"{df['MRR'].std():.2f}")
    metric3.metric("최대 MRR", f"{df['MRR'].max():.2f}")
    metric4.metric("최소 MRR", f"{df['MRR'].min():.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 공정 변수 - MRR 산점도")

        x_feature = st.selectbox(
            "X축 변수 선택",
            FEATURE_COLUMNS,
            index=0
        )

        fig_scatter = px.scatter(
            df,
            x=x_feature,
            y="MRR",
            color="Paper ID",
            size="Slurry Flow Rate",
            hover_data=REQUIRED_COLUMNS + ["Paper ID", "Table/Figure"],
            title=f"{x_feature} vs MRR"
        )

        fig_scatter.update_layout(height=460)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.markdown("#### 상관관계 히트맵")

        corr = df[REQUIRED_COLUMNS].corr()

        fig_heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )

        fig_heatmap.update_layout(height=460)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### MRR 분포")
        fig_hist = px.histogram(
            df,
            x="MRR",
            nbins=25,
            marginal="box",
            title="MRR Distribution"
        )
        fig_hist.update_layout(height=420)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        st.markdown("#### 논문 출처별 MRR 비교")

        if "Paper ID" in df.columns:
            fig_box = px.box(
                df,
                x="Paper ID",
                y="MRR",
                points="all",
                title="MRR by Paper Source"
            )
            fig_box.update_layout(height=420)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Paper ID 컬럼이 있으면 논문 출처별 비교가 표시됩니다.")


# ============================================================
# TAB 2. AI 예측 모델
# ============================================================

with tab2:
    st.subheader("🤖 실제 RandomForest 기반 MRR 예측")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 현재 입력 공정 조건")

        input_display = current_input_df.T.reset_index()
        input_display.columns = ["Process Variable", "Input Value"]
        input_display["Unit"] = input_display["Process Variable"].map(UNIT_MAP)

        st.dataframe(input_display, use_container_width=True)

        if st.button("품질 예측 실행", type="primary"):
            st.session_state["last_prediction"] = current_prediction

        if "last_prediction" not in st.session_state:
            st.session_state["last_prediction"] = current_prediction

    with col2:
        st.markdown("#### 예측 결과")

        st.metric(
            "Predicted MRR",
            f"{st.session_state['last_prediction']:.2f} nm/min"
        )

        if st.session_state["last_prediction"] >= df["MRR"].quantile(0.75):
            st.success("현재 조건은 데이터셋 기준 상위 MRR 영역에 해당합니다.")
        elif st.session_state["last_prediction"] >= df["MRR"].quantile(0.35):
            st.info("현재 조건은 데이터셋 기준 중간 수준의 안정적 MRR 영역입니다.")
        else:
            st.warning("현재 조건은 상대적으로 낮은 MRR 영역입니다. Pressure, Pad Speed, Slurry Flow Rate 조정을 검토할 수 있습니다.")

    st.divider()

    st.markdown("#### 모델 성능 평가")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("R²", f"{metrics['R2']:.3f}" if not np.isnan(metrics["R2"]) else "N/A")
    c2.metric("MAE", f"{metrics['MAE']:.3f}")
    c3.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c4.metric("Train Size", int(metrics["Train Size"]))
    c5.metric("Test Size", int(metrics["Test Size"]))

    if metrics["Test Size"] > 0:
        eval_df = pd.DataFrame({
            "Actual MRR": y_test.values,
            "Predicted MRR": y_pred
        })

        fig_pred = px.scatter(
            eval_df,
            x="Actual MRR",
            y="Predicted MRR",
            title="Actual vs Predicted MRR",
            trendline="ols"
        )

        min_val = min(eval_df["Actual MRR"].min(), eval_df["Predicted MRR"].min())
        max_val = max(eval_df["Actual MRR"].max(), eval_df["Predicted MRR"].max())

        fig_pred.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideal Prediction"
            )
        )

        fig_pred.update_layout(height=480)
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("데이터 수가 적어 별도 테스트셋 없이 전체 데이터로 학습했습니다.")

    st.divider()

    st.markdown("#### Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="RandomForest Feature Importance"
    )

    fig_importance.update_layout(height=430)
    st.plotly_chart(fig_importance, use_container_width=True)

    top_feature = importance_df.sort_values("Importance", ascending=False).iloc[0]
    st.info(
        f"현재 학습된 모델에서 가장 중요한 변수는 **{top_feature['Feature']}**입니다. "
        f"중요도는 **{top_feature['Importance']:.3f}**입니다."
    )


# ============================================================
# TAB 3. XAI / SHAP 분석
# ============================================================

with tab3:
    st.subheader("🔍 XAI / SHAP 기반 모델 해석")

    shap_values, base_value, shap_method = compute_shap_values(
        model,
        X_train,
        current_input_df
    )

    shap_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP Contribution": shap_values,
        "Input Value": current_input_df.iloc[0].values
    })

    shap_df["Direction"] = np.where(
        shap_df["SHAP Contribution"] >= 0,
        "MRR 증가 기여",
        "MRR 감소 기여"
    )

    st.markdown(
        f"""
        <div class="info-box">
        사용된 설명 방식: <b>{shap_method}</b><br>
        Base MRR: <b>{base_value:.2f} nm/min</b> → 현재 예측 MRR: <b>{current_prediction:.2f} nm/min</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        shap_plot_df = shap_df.sort_values("SHAP Contribution", ascending=True)

        fig_shap = go.Figure()

        fig_shap.add_trace(
            go.Bar(
                x=shap_plot_df["SHAP Contribution"],
                y=shap_plot_df["Feature"],
                orientation="h",
                text=shap_plot_df["Direction"],
                textposition="auto"
            )
        )

        fig_shap.update_layout(
            title="Feature Contribution to MRR",
            xaxis_title="Contribution to Predicted MRR",
            yaxis_title="Process Variable",
            height=520
        )

        st.plotly_chart(fig_shap, use_container_width=True)

    with col2:
        st.markdown("#### SHAP 기여도 테이블")
        st.dataframe(shap_df.round(4), use_container_width=True)

        st.markdown("#### 엔지니어링 해석")

        for _, row in shap_df.sort_values("SHAP Contribution", key=np.abs, ascending=False).iterrows():
            feature = row["Feature"]
            contribution = row["SHAP Contribution"]
            input_value = row["Input Value"]
            unit = UNIT_MAP.get(feature, "")

            if contribution > 0:
                st.success(
                    f"현재 설정된 {feature} = {input_value:.2f} {unit} 조건은 "
                    f"MRR을 높이는 방향으로 {contribution:.2f} 기여했습니다."
                )
            elif contribution < 0:
                st.warning(
                    f"현재 설정된 {feature} = {input_value:.2f} {unit} 조건은 "
                    f"MRR을 낮추는 방향으로 {contribution:.2f} 기여했습니다."
                )
            else:
                st.info(
                    f"현재 설정된 {feature}는 기준 조건과 유사하여 MRR에 큰 영향을 주지 않았습니다."
                )

    st.divider()

    st.markdown("#### XAI 해석 주의사항")
    st.info(
        "SHAP 값은 모델의 예측 결과를 설명하는 값입니다. "
        "즉, 실제 물리적 인과관계 그 자체라기보다는 학습 데이터 안에서 모델이 학습한 패턴을 설명합니다. "
        "따라서 CMP 공정 조건 변경 시에는 Defect, WIWNU, Scratch, Dishing, Erosion 같은 품질 지표와 함께 검토해야 합니다."
    )


# ============================================================
# TAB 4. 공정 최적화
# ============================================================

with tab4:
    st.subheader("🎯 목표 MRR 기반 공정 최적화")

    st.markdown(
        "목표 MRR을 입력하면 현재 학습된 RandomForest 모델을 기준으로 "
        "목표값에 가까운 공정 조건 조합을 추천합니다."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        target_mrr = st.number_input(
            "목표 MRR [nm/min]",
            min_value=float(max(0, df["MRR"].min() - 30)),
            max_value=float(df["MRR"].max() + 50),
            value=float(df["MRR"].mean()),
            step=1.0
        )

        n_candidates = st.slider(
            "탐색 후보 개수",
            min_value=500,
            max_value=20000,
            value=5000,
            step=500
        )

        st.markdown("#### 탐색 범위 설정")

        pressure_range = st.slider(
            "Pressure Range [psi]",
            3.0,
            7.0,
            (3.0, 7.0),
            0.1
        )

        pad_speed_range = st.slider(
            "Pad Speed Range [rpm]",
            40.0,
            120.0,
            (40.0, 120.0),
            1.0
        )

        carrier_speed_range = st.slider(
            "Carrier Speed Range [rpm]",
            30.0,
            100.0,
            (30.0, 100.0),
            1.0
        )

        slurry_flow_range = st.slider(
            "Slurry Flow Rate Range [mL/min]",
            100.0,
            300.0,
            (100.0, 300.0),
            5.0
        )


        run_optimization = st.button("최적 공정 조건 추천", type="primary")

    with col2:
        st.markdown("#### 최적화 개념")
        st.info(
            "이 플랫폼은 단순히 MRR을 가장 크게 만드는 조건만 찾지 않고, "
            "목표 MRR에 가까우면서도 공정 스트레스가 과도하지 않은 조건을 우선 추천합니다."
        )

        st.markdown(
            """
            추천 점수는 다음 요소를 함께 고려합니다.

            - 목표 MRR과의 오차
            - Pressure가 너무 높은지
            - Pad Speed가 너무 높은지
            - Slurry Flow Rate가 과도한지
            """
        )

    if run_optimization:
        rec_df = optimize_process_conditions(
            model=model,
            target_mrr=target_mrr,
            pressure_range=pressure_range,
            pad_speed_range=pad_speed_range,
            carrier_speed_range=carrier_speed_range,
            slurry_flow_range=slurry_flow_range,
            polishing_time_range=polishing_time_range,
            n_candidates=n_candidates,
            random_state=int(random_state)
        )

        st.divider()

        st.markdown("#### 추천 공정 조건 TOP 10")
        st.dataframe(rec_df, use_container_width=True)

        best = rec_df.iloc[0]

        st.success(
            f"가장 추천되는 조건은 Pressure {best['Pressure']:.2f} psi, "
            f"Pad Speed {best['Pad Speed']:.2f} rpm, "
            f"Carrier Speed {best['Carrier Speed']:.2f} rpm, "
            f"Slurry Flow Rate {best['Slurry Flow Rate']:.2f} mL/min, "
            f"예측 MRR은 {best['Predicted MRR']:.2f} nm/min입니다."
        )

        csv_bytes = dataframe_to_csv_download(rec_df)
        st.download_button(
            "추천 조건 CSV 다운로드",
            data=csv_bytes,
            file_name=f"recommended_cmp_conditions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

        fig_opt = px.scatter(
            rec_df,
            x="Process Stress Score",
            y="Predicted MRR",
            size="Target Error",
            hover_data=FEATURE_COLUMNS,
            title="Recommended Conditions Map"
        )

        fig_opt.add_hline(
            y=target_mrr,
            line_dash="dash",
            annotation_text="Target MRR"
        )

        fig_opt.update_layout(height=480)
        st.plotly_chart(fig_opt, use_container_width=True)

st.divider()
# ============================================================
# AI 공정 전문가 (Gemini)
# ============================================================

st.divider()
st.header("🤖 AI 공정 전문가")

user_question = st.text_input("CMP 공정에 대해 질문하세요.")

if st.button("질문하기"):

    if client is None:
        st.error("Gemini API Key가 등록되지 않았습니다.")

    elif not user_question.strip():
        st.warning("질문을 입력해주세요.")

    else:
        try:
            with st.spinner("AI가 답변을 생성하는 중입니다..."):

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=f"""
당신은 PRISM CMP 공정 의사결정 지원 플랫폼의 AI 공정 엔지니어입니다.

다음 분야의 전문가처럼 답변하세요.

- CMP(Chemical Mechanical Polishing)
- MRR(Material Removal Rate)
- Pressure
- Pad Speed
- Carrier Speed
- Slurry Flow Rate
- RandomForest
- SHAP
- 반도체 공정 최적화

사용자 질문:
{user_question}

답변은 한국어로 작성하고,
가능하면 이유와 개선 방향까지 설명하세요.
"""
                )

            st.success(response.text)

        except Exception as e:
            st.error(f"오류가 발생했습니다.\n\n{e}")
# ============================================================
# 15. 하단 안내
# ============================================================

st.divider()

st.caption(
    "Prototype for CMP Process Decision Support. "
    "For real fab deployment, validate with controlled DOE data, metrology uncertainty, defectivity, wafer-level uniformity, and process window constraints."
)
