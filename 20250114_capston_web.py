import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
import os
import streamlit as st

# Load the required files
model = load('241225_rf_model.pkl')
scaler = load('241225_rf_scaler.pkl')
with open('241225_selected_features.json', 'r') as f:
    features = pd.read_json(f, typ='series')
shap_values = load('241225_shap_rf.pkl')
summary_stats = pd.read_csv('241225_summary_stats.csv')
explainer = load("241225_shap_rf_explainer.pkl")


# 한글 폰트 설정 (예: Apple SD Gothic Neo for macOS)
rc('font', family='Apple SD Gothic Neo')  # macOS의 경우
# rc('font', family='Malgun Gothic')  # Windows의 경우
# rc('font', family='NanumGothic')  # Linux의 경우

# 유니코드에서 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# Streamlit 세션 상태 초기화
if "loan_data" not in st.session_state:
    st.session_state.loan_data = []

# 입력 화면 (왼쪽)
st.sidebar.title("대출 데이터 입력")

loan_amount = st.sidebar.number_input("대출 금액 (₩)", min_value=0, max_value=100000000, value=5000000, step=100000)
# 사용자 입력: 대출금리
loan_rate = st.sidebar.slider("대출 금리 (%)", min_value=0, max_value=24, value=12, step=1)
age = st.sidebar.number_input("나이", min_value=18, max_value=100, value=30)
gender = st.sidebar.radio("성별", ["남성", "여성"])
delinquency = st.sidebar.checkbox("연체 여부", value=False)
loan_type = st.sidebar.selectbox("대출 종류", [
    "부동산 관련 담보대출", "기타할부 및 리스", "신용대출", 
    "신차 할부", "중고차 할부", "카드대출", "부동산 외 담보대출"
])
upkwon = st.sidebar.selectbox("업권 코드", ["1금융권", "2금융권", "3금융권"])

if st.sidebar.button("대출데이터 입력"):
    # 입력된 데이터를 세션 상태에 저장
    new_data = {
        "대출종류": loan_type,
        "업권": upkwon,
        '대출금액':loan_amount,
        '대출건수': 1, 
        '나이':age, 
        '대출금리': loan_rate, 
       '성별_여성': 1 if gender == "여성" else 0,
       "연체여부": int(delinquency),
    }
    st.session_state.loan_data.append(new_data)
    st.sidebar.success("대출 데이터가 입력되었습니다.")

# session_state 초기화
if "ins_data" not in st.session_state:
    st.session_state["ins_data"] = []

ins_abnormal = st.sidebar.number_input("보험실효건수", min_value=0, max_value=100, value=0)
ins_type = st.sidebar.multiselect("보험 종류", [
    '간병(요양)보험', 
       '기타', 
       '상해보험', 
       '암보험', 
       '어린이보험', 
       '여행자보험', 
       '운전자보험', 
       '종신보험',
       '질병(건강)보험', 
       '치아보험', 
])


if st.sidebar.button("보험데이터 입력"):
    # 입력된 데이터를 세션 상태에 저장
    new_data_ins = {
        "보험 종류": ins_type,
        "보험정상건수": len(ins_type),
        "보험실효건수": ins_abnormal,
    }
    st.session_state.ins_data.append(new_data_ins)
    st.sidebar.success("보험 데이터가 입력되었습니다.")


# 데이터 표시 화면 (오른쪽)
st.title("입력된 대출 및 보험 데이터")
loan_df = pd.DataFrame(st.session_state.loan_data)
ins_df = pd.DataFrame(st.session_state.ins_data)


if not loan_df.empty:
    st.dataframe(loan_df)
    if not ins_df.empty:
        st.dataframe(ins_df)

###################데이터 정리
# 제출 버튼

def log_transform(data, columns):
    transformed_data = data.copy()
    for column in columns:
        transformed_data[column] = np.log1p(transformed_data[column])  # log1p는 log(1+x) 변환
    return transformed_data


if st.button("제출"):
    if st.session_state.loan_data and st.session_state.ins_data:
        # 대출 데이터 정리
        loan_df = pd.DataFrame(st.session_state.loan_data)
        ins_df = pd.DataFrame(st.session_state.ins_data)

        # 대출 데이터 관련 변수 생성
        aggregated_data = {
            '대출금액': loan_df['대출금액'].sum() / 1000,
            '대출건수': len(loan_df),
            '나이': loan_df['나이'].iloc[0] if not loan_df.empty else np.nan,
            '평균금리': (loan_df['대출금액'] * loan_df['대출금리']).sum() / loan_df['대출금액'].sum() / 12 * 1000 if loan_df['대출금액'].sum() > 0 else 0,
            '성별_여성': loan_df['성별_여성'].iloc[0] if not loan_df.empty else 0
        }

        # 대출 종류별 개수 계산
        loan_types = [
            "기타할부 및 리스", "부동산 관련 담보대출", "부동산 외 담보대출",
            "신용대출", "신차 할부", "중고차 할부", "카드대출"
        ]
        for loan_type in loan_types:
            aggregated_data[f'대출세부종류_{loan_type}'] = len(loan_df[loan_df['대출종류'] == loan_type])

        # 업권 코드별 개수 계산
        upkwon_types = ["1금융권", "2금융권", "3금융권"]
        aggregated_data['고유기관수'] = loan_df['업권'].nunique()
        for upkwon in upkwon_types:
            aggregated_data[f'업권코드_{upkwon}'] = len(loan_df[loan_df['업권'] == upkwon])

        # 보험 데이터 관련 변수 생성
        aggregated_data['보험실효건수'] = ins_df['보험실효건수'].sum() if not ins_df.empty else 0
        aggregated_data['보험정상건수'] = ins_df['보험정상건수'].sum() if not ins_df.empty else 0

        # 보험 종류별 개수 계산
        insurance_types = [
            '간병(요양)보험', '기타', '상해보험', '암보험', '어린이보험', '여행자보험',
            '운전자보험', '종신보험', '질병(건강)보험', '치아보험'
        ]
        for ins_type in insurance_types:
            aggregated_data[ins_type] = sum([ins_type in types for types in ins_df['보험 종류']])
        # 추가 변수 초기화
        additional_vars = [
            '평균추가대출시간_월', '금리경로_저금리 → 저금리 → 저금리_여부', '금리경로_저금리 → 저금리_여부',
            '대출세부업권경로_신용협동기구_여부', '대출세부업권경로_대부업권_여부',
            '대출세부업권경로_할부금융사_여부', '대출세부업권경로_국내은행 → 국내은행_여부',
            '대출세부업권경로_국내은행_여부', '대출세부종류경로_신용대출 → 부동산 관련 담보대출_여부'
        ]
        for var in additional_vars:
            aggregated_data[var] = 0

        # 데이터프레임으로 변환
        final_df = pd.DataFrame([aggregated_data])

        
        columns_to_log_transform = [
        '대출금액',
        '대출건수', 
        '나이', 
        '평균금리', 
        '대출세부종류_기타할부 및 리스',
        '대출세부종류_부동산 관련 담보대출', 
        '대출세부종류_부동산 외 담보대출', 
        '대출세부종류_신용대출',
        '대출세부종류_신차 할부', 
        '대출세부종류_중고차 할부', 
        '대출세부종류_카드대출', 
        '고유기관수',
        '업권코드_1금융권', 
        '업권코드_2금융권', 
        '업권코드_3금융권', 
        '보험실효건수', 
        '보험정상건수',
       '간병(요양)보험', 
       '기타', 
       '상해보험', 
       '암보험', 
       '어린이보험', 
       '여행자보험', 
       '운전자보험', 
       '종신보험',
       '질병(건강)보험', 
       '치아보험',
        '평균추가대출시간_월' ]
        # 정규화 수행
        scaled_data = final_df.copy()
        scaled_data[columns_to_log_transform] = scaler.transform(scaled_data[columns_to_log_transform])
        scaled_df = pd.DataFrame(scaled_data, columns=final_df.columns)

        # 세션 상태에 저장
        st.session_state['final_df'] = final_df
        st.session_state['scaled_data'] = scaled_data

        # 정리된 데이터와 정규화된 데이터 출력
        st.write("#### 정리된 데이터(단위수정)")
        st.dataframe(final_df)

        st.write("#### 정규화된 데이터")
        st.dataframe(scaled_df)
    else:
        st.warning("대출 데이터 또는 보험 데이터가 부족합니다.")

# 분석 버튼
if st.button("분석"):
    if "final_df" in st.session_state and "scaled_data" in st.session_state:
        final_df = st.session_state['final_df']
        scaled_data = st.session_state['scaled_data']

        # 1. 입력한 데이터를 모델에 입력하여 연체확률 추출
        probabilities = model.predict_proba(scaled_data)[:, 1]  # 연체 확률 (1 클래스)
        delinquency_prob = probabilities[0]
        
       # 2. 연체확률 단계 출력 및 Funnel Chart로 시각화
        if delinquency_prob <= 0.007832:
            current_stage = "4단계(상위 10%)"
        elif 0.007832 < delinquency_prob <= 0.058911:
            current_stage = "3단계(상위10~40%)"
        elif 0.058911 < delinquency_prob <= 0.105027:
            current_stage = "2단계(상위40~70%)"
        else:
            current_stage = "1단계(상위70~100%)"

        st.write(f"# 나의 현재 단계: {current_stage}")
        st.write(f"###### 연체 확률: {delinquency_prob:.2%}")

        # 단계별 평균 금리 데이터
        stage_data = {
            "단계": ["1단계(상위70~100%)", "2단계(상위40~70%)", "3단계(상위10~40%)", "4단계(상위 10%)"],
            "평균 금리 (%)": [9.3, 5.1, 3.9, 3.1]  # 예제 데이터
        }
        df_stages = pd.DataFrame(stage_data)

        # 단계를 역순으로 정렬
        df_stages = df_stages.iloc[::-1].reset_index(drop=True)

        # 단계별 색상 (선택된 단계는 진한 색상)
        stage_colors = ["#FF6347", "#FFD700", "#90EE90", "#87CEEB"]
        colors = [
            stage_colors[i] if stage == current_stage else "rgba(211, 211, 211, 0.5)"
            for i, stage in enumerate(df_stages["단계"])
        ]

        # Plotly Funnel Chart 생성
        fig = go.Figure()

        fig.add_trace(go.Funnel(
            y=df_stages["단계"],
            x=df_stages["평균 금리 (%)"],
            text=[f"평균 연금리 {val}%" for val in df_stages["평균 금리 (%)"]],  # % 추가
            textinfo="text",  # 자동 비율 대신 텍스트만 표시
            marker=dict(
                color=colors,  # 동적 색상 설정
                line=dict(color="black", width=2)  # 테두리 추가
            )
        ))

        # 레이아웃 업데이트
        fig.update_layout(
            #title="단계별 평균 금리",
            funnelmode="stack",
        )

        # Streamlit에서 그래프 표시
        st.plotly_chart(fig, use_container_width=True)


        ########## 그룹별 특성 제시
        ### 상위10%
        group_paths_4 = {
            "대출 경로": ["부동산 관련 담보대출", "부동산 관련 담보대출 → 신차 할부"],
            "업권 경로": ["국내은행", "신용협동기구"],
            "금리 경로": ["저금리", "저금리 → 저금리"],
            "추가 대출 기간": ["28.8개월", ""]
        }
        group_paths_4_df = pd.DataFrame(group_paths_4)
        ### 10~40%
        group_paths_3 = {
            "대출 경로": ["신차 할부", "부동산 외 담보대출"],
            "업권 경로": ["할부금융사", "국내은행"],
            "금리 경로": ["저금리", "고금리"],
            "추가 대출 기간": ["21.1개월", ""]
        }
        group_paths_3_df = pd.DataFrame(group_paths_3)

        ### 40~70%
        group_paths_2 = {
            "대출 경로": ["신용대출", "카드대출"],
            "업권 경로": ["국내은행", "대부업권"],
            "금리 경로": ["고금리", "저금리"],
            "추가 대출 기간": ["13.4개월", ""]
        }
        group_paths_2_df = pd.DataFrame(group_paths_2)
        
        ### 70~100%
        group_paths_1 = {
            "대출 경로": ["카드대출", "신용대출"],
            "업권 경로": ["신용카드사", "대부업권"],
            "금리 경로": ["고금리", "고금리 → 고금리"],
            "추가 대출 기간": ["7.5개월", ""]
        }
        group_paths_1_df = pd.DataFrame(group_paths_1)
        
        # 그룹 데이터를 저장
        group_data = {
            "1단계(상위70~100%)": group_paths_1_df,
            "2단계(상위40~70%)": group_paths_2_df,
            "3단계(상위10~40%)": group_paths_3_df,
            "4단계(상위 10%)": group_paths_4_df,
        }
        # 현재 단계와 한 단계 다음 단계 선택
        stages = list(group_data.keys())  # ["4단계", "3단계", "2단계", "1단계"]
        current_index = stages.index(current_stage)

        # 현재 단계 테이블 출력
        st.write("## 금융 경로 안내")
        st.write(f"###### 내가 속한 {current_stage} 그룹의 주요 경로")
        st.dataframe(group_data[current_stage])

        # 다음 단계가 있는 경우 출력
        if current_index + 1 < len(stages):
            next_stage = stages[current_index + 1]
            st.write(f"###### 다음 단계 {next_stage} 그룹의 주요 경로")
            st.dataframe(group_data[next_stage])


        # 4. SHAP 나의 점수에 영향을 미친 항목 시각화
        st.write("## 나의 점수에 영향을 미친 항목")

        # SHAP 값 생성
        shap_values = explainer(final_df)
        shap_values_class_1 = shap_values[..., 1]  # 클래스 1의 SHAP 값
        # Matplotlib의 figure와 ax를 명시적으로 생성
        fig, ax = plt.subplots(figsize=(10, 6))     

        beeswarm = shap.plots.beeswarm(shap_values_class_1, max_display=10)
        # Streamlit에 안전하게 플롯 전달
        st.pyplot(fig)
        
