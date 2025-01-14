import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load

#!/usr/pin/python3

import os
import streamlit as st

current_dir = os.getcwd()
st.write(f"현재 작업 디렉토리: {current_dir}")


# Load the required files
model = load('./241225_rf_model_외국인.pkl')
scaler = load('./241225_rf_scaler_외국인.pkl')
with open('./241225_selected_features_외국인.json', 'r') as f:
    features = pd.read_json(f, typ='series')
shap_values = load('./241225_shap_rf_외국인.pkl')
summary_stats = pd.read_csv('./241225_summary_stats_외국인.csv')
explainer = load("./241225_shap_rf_explainer_외국인.pkl")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from joblib import load

# 모델과 스케일러 로드
model = load('[반출]241226(총10개)/241225_rf_model_외국인.pkl')
scaler = load('[반출]241226(총10개)/241225_rf_scaler_외국인.pkl')

# Streamlit 세션 상태 초기화
if "loan_data" not in st.session_state:
    st.session_state.loan_data = []

# 입력 화면 (왼쪽)
st.sidebar.title("대출 데이터 입력")
loan_amount = st.sidebar.number_input("대출 금액 (₩)", min_value=0, max_value=100000000, value=5000000, step=100000)
age = st.sidebar.number_input("나이", min_value=18, max_value=100, value=30)
gender = st.sidebar.radio("성별", ["남성", "여성"])
delinquency = st.sidebar.checkbox("연체 여부", value=False)
loan_type = st.sidebar.selectbox("대출 종류", [
    "부동산 관련 담보대출", "기타할부 및 리스", "신용대출", 
    "신차 할부", "중고차 할부", "카드대출", "부동산 외 담보대출"
])
upkwon = st.sidebar.selectbox("업권 코드", ["1금융권", "2금융권", "3금융권"])

if st.sidebar.button("입력"):
    # 입력된 데이터를 세션 상태에 저장
    new_data = {
        "loan_amount": loan_amount,
        "age": age,
        "gender": 1 if gender == "남성" else 0,
        "delinquency": int(delinquency),
        "loan_type": loan_type,
        "upkwon": upkwon,
    }
    st.session_state.loan_data.append(new_data)
    st.sidebar.success("데이터가 입력되었습니다!")

# 데이터 표시 화면 (오른쪽)
st.title("입력된 대출 데이터")
loan_df = pd.DataFrame(st.session_state.loan_data)
if not loan_df.empty:
    st.dataframe(loan_df)

# 제출 버튼
if st.button("제출"):
    if loan_df.empty:
        st.error("제출할 데이터가 없습니다. 먼저 데이터를 입력하세요.")
    else:
        # 전처리 수행
        loan_df["loan_amount_scaled"] = scaler.transform(loan_df[["loan_amount"]])  # 예시로 loan_amount만 스케일링
        loan_df_encoded = pd.get_dummies(loan_df, columns=["loan_type", "upkwon"])  # 원핫 인코딩 예시

        # 모델 예측
        predictions = model.predict(loan_df_encoded)

        # 결과 출력
        loan_df["Prediction"] = predictions
        st.success("모델 예측이 완료되었습니다!")
        st.write(loan_df)

        # SHAP 설명 출력
        st.subheader("SHAP 설명")
        explainer = load("[반출]241226(총10개)/241225_shap_rf_explainer_외국인.pkl")
        shap_values = explainer(loan_df_encoded)
        st.pyplot(shap.summary_plot(shap_values, loan_df_encoded))
