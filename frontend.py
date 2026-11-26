import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter a passenger's details and the model will predict whether they would have "
    "survived the Titanic disaster. The prediction comes from the FastAPI service."
)

with st.sidebar:
    st.header("API status")
    st.write(f"API URL: `{API_URL}`")
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        response.raise_for_status()
        st.success(response.json().get("message", "API is running"))
    except requests.exceptions.RequestException:
        st.error("API is not reachable. Start it with:\n\n"
                 "`python -m uvicorn main:app --reload`\n\n"
                 "or `docker run -p 8000:8000 titanic-api`")

with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Ticket class", options=[1, 2, 3], index=2,
            format_func=lambda c: {1: "1st (upper)", 2: "2nd (middle)", 3: "3rd (lower)"}[c],
        )
        sex = st.radio("Sex", options=["male", "female"], horizontal=True)
        age_unknown = st.checkbox("Age unknown")
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0, step=1.0,
                              disabled=age_unknown)

    with col2:
        sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0, step=1)
        fare = st.number_input("Fare (£)", min_value=0.0, max_value=600.0, value=7.25, step=1.0)
        embarked = st.selectbox(
            "Port of embarkation", options=["S", "C", "Q", None],
            format_func=lambda p: {"S": "Southampton", "C": "Cherbourg",
                                   "Q": "Queenstown", None: "Unknown"}[p],
        )

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = {
        "pclass": pclass,
        "sex": sex,
        "age": None if age_unknown else age,
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": fare,
        "embarked": embarked,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.RequestException:
        st.error(f"Could not connect to the API at {API_URL}. Is it running?")
        st.stop()

    if response.status_code != 200:
        st.error(f"The API returned an error ({response.status_code}).")
        st.json(response.json())
        st.stop()

    result = response.json()
    probability = result["survival_probability"]

    st.subheader("Result")
    if result["survived"] == 1:
        st.success(f"✅ **{result['prediction']}**")
    else:
        st.error(f"❌ **{result['prediction']}**")

    st.metric("Probability of survival", f"{probability:.1%}")
    st.progress(probability)

    with st.expander("Request and response"):
        st.write("Sent to `POST /predict`:")
        st.json(payload)
        st.write("Received:")
        st.json(result)
