# Streamlit app for the diabetes risk checker
# run with: streamlit run app.py
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

GITHUB_URL = "https://github.com/pamithudulwan892-max/diabetes-risk-predictor"

st.set_page_config(page_title="Diabetes Risk Checker", page_icon=":bar_chart:", layout="centered")


@st.cache_resource
def load():
    return joblib.load("model/model.joblib"), json.loads(Path("model/metrics.json").read_text())


model, meta = load()
AVG = meta["positive_rate"]

AGES = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
        8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80 or older"}
HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
LABELS = {
    "high_blood_pressure": "High blood pressure", "high_cholesterol": "High cholesterol",
    "cholesterol_check_5y": "Cholesterol check in last 5 years", "bmi": "BMI",
    "smoked_100_cigarettes": "Smoked 100+ cigarettes in life", "had_stroke": "Had a stroke",
    "heart_disease_or_attack": "Heart disease or heart attack", "physically_active": "Physically active",
    "fruit_daily": "Fruit every day", "veggies_daily": "Vegetables every day", "heavy_drinker": "Heavy drinking",
    "general_health": "Your general health rating", "bad_mental_health_days": "Bad mental health days",
    "bad_physical_health_days": "Bad physical health days", "difficulty_walking": "Difficulty walking",
    "male": "Male", "age_group": "Age group",
}

st.title("Diabetes Risk Checker")
st.caption("Answer a few health and lifestyle questions to see an estimated risk of prediabetes or diabetes. "
           "Based on a model trained on 253,680 CDC health survey answers.")
st.warning("This is a student machine learning project, not a medical test. Only a blood test (like HbA1c or "
           "fasting glucose) can diagnose diabetes. If you are worried, talk to a doctor.")

tab_pred, tab_perf, tab_about = st.tabs(["Check my risk", "Model performance", "About"])


def yn(col, label, default=False, help=None):
    return 1 if col.radio(label, ["No", "Yes"], index=int(default), horizontal=True, help=help) == "Yes" else 0


with tab_pred:
    with st.form("inputs"):
        st.subheader("About you")
        a, b, c = st.columns(3)
        age = a.selectbox("Age", list(AGES), index=7, format_func=AGES.get)
        sex = b.radio("Sex", ["Female", "Male"], horizontal=True)
        height = c.number_input("Height (cm)", 120, 220, 168)
        a, b = st.columns(2)
        weight = a.number_input("Weight (kg)", 30, 250, 72)
        gen = b.selectbox("General health", list(HEALTH), index=2, format_func=HEALTH.get)

        st.subheader("Medical history")
        a, b = st.columns(2)
        hbp = yn(a, "Told you have high blood pressure?")
        hchol = yn(b, "Told you have high cholesterol?")
        chk = yn(a, "Cholesterol checked in the last 5 years?", default=True)
        heart = yn(b, "Heart disease or heart attack?")
        stroke = yn(a, "Ever had a stroke?")
        walk = yn(b, "Serious difficulty walking or climbing stairs?")

        st.subheader("Lifestyle")
        a, b = st.columns(2)
        active = yn(a, "Exercise in the last 30 days (not job)?", default=True)
        smoke = yn(b, "Smoked at least 100 cigarettes in your life?")
        fruit = yn(a, "Eat fruit at least once a day?", default=True)
        veg = yn(b, "Eat vegetables at least once a day?", default=True)
        drink = yn(a, "Heavy drinker?", help="Men: more than 14 drinks a week. Women: more than 7.")
        a, b = st.columns(2)
        ment = a.slider("Days of poor mental health (last 30)", 0, 30, 0)
        phys = b.slider("Days of poor physical health (last 30)", 0, 30, 0)

        submitted = st.form_submit_button("Check my risk", type="primary", use_container_width=True)

    # ?example=1 shows a sample person so people can see a result without filling the form
    example = st.query_params.get("example") == "1" and not submitted
    if example:
        st.caption("Example: a 55-59 year old woman, 168 cm and 98 kg, with high blood pressure and cholesterol.")
        age, sex, height, weight, gen, hbp, hchol = 8, "Female", 168, 98, 3, 1, 1

    if submitted or example:
        bmi = weight / (height / 100) ** 2
        row = {"high_blood_pressure": hbp, "high_cholesterol": hchol, "cholesterol_check_5y": chk,
               "bmi": round(bmi), "smoked_100_cigarettes": smoke, "had_stroke": stroke,
               "heart_disease_or_attack": heart, "physically_active": active, "fruit_daily": fruit,
               "veggies_daily": veg, "heavy_drinker": drink, "general_health": gen,
               "bad_mental_health_days": ment, "bad_physical_health_days": phys, "difficulty_walking": walk,
               "male": int(sex == "Male"), "age_group": age}
        X = pd.DataFrame([row])
        p = float(model.predict_proba(X)[0, 1])
        ratio = p / AVG

        if p >= meta["threshold"] and ratio >= 2:
            st.error(f"### Higher risk: {p:.0%}")
        elif p >= meta["threshold"]:
            st.warning(f"### Above average risk: {p:.0%}")
        else:
            st.success(f"### Lower risk: {p:.0%}")
        st.write(f"Your BMI is **{bmi:.1f}**. The average risk in the data is {AVG:.0%}, "
                 f"and your estimate is **{ratio:.1f}x** the average.")
        st.progress(min(p, 1.0))

        # what-if: swap one answer at a time for the typical answer and see how the risk moves
        typical = {k: v["default"] for k, v in meta["schema"].items()}
        rows = []
        for k in row:
            if row[k] != typical[k]:
                alt = dict(row, **{k: typical[k]})
                rows.append((k, alt))
        if rows:
            alts = pd.DataFrame([r[1] for r in rows])
            contrib = pd.Series(p - model.predict_proba(alts)[:, 1], index=[r[0] for r in rows])
            up = contrib[contrib > 0.005].sort_values(ascending=False).head(4)
            down = contrib[contrib < -0.005].sort_values().head(3)
            a, b = st.columns(2)
            a.write("**What raised your estimate**")
            for k in up.index:
                a.write(f"- {LABELS[k]}")
            if up.empty:
                a.write("- Nothing stood out")
            b.write("**What lowered it**")
            for k in down.index:
                b.write(f"- {LABELS[k]}")
            if down.empty:
                b.write("- Nothing stood out")
        tips = []
        if bmi >= 25:
            tips.append("BMI is one of the strongest factors. Even losing 5-7% of body weight lowers diabetes risk.")
        if not active:
            tips.append("Regular activity (around 150 minutes a week) helps the body use insulin better.")
        if hbp or hchol:
            tips.append("Blood pressure and cholesterol often go together with diabetes, so regular check-ups matter.")
        if p >= meta["threshold"]:
            tips.append("Your estimate is above the screening cut-off the model uses. A simple blood test can tell you for sure.")
        if tips:
            st.info("\n\n".join(tips))

with tab_perf:
    t = meta["test"]
    st.write(f"**Model:** {meta['best_model']}, picked with 5-fold cross-validation (ROC-AUC). "
             f"Trained on {meta['n_train']:,} people and tested on {meta['n_test']:,} people it never saw.")
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    b.metric("Recall", f"{t['at_screening']['recall']:.0%}")
    c.metric("Precision", f"{t['at_screening']['precision']:.0%}")
    d.metric("Cut-off", f"{meta['threshold']:.2f}")
    st.caption(f"Recall and precision are at the screening cut-off of {meta['threshold']:.2f} instead of 0.5, "
               f"because only {AVG:.0%} of people in the data have diabetes or prediabetes.")
    st.write("**Models compared (cross-validation ROC-AUC):**")
    st.dataframe(pd.Series(meta["cv_scores"], name="CV ROC-AUC").sort_values(ascending=False).round(3))
    for img, cap in [("roc_curve.png", "ROC curve on the test set"),
                     ("confusion_matrix.png", "Confusion matrix at the screening cut-off"),
                     ("feature_importance.png", "Which answers matter most")]:
        if Path("assets", img).exists():
            st.image(f"assets/{img}", caption=cap)

with tab_about:
    st.markdown(f"""
**Data:** [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
on the UCI Machine Learning Repository, from the CDC's BRFSS phone survey of US adults
({meta['n_rows']:,} answers).

**Target:** told by a doctor they have prediabetes or diabetes (about {AVG:.0%} of people).

I only used questions a person can answer about their own health. I left out income, education and
health insurance, which barely changed the score.

**Code:** [{GITHUB_URL}]({GITHUB_URL})
""")
    st.caption("Learning project only. The data is self-reported survey answers from US adults, "
               "so it may not fit other countries well.")
