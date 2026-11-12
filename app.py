import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap

from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Osteoporosis Prediction System",
    page_icon="🦴",
    layout="wide"
)

st.title("🦴 Osteoporosis Prediction System")

st.write(
    "Enter patient information to compare osteoporosis-risk "
    "predictions from six trained machine-learning models."
)


# =====================================================
# LOAD SAVED MODELS
# =====================================================

@st.cache_resource
def load_models():

    # Logistic Regression
    logistic = joblib.load(
        "saved_models/logistic_regression.pkl"
    )

    # Random Forest
    random_forest = joblib.load(
        "saved_models/random_forest.pkl"
    )

    # XGBoost
    xgboost = joblib.load(
        "saved_models/xgboost.pkl"
    )

    # LightGBM
    lightgbm = joblib.load(
        "saved_models/lightgbm.pkl"
    )

    # CatBoost
    catboost = CatBoostClassifier()

    catboost.load_model(
        "saved_models/catboost_model.cbm"
    )

    cat_preprocessor = joblib.load(
        "saved_models/catboost_preprocessor.pkl"
    )

    # TabNet
    tabnet = TabNetClassifier()

    tabnet.load_model(
        "saved_models/tabnet_model.zip"
    )

    tabnet_preprocessor = joblib.load(
        "saved_models/tabnet_preprocessor.pkl"
    )

    # Thresholds
    with open(
        "saved_models/thresholds.json",
        "r"
    ) as f:
        thresholds = json.load(f)

    return (
        logistic,
        random_forest,
        xgboost,
        lightgbm,
        catboost,
        cat_preprocessor,
        tabnet,
        tabnet_preprocessor,
        thresholds
    )


try:

    (
        logistic_model,
        random_forest_model,
        xgb_model,
        lgbm_model,
        catboost_model,
        cat_preprocessor,
        tabnet_model,
        tabnet_preprocessor,
        thresholds
    ) = load_models()

    st.success(
        "✅ All 6 trained models loaded successfully!"
    )

except Exception as e:

    st.error(
        "❌ Error while loading saved models."
    )

    st.exception(e)

    st.stop()


# =====================================================
# PATIENT INPUT FORM
# =====================================================

st.divider()

st.header("Patient Information")


with st.form("patient_form"):

    # =================================================
    # 1. DEMOGRAPHIC INFORMATION
    # =================================================

    st.subheader(
        "1. Demographic Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=50,
            max_value=100,
            value=60
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

    with col3:

        ethnicity = st.selectbox(
            "Race / Ethnicity",
            [
                "Mexican American",
                "Other Hispanic",
                "Non-Hispanic White",
                "Non-Hispanic Black",
                "Non-Hispanic Asian",
                "Other / Multiracial"
            ]
        )

    income_ratio = st.number_input(
        "Income-to-Poverty Ratio",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    )


    # =================================================
    # 2. BODY MEASUREMENTS
    # =================================================

    st.subheader(
        "2. Body Measurements"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=70.0
        )

    with col2:

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=220.0,
            value=165.0
        )

    with col3:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=25.0
        )

    col1, col2 = st.columns(2)

    with col1:

        waist = st.number_input(
            "Waist Circumference (cm)",
            min_value=40.0,
            max_value=200.0,
            value=90.0
        )

    with col2:

        hip = st.number_input(
            "Hip Circumference (cm)",
            min_value=40.0,
            max_value=200.0,
            value=100.0
        )


    # =================================================
    # 3. NUTRITION
    # =================================================

    st.subheader(
        "3. Nutrition"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        calories = st.number_input(
            "Energy Intake (kcal/day)",
            min_value=1.0,
            value=2000.0
        )

        protein = st.number_input(
            "Protein Intake (g/day)",
            min_value=0.0,
            value=70.0
        )

    with col2:

        vitamin_d = st.number_input(
            "Vitamin D Intake",
            min_value=0.0,
            value=5.0
        )

        calcium = st.number_input(
            "Calcium Intake (mg/day)",
            min_value=0.0,
            value=900.0
        )

    with col3:

        phosphorus = st.number_input(
            "Phosphorus Intake (mg/day)",
            min_value=0.0,
            value=1200.0
        )

        magnesium = st.number_input(
            "Magnesium Intake (mg/day)",
            min_value=0.0,
            value=300.0
        )


    # =================================================
    # 4. PHYSICAL ACTIVITY
    # =================================================

    st.subheader(
        "4. Physical Activity"
    )

    col1, col2 = st.columns(2)

    with col1:

        vigorous_work = st.selectbox(
            "Vigorous Work Activity",
            [
                "No",
                "Yes"
            ]
        )

        moderate_work = st.selectbox(
            "Moderate Work Activity",
            [
                "No",
                "Yes"
            ]
        )

        walking = st.selectbox(
            "Walking / Cycling",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        vigorous_recreation = st.selectbox(
            "Vigorous Recreational Activity",
            [
                "No",
                "Yes"
            ]
        )

        moderate_recreation = st.selectbox(
            "Moderate Recreational Activity",
            [
                "No",
                "Yes"
            ]
        )

        sedentary_time = st.number_input(
            "Sedentary Time (minutes/day)",
            min_value=0.0,
            max_value=1440.0,
            value=300.0
        )


    # =================================================
    # 5. LIFESTYLE
    # =================================================

    st.subheader(
        "5. Lifestyle"
    )

    alcohol = st.selectbox(
        "Ever consumed alcohol?",
        [
            "No",
            "Yes"
        ]
    )


    # =================================================
    # 6. MEDICAL HISTORY
    # =================================================

    st.subheader(
        "6. Medical History"
    )

    col1, col2 = st.columns(2)

    with col1:

        overweight = st.selectbox(
            "Overweight History",
            [
                "No",
                "Yes"
            ]
        )

        arthritis = st.selectbox(
            "Arthritis",
            [
                "No",
                "Yes"
            ]
        )

        thyroid = st.selectbox(
            "Thyroid Condition",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        liver = st.selectbox(
            "Liver Condition",
            [
                "No",
                "Yes"
            ]
        )

        emphysema = st.selectbox(
            "Emphysema",
            [
                "No",
                "Yes"
            ]
        )

        cancer = st.selectbox(
            "Cancer History",
            [
                "No",
                "Yes"
            ]
        )


    # =================================================
    # SUBMIT BUTTON
    # =================================================

    submitted = st.form_submit_button(
        "Predict Osteoporosis Risk",
        use_container_width=True
    )


# =====================================================
# RUN PREDICTION AFTER FORM SUBMISSION
# =====================================================

if submitted:

    # =================================================
    # CONVERT READABLE INPUTS TO DATASET CODES
    # =================================================

    gender_map = {
        "Male": 1,
        "Female": 2
    }

    ethnicity_map = {
        "Mexican American": 1,
        "Other Hispanic": 2,
        "Non-Hispanic White": 3,
        "Non-Hispanic Black": 4,
        "Non-Hispanic Asian": 6,
        "Other / Multiracial": 7
    }

    yes_no_map = {
        "No": 0,
        "Yes": 1
    }


    # =================================================
    # CREATE ORIGINAL 28 FEATURES
    # =================================================

    patient = pd.DataFrame([{

        # Demographic
        "RIDAGEYR": age,
        "RIAGENDR": gender_map[gender],
        "RIDRETH3": ethnicity_map[ethnicity],
        "INDFMPIR": income_ratio,

        # Body measurements
        "BMXWT": weight,
        "BMXHT": height,
        "BMXBMI": bmi,
        "BMXWAIST": waist,
        "BMXHIP": hip,

        # Physical activity
        "PAQ605": yes_no_map[vigorous_work],
        "PAQ620": yes_no_map[moderate_work],
        "PAQ635": yes_no_map[walking],
        "PAQ650": yes_no_map[vigorous_recreation],
        "PAQ665": yes_no_map[moderate_recreation],
        "PAD680": sedentary_time,

        # Alcohol
        "Alcohol_Ever": yes_no_map[alcohol],

        # Nutrition
        "DR1TKCAL": calories,
        "DR1TPROT": protein,
        "DR1TVD": vitamin_d,
        "DR1TCALC": calcium,
        "DR1TPHOS": phosphorus,
        "DR1TMAGN": magnesium,

        # Medical history
        "MCQ080": yes_no_map[overweight],
        "MCQ160A": yes_no_map[arthritis],
        "MCQ160M": yes_no_map[thyroid],
        "MCQ160L": yes_no_map[liver],
        "MCQ160P": yes_no_map[emphysema],
        "MCQ220": yes_no_map[cancer]

    }])


    # =================================================
    # CREATE 7 ENGINEERED FEATURES
    # =================================================

    patient["Waist_Height_Ratio"] = (
        patient["BMXWAIST"]
        /
        patient["BMXHT"]
    )

    patient["Waist_Hip_Ratio"] = (
        patient["BMXWAIST"]
        /
        patient["BMXHIP"]
    )

    patient["Age_BMI"] = (
        patient["RIDAGEYR"]
        *
        patient["BMXBMI"]
    )

    patient["Calcium_per_1000kcal"] = (
        patient["DR1TCALC"]
        /
        patient["DR1TKCAL"]
        *
        1000
    )

    patient["VitaminD_per_1000kcal"] = (
        patient["DR1TVD"]
        /
        patient["DR1TKCAL"]
        *
        1000
    )

    patient["Protein_per_kg"] = (
        patient["DR1TPROT"]
        /
        patient["BMXWT"]
    )

    patient["Physical_Activity_Score"] = (
        patient[
            [
                "PAQ605",
                "PAQ620",
                "PAQ635",
                "PAQ650",
                "PAQ665"
            ]
        ].sum(axis=1)
    )


    # =================================================
    # RUN ALL SIX MODELS
    # =================================================

    results = []


    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------

    logistic_prob = (
        logistic_model
        .predict_proba(patient)[0, 1]
    )

    results.append({
        "Model": "Logistic Regression",
        "Probability": logistic_prob,
        "Threshold":
            thresholds["Logistic Regression"]
    })


    # -------------------------------------------------
    # Random Forest
    # -------------------------------------------------

    rf_prob = (
        random_forest_model
        .predict_proba(patient)[0, 1]
    )

    results.append({
        "Model": "Random Forest",
        "Probability": rf_prob,
        "Threshold":
            thresholds["Random Forest"]
    })


    # -------------------------------------------------
    # XGBoost
    # -------------------------------------------------

    xgb_prob = (
        xgb_model
        .predict_proba(patient)[0, 1]
    )

    results.append({
        "Model": "XGBoost",
        "Probability": xgb_prob,
        "Threshold":
            thresholds["XGBoost"]
    })


    # -------------------------------------------------
    # LightGBM
    # -------------------------------------------------

    lgbm_prob = (
        lgbm_model
        .predict_proba(patient)[0, 1]
    )

    results.append({
        "Model": "LightGBM",
        "Probability": lgbm_prob,
        "Threshold":
            thresholds["LightGBM"]
    })


    # -------------------------------------------------
    # CatBoost
    # -------------------------------------------------

    patient_cat = (
        cat_preprocessor
        .transform(patient)
    )

    cat_prob = (
        catboost_model
        .predict_proba(patient_cat)[0, 1]
    )

    results.append({
        "Model": "CatBoost",
        "Probability": cat_prob,
        "Threshold":
            thresholds["CatBoost"]
    })


    # -------------------------------------------------
    # TabNet
    # -------------------------------------------------

    patient_tab = (
        tabnet_preprocessor
        .transform(patient)
    )

    patient_tab = np.asarray(
        patient_tab,
        dtype=np.float32
    )

    tabnet_prob = (
        tabnet_model
        .predict_proba(patient_tab)[0, 1]
    )

    results.append({
        "Model": "TabNet",
        "Probability": tabnet_prob,
        "Threshold":
            thresholds["TabNet"]
    })


    # =================================================
    # CREATE RESULTS TABLE
    # =================================================

    results_df = pd.DataFrame(results)

    results_df["Prediction"] = np.where(
        results_df["Probability"]
        >=
        results_df["Threshold"],
        "Higher Risk",
        "Lower Risk"
    )


    # =================================================
    # DISPLAY RESULTS
    # =================================================

    st.divider()

    st.header(
        "Prediction Results"
    )

    st.dataframe(
        results_df.style.format({
            "Probability": "{:.3f}",
            "Threshold": "{:.2f}"
        }),
        use_container_width=True
    )


    # =================================================
    # MODEL AGREEMENT
    # =================================================

    positive_models = (
        results_df["Prediction"]
        ==
        "Higher Risk"
    ).sum()

    st.subheader(
        "Model Agreement"
    )

    st.write(
        f"**{positive_models} out of 6 models** "
        f"predict higher osteoporosis risk."
    )


    # =================================================
    # PROBABILITY COMPARISON
    # =================================================

    st.subheader(
        "Probability Comparison"
    )

    chart_df = (
        results_df[
            [
                "Model",
                "Probability"
            ]
        ]
        .set_index("Model")
    )

    st.bar_chart(
        chart_df
    )


    # =================================================
    # XGBOOST PATIENT-SPECIFIC SHAP EXPLANATION
    # =================================================

    st.divider()

    st.header(
        "Why this prediction?"
    )

    st.write(
        "The following factors had the strongest "
        "influence on the XGBoost prediction for "
        "this patient."
    )

    try:

        # ---------------------------------------------
        # Extract XGBoost preprocessor and classifier
        # ---------------------------------------------

        xgb_preprocessor = (
            xgb_model
            .named_steps["preprocessor"]
        )

        xgb_classifier = (
            xgb_model
            .named_steps["classifier"]
        )


        # ---------------------------------------------
        # Transform current patient
        # ---------------------------------------------

        patient_xgb = (
            xgb_preprocessor
            .transform(patient)
        )


        # ---------------------------------------------
        # Get transformed feature names
        # ---------------------------------------------

        xgb_feature_names = (
            xgb_preprocessor
            .get_feature_names_out()
        )


        # ---------------------------------------------
        # SHAP explainer
        # ---------------------------------------------

        explainer = shap.TreeExplainer(
            xgb_classifier
        )

        shap_values = (
            explainer
            .shap_values(patient_xgb)
        )


        # ---------------------------------------------
        # Handle SHAP output formats
        # ---------------------------------------------

        if isinstance(
            shap_values,
            list
        ):
            shap_values = (
                shap_values[-1]
            )

        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim > 1:
            patient_shap = (
                shap_values[0]
            )
        else:
            patient_shap = (
                shap_values
            )


        # ---------------------------------------------
        # Readable feature names
        # ---------------------------------------------

        readable_map = {

            # Demographic
            "num__RIDAGEYR":
                "Age",

            "num__INDFMPIR":
                "Income-to-Poverty Ratio",

            # Body measurements
            "num__BMXWT":
                "Weight",

            "num__BMXHT":
                "Height",

            "num__BMXBMI":
                "BMI",

            "num__BMXWAIST":
                "Waist Circumference",

            "num__BMXHIP":
                "Hip Circumference",

            # Activity
            "num__PAD680":
                "Sedentary Time",

            # Nutrition
            "num__DR1TKCAL":
                "Energy Intake",

            "num__DR1TPROT":
                "Protein Intake",

            "num__DR1TVD":
                "Vitamin D Intake",

            "num__DR1TCALC":
                "Calcium Intake",

            "num__DR1TPHOS":
                "Phosphorus Intake",

            "num__DR1TMAGN":
                "Magnesium Intake",

            # Engineered
            "num__Waist_Height_Ratio":
                "Waist-to-Height Ratio",

            "num__Waist_Hip_Ratio":
                "Waist-to-Hip Ratio",

            "num__Age_BMI":
                "Age × BMI",

            "num__Calcium_per_1000kcal":
                "Calcium per 1000 kcal",

            "num__VitaminD_per_1000kcal":
                "Vitamin D per 1000 kcal",

            "num__Protein_per_kg":
                "Protein per kg",

            "num__Physical_Activity_Score":
                "Physical Activity Score",

            # Gender
            "cat__RIAGENDR_1.0":
                "Gender - Male",

            "cat__RIAGENDR_2.0":
                "Gender - Female",

            # Ethnicity
            "cat__RIDRETH3_1.0":
                "Mexican American",

            "cat__RIDRETH3_2.0":
                "Other Hispanic",

            "cat__RIDRETH3_3.0":
                "Non-Hispanic White",

            "cat__RIDRETH3_4.0":
                "Non-Hispanic Black",

            "cat__RIDRETH3_6.0":
                "Non-Hispanic Asian",

            "cat__RIDRETH3_7.0":
                "Other / Multiracial",

            # Activity binary
            "bin__PAQ605":
                "Vigorous Work Activity",

            "bin__PAQ620":
                "Moderate Work Activity",

            "bin__PAQ635":
                "Walking / Cycling",

            "bin__PAQ650":
                "Vigorous Recreational Activity",

            "bin__PAQ665":
                "Moderate Recreational Activity",

            # Alcohol
            "bin__Alcohol_Ever":
                "Alcohol Use",

            # Medical
            "bin__MCQ080":
                "Overweight History",

            "bin__MCQ160A":
                "Arthritis",

            "bin__MCQ160M":
                "Thyroid Condition",

            "bin__MCQ160L":
                "Liver Condition",

            "bin__MCQ160P":
                "Emphysema",

            "bin__MCQ220":
                "Cancer History"
        }


        # ---------------------------------------------
        # Create explanation table
        # ---------------------------------------------

        explanation_df = pd.DataFrame({

            "Feature": [
                readable_map.get(
                    feature,
                    feature
                )
                for feature
                in xgb_feature_names
            ],

            "SHAP":
                patient_shap
        })


        explanation_df["Impact"] = (
            explanation_df["SHAP"]
            .abs()
        )


        # ---------------------------------------------
        # Top 8 patient-specific factors
        # ---------------------------------------------

        top_factors = (
            explanation_df
            .sort_values(
                "Impact",
                ascending=False
            )
            .head(8)
            .copy()
        )


        top_factors["Direction"] = np.where(

            top_factors["SHAP"] > 0,

            "Toward Higher Risk",

            "Toward Lower Risk"
        )


        # ---------------------------------------------
        # Display explanation
        # ---------------------------------------------

        st.dataframe(

            top_factors[
                [
                    "Feature",
                    "SHAP",
                    "Direction"
                ]
            ].style.format({
                "SHAP": "{:.3f}"
            }),

            use_container_width=True
        )


        st.caption(
            "Positive SHAP values push the XGBoost "
            "prediction toward higher predicted risk. "
            "Negative SHAP values push the prediction "
            "toward lower predicted risk."
        )


    except Exception as e:

        st.warning(
            "XGBoost explanation could not be generated."
        )

        st.exception(e)


    # =================================================
    # DISCLAIMER
    # =================================================

    st.divider()

    st.info(
        "This application is intended for research "
        "and educational risk-screening purposes only. "
        "It does not provide a clinical diagnosis and "
        "should not replace professional medical "
        "assessment or DXA testing."
    )