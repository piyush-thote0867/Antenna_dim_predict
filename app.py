# ============================================================
# ANTENNA INVERSE DESIGN
# Streamlit + XGBoost + Differential Evolution
# ============================================================

import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Antenna Inverse Design",
    page_icon="📡",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("📡 Antenna Inverse Design")

st.write(
    "XGBoost surrogate model + Differential Evolution "
    "for slotted microstrip patch antenna optimization."
)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

model = XGBRegressor()
model.load_model("xgboost_s11_model.json")


# ============================================================
# FEATURES
# ============================================================

features = [
    "width substrate in mm",
    "length impedance line in mm",
    "width slot in mm",
    "length slot in mm",
    "patch width in mm",
    "substrate length in mm",
    "patch legth in mm",
    "Frequency (GHz)"
]


# ============================================================
# USER INPUT
# ============================================================

st.subheader("Design Requirements")

target_frequency = st.number_input(
    "Target Frequency (GHz)",
    min_value=1.8,
    max_value=2.8,
    value=2.4,
    step=0.01
)

target_s11 = st.number_input(
    "Required S11 (dB)",
    min_value=-60.0,
    max_value=0.0,
    value=-10.0,
    step=1.0
)


# ============================================================
# DIMENSION BOUNDS
# ============================================================

bounds = [
    (41.0, 45.0),      # width substrate
    (3.0, 10.0),       # length impedance line
    (1.0, 5.0),        # width slot
    (5.0, 10.0),       # length slot
    (40.0, 44.0),      # patch width
    (38.0, 42.0),      # substrate length
    (28.0, 29.3)       # patch length
]


# ============================================================
# OPTIMIZATION FUNCTION
# ============================================================

def optimize_antenna(target_frequency):

    def objective(x):

        candidate = pd.DataFrame([[
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            target_frequency
        ]], columns=features)

        predicted_s11 = model.predict(candidate)[0]

        return predicted_s11

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=200,
        popsize=15,
        tol=1e-7,
        seed=42,
        workers=1
    )

    return result.x, result.fun


# ============================================================
# OPTIMIZE BUTTON
# ============================================================

if st.button("🚀 Optimize Antenna"):

    with st.spinner("Optimizing antenna geometry..."):

        best, predicted_s11 = optimize_antenna(
            target_frequency
        )

    st.success("Optimization completed!")


    # ========================================================
    # OPTIMIZED DIMENSIONS
    # ========================================================

    st.subheader("Optimized Antenna Dimensions")

    results = pd.DataFrame({
        "Parameter": [
            "Width substrate",
            "Length impedance line",
            "Width slot",
            "Length slot",
            "Patch width",
            "Substrate length",
            "Patch length"
        ],

        "Value (mm)": [
            best[0],
            best[1],
            best[2],
            best[3],
            best[4],
            best[5],
            best[6]
        ]
    })

    results["Value (mm)"] = results["Value (mm)"].round(4)

    st.table(results)


    # ========================================================
    # PREDICTED PERFORMANCE
    # ========================================================

    st.subheader("Predicted Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Target Frequency",
            f"{target_frequency:.3f} GHz"
        )

    with col2:

        st.metric(
            "Predicted S11",
            f"{predicted_s11:.2f} dB"
        )


    # ========================================================
    # REQUIREMENT CHECK
    # ========================================================

    if predicted_s11 <= target_s11:

        st.success(
            f"✓ Required S11 achieved: "
            f"{predicted_s11:.2f} dB ≤ {target_s11:.2f} dB"
        )

    else:

        st.error(
            f"✗ Required S11 not achieved: "
            f"{predicted_s11:.2f} dB > {target_s11:.2f} dB"
        )