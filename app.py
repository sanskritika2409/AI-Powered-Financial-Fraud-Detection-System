import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = Path(__file__).parent / "Fraud_detection_pipeline.pkl"


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None, (
            "Fraud_detection_pipeline.pkl was not found. "
            "Place the .pkl file in the same folder as app.py."
        )

    try:
        trained_model = joblib.load(MODEL_PATH)
        return trained_model, None

    except Exception as e:
        return None, str(e)


model, model_error = load_model()


# =========================================================
# TRANSACTION TYPE MAPPING
# =========================================================
# Your saved pipeline expects numeric values 0-4 for "type".
#
# Standard alphabetical mapping used for the common
# PaySim-style transaction types:
#
# CASH_IN  -> 0
# CASH_OUT -> 1
# DEBIT    -> 2
# PAYMENT  -> 3
# TRANSFER -> 4
#
# The uploaded pipeline's encoder contains categories:
# [0, 1, 2, 3, 4]
# =========================================================

TRANSACTION_TYPES = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827,
            #1f2937
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin-bottom: 8px;
    }

    .hero p {
        color: #d1d5db;
        font-size: 16px;
    }

    .risk-high {
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
        padding: 18px;
        border-radius: 10px;
        color: #991b1b;
        margin-top: 15px;
    }

    .risk-medium {
        background-color: #fef3c7;
        border-left: 6px solid #f59e0b;
        padding: 18px;
        border-radius: 10px;
        color: #92400e;
        margin-top: 15px;
    }

    .risk-low {
        background-color: #dcfce7;
        border-left: 6px solid #16a34a;
        padding: 18px;
        border-radius: 10px;
        color: #166534;
        margin-top: 15px;
    }

    .info-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        background: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DEMO DATA FOR DASHBOARD
# =========================================================

@st.cache_data
def create_demo_data():

    rng = np.random.default_rng(42)

    n = 250

    df = pd.DataFrame({

        "Transaction ID":
            [f"TXN-{100001 + i}" for i in range(n)],

        "step":
            rng.integers(1, 100, n),

        "type":
            rng.integers(0, 5, n),

        "amount":
            rng.integers(100, 100000, n).astype(float),

        "oldbalanceOrg":
            rng.integers(0, 150000, n).astype(float),

        "newbalanceOrig":
            rng.integers(0, 150000, n).astype(float),

        "oldbalanceDest":
            rng.integers(0, 200000, n).astype(float),

        "newbalanceDest":
            rng.integers(0, 200000, n).astype(float)
    })

    features = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    if model is not None:

        try:

            df["Prediction"] = model.predict(
                df[features]
            )

            if hasattr(model, "predict_proba"):

                df["Fraud Probability"] = (
                    model.predict_proba(
                        df[features]
                    )[:, 1]
                )

            else:

                df["Fraud Probability"] = (
                    df["Prediction"].astype(float)
                )

        except Exception:

            df["Prediction"] = 0
            df["Fraud Probability"] = 0.0

    else:

        df["Prediction"] = 0
        df["Fraud Probability"] = 0.0

    df["Status"] = np.where(
        df["Prediction"] == 1,
        "Fraud",
        "Legitimate"
    )

    return df


data = create_demo_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💳 FraudGuard")

st.sidebar.caption(
    "AI-Powered Financial Security"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Predict",
        "📊 Dashboard",
        "📋 Transactions",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("System Status")

if model is not None:

    st.sidebar.success(
        "● Model Online"
    )

    st.sidebar.caption(
        "Logistic Regression Pipeline"
    )

else:

    st.sidebar.error(
        "● Model Offline"
    )

    st.sidebar.caption(
        "Model file not loaded"
    )

st.sidebar.divider()

st.sidebar.caption(
    "Version 2.0"
)

st.sidebar.caption(
    "Powered by Python + Scikit-learn + Streamlit"
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">

        <h1>💳 Financial Fraud Detection System</h1>

        <p>
        AI-powered transaction monitoring designed to identify
        suspicious financial activities and help reduce fraud.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:

        st.warning(
            "⚠️ The trained model could not be loaded."
        )

    total_transactions = len(data)

    fraud_transactions = int(
        (data["Prediction"] == 1).sum()
    )

    fraud_rate = (
        fraud_transactions /
        total_transactions *
        100
    )

    total_amount = data["amount"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Detected",
        f"{fraud_transactions:,}"
    )

    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    col4.metric(
        "Transaction Volume",
        f"₹{total_amount:,.0f}"
    )

    st.divider()

    st.subheader(
        "📌 How the System Works"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown("### 1️⃣ Input")

        st.write(
            "Transaction details are entered into the system."
        )

    with c2:

        st.markdown("### 2️⃣ Processing")

        st.write(
            "The trained pipeline preprocesses the transaction."
        )

    with c3:

        st.markdown("### 3️⃣ Prediction")

        st.write(
            "The machine learning model calculates fraud probability."
        )

    with c4:

        st.markdown("### 4️⃣ Decision")

        st.write(
            "The transaction receives a fraud prediction and risk level."
        )

    st.divider()

    st.subheader(
        "📈 Transaction Activity"
    )

    recent = data.head(30)

    fig = px.line(
        recent,
        x="Transaction ID",
        y="amount",
        markers=True,
        title="Recent Transaction Amounts"
    )

    fig.update_layout(
        xaxis_title="Transaction",
        yaxis_title="Amount (₹)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PREDICT
# =========================================================

elif page == "🔍 Predict":

    st.title(
        "🔍 Fraud Prediction"
    )

    st.write(
        "Enter transaction details to analyze its fraud risk."
    )

    if model is None:

        st.error(
            "The trained model could not be loaded."
        )

        if model_error:

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    model_error
                )

        st.stop()

    st.divider()

    st.subheader(
        "💳 Transaction Information"
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

    with col1:

        step = st.number_input(
            "Transaction Step",
            min_value=1,
            value=1,
            step=1,
            help="Time step of the transaction."
        )

        transaction_type_name = st.selectbox(
            "Transaction Type",
            list(TRANSACTION_TYPES.keys())
        )

        transaction_type = TRANSACTION_TYPES[
            transaction_type_name
        ]

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        old_balance_org = st.number_input(
            "Sender Balance Before (₹)",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )

    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

    with col2:

        new_balance_orig = st.number_input(
            "Sender Balance After (₹)",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        old_balance_dest = st.number_input(
            "Receiver Balance Before (₹)",
            min_value=0.0,
            value=20000.0,
            step=100.0
        )

        new_balance_dest = st.number_input(
            "Receiver Balance After (₹)",
            min_value=0.0,
            value=25000.0,
            step=100.0
        )

    st.divider()

    st.caption(
        f"Model input type value: {transaction_type}"
    )

    if st.button(
        "🚨 Analyze Transaction",
        type="primary",
        use_container_width=True
    ):

        # -------------------------------------------------
        # CREATE MODEL INPUT
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "step": [step],

            "type": [transaction_type],

            "amount": [amount],

            "oldbalanceOrg": [
                old_balance_org
            ],

            "newbalanceOrig": [
                new_balance_orig
            ],

            "oldbalanceDest": [
                old_balance_dest
            ],

            "newbalanceDest": [
                new_balance_dest
            ]
        })

        try:

            # -------------------------------------------------
            # REAL MODEL PREDICTION
            # -------------------------------------------------

            prediction = int(
                model.predict(
                    input_data
                )[0]
            )

            # -------------------------------------------------
            # FRAUD PROBABILITY
            # -------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = float(
                    model.predict_proba(
                        input_data
                    )[0][1]
                )

            else:

                probability = float(
                    prediction
                )

            # -------------------------------------------------
            # RISK LEVEL
            # -------------------------------------------------

            if probability >= 0.70:

                risk_level = "HIGH"

            elif probability >= 0.40:

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "📊 Prediction Result"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            with c2:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            with c3:

                if prediction == 1:

                    st.metric(
                        "Prediction",
                        "🚨 FRAUD"
                    )

                else:

                    st.metric(
                        "Prediction",
                        "✅ LEGITIMATE"
                    )

            # -------------------------------------------------
            # PROGRESS BAR
            # -------------------------------------------------

            st.progress(
                probability,
                text=(
                    f"Fraud Risk Score: "
                    f"{probability * 100:.2f}%"
                )
            )

            # -------------------------------------------------
            # RISK MESSAGE
            # -------------------------------------------------

            if risk_level == "HIGH":

                st.markdown(
                    """
                    <div class="risk-high">

                    🚨 <b>HIGH RISK</b>

                    <br><br>

                    This transaction has a high predicted
                    probability of being fraudulent.

                    Further investigation is recommended.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif risk_level == "MEDIUM":

                st.markdown(
                    """
                    <div class="risk-medium">

                    ⚠️ <b>MEDIUM RISK</b>

                    <br><br>

                    This transaction shows suspicious
                    characteristics.

                    Additional verification is recommended.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="risk-low">

                    ✅ <b>LOW RISK</b>

                    <br><br>

                    The transaction appears legitimate
                    according to the trained model.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -------------------------------------------------
            # TRANSACTION SUMMARY
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "📋 Transaction Summary"
            )

            summary = pd.DataFrame({

                "Feature": [

                    "Transaction Step",

                    "Transaction Type",

                    "Amount",

                    "Sender Balance Before",

                    "Sender Balance After",

                    "Receiver Balance Before",

                    "Receiver Balance After"
                ],

                "Value": [

                    step,

                    transaction_type_name,

                    f"₹{amount:,.2f}",

                    f"₹{old_balance_org:,.2f}",

                    f"₹{new_balance_orig:,.2f}",

                    f"₹{old_balance_dest:,.2f}",

                    f"₹{new_balance_dest:,.2f}"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                "❌ Prediction could not be completed."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    str(e)
                )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Fraud Analytics Dashboard"
    )

    st.write(
        "Monitor transaction patterns and model predictions."
    )

    st.divider()

    total = len(data)

    fraud = int(
        (data["Prediction"] == 1).sum()
    )

    legitimate = total - fraud

    fraud_rate = (
        fraud / total * 100
        if total > 0
        else 0
    )

    fraud_amount = data.loc[
        data["Prediction"] == 1,
        "amount"
    ].sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transactions",
        f"{total:,}"
    )

    c2.metric(
        "Fraud Cases",
        f"{fraud:,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    c4.metric(
        "Fraud Amount",
        f"₹{fraud_amount:,.0f}"
    )

    st.divider()

    # -----------------------------------------------------
    # PIE CHART
    # -----------------------------------------------------

    fraud_count = pd.DataFrame({

        "Status": [
            "Legitimate",
            "Fraud"
        ],

        "Count": [
            legitimate,
            fraud
        ]
    })

    c1, c2 = st.columns(2)

    with c1:

        fig = px.pie(
            fraud_count,
            names="Status",
            values="Count",
            hole=0.5,
            title="Fraud vs Legitimate Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # TRANSACTION TYPE
    # -----------------------------------------------------

    with c2:

        type_data = (
            data
            .groupby("type")
            .size()
            .reset_index(
                name="Transactions"
            )
        )

        type_data["Transaction Type"] = (
            type_data["type"]
            .map(
                {
                    0: "CASH_IN",
                    1: "CASH_OUT",
                    2: "DEBIT",
                    3: "PAYMENT",
                    4: "TRANSFER"
                }
            )
        )

        fig = px.bar(
            type_data,
            x="Transaction Type",
            y="Transactions",
            title="Transactions by Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # FRAUD BY TYPE
    # -----------------------------------------------------

    st.subheader(
        "🚨 Fraud Cases by Transaction Type"
    )

    fraud_type = (
        data[
            data["Prediction"] == 1
        ]
        .groupby("type")
        .size()
        .reset_index(
            name="Fraud Cases"
        )
    )

    fraud_type["Transaction Type"] = (
        fraud_type["type"]
        .map(
            {
                0: "CASH_IN",
                1: "CASH_OUT",
                2: "DEBIT",
                3: "PAYMENT",
                4: "TRANSFER"
            }
        )
    )

    if not fraud_type.empty:

        fig = px.bar(
            fraud_type,
            x="Transaction Type",
            y="Fraud Cases",
            title="Detected Fraud by Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No fraud cases were predicted."
        )

    # -----------------------------------------------------
    # AMOUNT DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "💰 Transaction Amount Distribution"
    )

    chart_data = data.copy()

    fig = px.histogram(
        chart_data,
        x="amount",
        color="Status",
        nbins=30,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # FRAUD PROBABILITY
    # -----------------------------------------------------

    st.subheader(
        "🎯 Fraud Probability Distribution"
    )

    fig = px.histogram(
        data,
        x="Fraud Probability",
        nbins=30,
        title="Model Fraud Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TRANSACTIONS
# =========================================================

elif page == "📋 Transactions":

    st.title(
        "📋 Transaction Monitoring"
    )

    st.write(
        "Search and filter transactions predicted by the model."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        status_filter = st.selectbox(
            "Status",
            [
                "All",
                "Legitimate",
                "Fraud"
            ]
        )

    with c2:

        type_filter = st.selectbox(
            "Transaction Type",
            [
                "All",
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ]
        )

    with c3:

        search = st.text_input(
            "🔎 Search Transaction ID"
        )

    filtered_data = data.copy()

    # -----------------------------------------------------
    # STATUS FILTER
    # -----------------------------------------------------

    if status_filter != "All":

        filtered_data = filtered_data[
            filtered_data["Status"]
            == status_filter
        ]

    # -----------------------------------------------------
    # TYPE FILTER
    # -----------------------------------------------------

    if type_filter != "All":

        type_number = TRANSACTION_TYPES[
            type_filter
        ]

        filtered_data = filtered_data[
            filtered_data["type"]
            == type_number
        ]

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    if search:

        filtered_data = filtered_data[
            filtered_data[
                "Transaction ID"
            ]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # -----------------------------------------------------
    # DISPLAY DATA
    # -----------------------------------------------------

    display_data = filtered_data.copy()

    display_data["Transaction Type"] = (
        display_data["type"]
        .map(
            {
                0: "CASH_IN",
                1: "CASH_OUT",
                2: "DEBIT",
                3: "PAYMENT",
                4: "TRANSFER"
            }
        )
    )

    display_data = display_data[
        [
            "Transaction ID",
            "step",
            "Transaction Type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "Fraud Probability",
            "Status"
        ]
    ]

    st.caption(
        f"Showing {len(display_data):,} transaction(s)"
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    csv = display_data.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Filtered Transactions",
        data=csv,
        file_name="fraud_transactions.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ### AI-Powered Financial Fraud Detection System

        This application uses machine learning to analyze
        financial transactions and identify potentially
        fraudulent activities.

        The system provides:

        - 🔍 Transaction-level fraud prediction
        - 🎯 Fraud probability
        - 🚨 Risk classification
        - 📊 Interactive fraud analytics
        - 📋 Transaction monitoring
        - ⬇️ Downloadable results
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🛠 Technologies"
        )

        st.markdown(
            """
            - Python
            - Pandas
            - NumPy
            - Scikit-learn
            - Joblib
            - Plotly
            - Streamlit
            """
        )

    with col2:

        st.subheader(
            "🤖 Machine Learning"
        )

        st.markdown(
            """
            **Model:** Logistic Regression

            **Preprocessing:** StandardScaler +
            OneHotEncoder

            **Output:** Fraud prediction +
            probability

            **Features:** Transaction step,
            type, amount and account balances
            """
        )

    st.divider()

    if model is not None:

        st.success(
            "✅ Trained fraud detection pipeline "
            "is loaded and ready."
        )

    else:

        st.warning(
            "⚠️ Trained pipeline is not available."
        )

    st.caption(
        "Built as an AI-powered financial security project."
    )