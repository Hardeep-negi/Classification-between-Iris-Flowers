from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import SVC


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "SVM.pickle"
DATA_PATH = BASE_DIR / "iris.data"

FEATURE_NAMES = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width",
]

DISPLAY_NAMES = {
    "Iris-setosa": "Iris Setosa",
    "Iris-versicolor": "Iris Versicolor",
    "Iris-versicolour": "Iris Versicolor",
    "Iris-virginica": "Iris Virginica",
}


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as model_file:
                return pickle.load(model_file), "Loaded saved SVM model."
        except Exception as exc:
            st.warning(
                "The saved model could not be loaded, so the app trained a fresh "
                f"SVM model from iris.data instead. Details: {exc}"
            )

    df = pd.read_csv(
        DATA_PATH,
        names=[*FEATURE_NAMES, "Class"],
    )
    x = df[FEATURE_NAMES].values
    y = df["Class"].values

    model = SVC(probability=True, random_state=42)
    model.fit(x, y)
    return model, "Trained SVM model from iris.data."


def format_species(label):
    return DISPLAY_NAMES.get(label, label.replace("Iris-", "Iris ").title())


st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

st.title("Iris Flower Classifier")
st.write(
    "Enter the flower measurements in centimeters and the SVM model will "
    "predict the Iris species."
)

model, model_status = load_model()

with st.sidebar:
    st.header("Model")
    st.caption(model_status)
    st.header("Typical ranges")
    st.caption("Sepal length: 4.3-7.9 cm")
    st.caption("Sepal width: 2.0-4.4 cm")
    st.caption("Petal length: 1.0-6.9 cm")
    st.caption("Petal width: 0.1-2.5 cm")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1,
    )
    petal_length = st.number_input(
        "Petal length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1,
    )

with col2:
    sepal_width = st.number_input(
        "Sepal width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
    )
    petal_width = st.number_input(
        "Petal width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
    )

sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict species", type="primary"):
    prediction = model.predict(sample)[0]
    st.subheader(format_species(prediction))
    st.success(f"Predicted class: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        probability_table = pd.DataFrame(
            {
                "Species": [format_species(label) for label in model.classes_],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        st.bar_chart(probability_table, x="Species", y="Probability")
        st.dataframe(
            probability_table.assign(
                Probability=lambda df: (df["Probability"] * 100).round(2)
            ),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.caption(
    "This app uses an SVM classifier trained on the classic Iris flower dataset."
)
