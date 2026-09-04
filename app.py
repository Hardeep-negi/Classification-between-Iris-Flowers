from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

SPECIES_IMAGES = {
    "Iris-setosa": BASE_DIR / "Irissetosa1.png",
    "Iris-versicolor": BASE_DIR / "Iris-versicolor-1.png",
    "Iris-versicolour": BASE_DIR / "Iris-versicolor-1.png",
    "Iris-virginica": BASE_DIR / "Iris-virginica.png",
}

SPECIES_DESCRIPTIONS = {
    "Iris-setosa": (
        "Iris Setosa usually has smaller petals and wider sepals than the "
        "other Iris species, which makes it the easiest class to separate."
    ),
    "Iris-versicolor": (
        "Iris Versicolor sits between Setosa and Virginica in many feature "
        "values, with medium petal length and width."
    ),
    "Iris-versicolour": (
        "Iris Versicolor sits between Setosa and Virginica in many feature "
        "values, with medium petal length and width."
    ),
    "Iris-virginica": (
        "Iris Virginica generally has the largest petals in the dataset, "
        "especially petal length and petal width."
    ),
}

EXAMPLES = {
    "Setosa": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    },
    "Versicolor": {
        "sepal_length": 5.9,
        "sepal_width": 3.0,
        "petal_length": 4.2,
        "petal_width": 1.5,
    },
    "Virginica": {
        "sepal_length": 6.5,
        "sepal_width": 3.0,
        "petal_length": 5.8,
        "petal_width": 2.2,
    },
}


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, names=[*FEATURE_NAMES, "Class"])


def train_probability_model(df):
    model = SVC(probability=True, random_state=42)
    model.fit(
        df[FEATURE_NAMES].to_numpy(dtype=float),
        df["Class"].to_numpy(dtype=str),
    )
    return model


@st.cache_resource
def load_model():
    df = load_dataset()

    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as model_file:
                saved_model = pickle.load(model_file)
            if hasattr(saved_model, "predict_proba"):
                return saved_model, "Loaded saved SVM model."
            return (
                train_probability_model(df),
                "Trained probability-enabled SVM from iris.data.",
            )
        except Exception as exc:
            st.warning(
                "The saved model could not be loaded, so the app trained a fresh "
                f"SVM model from iris.data instead. Details: {exc}"
            )

    return train_probability_model(df), "Trained SVM model from iris.data."


@st.cache_data
def calculate_model_accuracy():
    df = load_dataset()
    x = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["Class"].to_numpy(dtype=str)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    accuracy_model = SVC(probability=True, random_state=42)
    accuracy_model.fit(x_train, y_train)
    predictions = accuracy_model.predict(x_test)
    return accuracy_score(y_test, predictions)


def format_species(label):
    return DISPLAY_NAMES.get(label, label.replace("Iris-", "Iris ").title())


def set_example(example_name):
    for key, value in EXAMPLES[example_name].items():
        st.session_state[key] = value


def measurement_chart(sample_values):
    return pd.DataFrame(
        {
            "Measurement": FEATURE_NAMES,
            "Value (cm)": sample_values,
        }
    )


def annotated_species_image(species, sample_values):
    image_path = SPECIES_IMAGES.get(species)
    if image_path is None or not image_path.exists():
        return None

    image = Image.open(image_path).convert("RGBA")
    max_width = 900
    if image.width > max_width:
        height = int(image.height * max_width / image.width)
        image = image.resize((max_width, height))

    canvas_padding = int(image.width * 0.18)
    canvas = Image.new(
        "RGBA",
        (image.width + canvas_padding * 2, image.height + canvas_padding),
        (255, 255, 255, 255),
    )
    image_x = canvas_padding
    image_y = 0
    canvas.alpha_composite(image, (image_x, image_y))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    colors = [
        (64, 196, 255, 235),
        (255, 193, 7, 235),
        (76, 175, 80, 235),
        (255, 112, 166, 235),
    ]

    width, height = image.size
    callouts = [
        {
            "feature": "Sepal length",
            "anchor": (0.43, 0.58),
            "label": (0.04, 0.78),
            "align": "left",
        },
        {
            "feature": "Sepal width",
            "anchor": (0.38, 0.65),
            "label": (0.04, 0.92),
            "align": "left",
        },
        {
            "feature": "Petal length",
            "anchor": (0.56, 0.35),
            "label": (0.78, 0.08),
            "align": "right",
        },
        {
            "feature": "Petal width",
            "anchor": (0.62, 0.42),
            "label": (0.80, 0.22),
            "align": "right",
        },
    ]

    for index, (callout, value) in enumerate(zip(callouts, sample_values)):
        color = colors[index]
        anchor_x = image_x + int(width * callout["anchor"][0])
        anchor_y = image_y + int(height * callout["anchor"][1])
        label_x = int(canvas.width * callout["label"][0])
        label_y = int(canvas.height * callout["label"][1])
        label = f'{callout["feature"]}: {value:.1f} cm'

        if callout["align"] == "right":
            text_width = draw.textlength(label, font=font)
            text_x = label_x - int(text_width)
        else:
            text_x = label_x

        elbow_x = int((anchor_x + label_x) / 2)
        elbow_y = label_y
        line_points = [(anchor_x, anchor_y), (elbow_x, elbow_y), (label_x, label_y)]

        outline_color = (255, 255, 255, 240)
        draw.line(line_points, fill=outline_color, width=9, joint="curve")
        draw.line(line_points, fill=color, width=4, joint="curve")
        draw.ellipse(
            (anchor_x - 8, anchor_y - 8, anchor_x + 8, anchor_y + 8),
            fill=color,
            outline=(255, 255, 255, 255),
            width=3,
        )
        draw.text(
            (text_x + 2, label_y - 15 + 2),
            label,
            fill=(255, 255, 255, 230),
            font=font,
        )
        draw.text((text_x, label_y - 15), label, fill=(20, 24, 33, 255), font=font)

    return canvas


st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon=":hibiscus:",
    layout="centered",
)

st.title("Iris Flower Classifier")
st.write(
    "Enter the flower measurements in centimeters and the SVM model will "
    "predict the Iris species."
)

model, model_status = load_model()
accuracy = calculate_model_accuracy()

with st.sidebar:
    st.header("Model")
    st.caption(model_status)
    st.metric("Test accuracy", f"{accuracy * 100:.1f}%")
    st.header("Typical ranges")
    st.caption("Sepal length: 4.3-7.9 cm")
    st.caption("Sepal width: 2.0-4.4 cm")
    st.caption("Petal length: 1.0-6.9 cm")
    st.caption("Petal width: 0.1-2.5 cm")

st.subheader("Try an example")
example_cols = st.columns(3)
for column, example_name in zip(example_cols, EXAMPLES):
    with column:
        st.button(
            example_name,
            on_click=set_example,
            args=(example_name,),
            use_container_width=True,
        )

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1,
        key="sepal_length",
    )
    petal_length = st.number_input(
        "Petal length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1,
        key="petal_length",
    )

with col2:
    sepal_width = st.number_input(
        "Sepal width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        key="sepal_width",
    )
    petal_width = st.number_input(
        "Petal width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
        key="petal_width",
    )

sample_values = [sepal_length, sepal_width, petal_length, petal_width]
sample = np.array([sample_values])

st.subheader("Input comparison")
st.bar_chart(measurement_chart(sample_values), x="Measurement", y="Value (cm)")

if st.button("Predict species", type="primary"):
    prediction = model.predict(sample)[0]
    st.subheader(format_species(prediction))
    st.success(f"Predicted class: {prediction}")
    st.write(SPECIES_DESCRIPTIONS.get(prediction, "This is one of the Iris species."))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        probability_table = pd.DataFrame(
            {
                "Species": [format_species(label) for label in model.classes_],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        top_probability = probability_table.iloc[0]["Probability"]
        st.metric("Prediction confidence", f"{top_probability * 100:.1f}%")
        st.bar_chart(probability_table, x="Species", y="Probability")
        st.dataframe(
            probability_table.assign(
                Probability=lambda df: (df["Probability"] * 100).round(2)
            ),
            hide_index=True,
            use_container_width=True,
        )

    annotated_image = annotated_species_image(prediction, sample_values)
    if annotated_image is not None:
        st.image(
            annotated_image,
            caption=f"{format_species(prediction)} with your input measurements",
            use_container_width=True,
        )

st.divider()
st.caption(
    "This app uses an SVM classifier trained on the classic Iris flower dataset."
)
