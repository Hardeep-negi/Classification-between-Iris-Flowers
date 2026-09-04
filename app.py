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


def get_font(size, bold=True):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "FreeSansBold.ttf" if bold else "FreeSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


SPECIES_CALLOUTS = {
    "Iris-virginica": [
        {"feature": "Sepal Length", "anchor_rel": (0.28, 0.64), "side": "left", "y_target_rel": 0.40},
        {"feature": "Sepal Width", "anchor_rel": (0.22, 0.72), "side": "left", "y_target_rel": 0.72},
        {"feature": "Petal Length", "anchor_rel": (0.64, 0.28), "side": "right", "y_target_rel": 0.25},
        {"feature": "Petal Width", "anchor_rel": (0.62, 0.44), "side": "right", "y_target_rel": 0.58},
    ],
    "Iris-versicolor": [
        {"feature": "Sepal Length", "anchor_rel": (0.48, 0.58), "side": "left", "y_target_rel": 0.40},
        {"feature": "Sepal Width", "anchor_rel": (0.40, 0.68), "side": "left", "y_target_rel": 0.72},
        {"feature": "Petal Length", "anchor_rel": (0.62, 0.28), "side": "right", "y_target_rel": 0.25},
        {"feature": "Petal Width", "anchor_rel": (0.60, 0.45), "side": "right", "y_target_rel": 0.58},
    ],
    "Iris-versicolour": [
        {"feature": "Sepal Length", "anchor_rel": (0.48, 0.58), "side": "left", "y_target_rel": 0.40},
        {"feature": "Sepal Width", "anchor_rel": (0.40, 0.68), "side": "left", "y_target_rel": 0.72},
        {"feature": "Petal Length", "anchor_rel": (0.62, 0.28), "side": "right", "y_target_rel": 0.25},
        {"feature": "Petal Width", "anchor_rel": (0.60, 0.45), "side": "right", "y_target_rel": 0.58},
    ],
    "Iris-setosa": [
        {"feature": "Sepal Length", "anchor_rel": (0.25, 0.62), "side": "left", "y_target_rel": 0.40},
        {"feature": "Sepal Width", "anchor_rel": (0.24, 0.72), "side": "left", "y_target_rel": 0.72},
        {"feature": "Petal Length", "anchor_rel": (0.62, 0.28), "side": "right", "y_target_rel": 0.25},
        {"feature": "Petal Width", "anchor_rel": (0.60, 0.45), "side": "right", "y_target_rel": 0.58},
    ],
}


def annotated_species_image(species, sample_values):
    image_path = SPECIES_IMAGES.get(species)
    if image_path is None or not image_path.exists():
        return None

    orig_image = Image.open(image_path).convert("RGBA")
    orig_w, orig_h = orig_image.size

    # Add canvas padding so callout cards never crowd the flower or clip screen edges
    pad_left = 320
    pad_right = 320
    pad_top = 80
    pad_bottom = 80

    canvas_w = orig_w + pad_left + pad_right
    canvas_h = orig_h + pad_top + pad_bottom

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(orig_image, (pad_left, pad_top), orig_image)

    draw = ImageDraw.Draw(canvas)

    # Scaled typography for high readability
    title_font = get_font(28, bold=True)
    val_font = get_font(34, bold=True)

    # High-contrast palette visible in BOTH light and dark modes
    accent_color = (245, 158, 11, 255)       # Vibrant Amber - warm, high-contrast, botanical
    halo_color = (15, 23, 42, 230)          # Deep Slate shadow for crisp contrast on light backgrounds
    badge_bg = (15, 23, 42, 235)            # Dark slate card background (visible on both light & dark)
    badge_border = (245, 158, 11, 255)      # Amber border
    text_label_color = (203, 213, 225, 255) # Light slate-gray
    text_val_color = (255, 255, 255, 255)   # Pure crisp white

    callouts = SPECIES_CALLOUTS.get(
        species,
        [
            {"feature": "Sepal Length", "anchor_rel": (0.28, 0.64), "side": "left", "y_target_rel": 0.40},
            {"feature": "Sepal Width", "anchor_rel": (0.22, 0.72), "side": "left", "y_target_rel": 0.72},
            {"feature": "Petal Length", "anchor_rel": (0.64, 0.28), "side": "right", "y_target_rel": 0.25},
            {"feature": "Petal Width", "anchor_rel": (0.62, 0.44), "side": "right", "y_target_rel": 0.58},
        ],
    )

    for spec, val in zip(callouts, sample_values):
        ax = pad_left + int(orig_w * spec["anchor_rel"][0])
        ay = pad_top + int(orig_h * spec["anchor_rel"][1])

        feature_text = spec["feature"]
        val_text = f"{val:.1f} cm"

        # Calculate text bounding boxes
        f_bbox = draw.textbbox((0, 0), feature_text, font=title_font)
        v_bbox = draw.textbbox((0, 0), val_text, font=val_font)

        f_w = f_bbox[2] - f_bbox[0]
        f_h = f_bbox[3] - f_bbox[1]
        v_w = v_bbox[2] - v_bbox[0]
        v_h = v_bbox[3] - v_bbox[1]

        card_w = max(f_w, v_w) + 40
        card_h = f_h + v_h + 36

        card_y = pad_top + int(orig_h * spec["y_target_rel"]) - card_h // 2
        card_y = max(20, min(canvas_h - card_h - 20, card_y))

        if spec["side"] == "left":
            card_x = 30
            conn_x = card_x + card_w
            conn_y = card_y + card_h // 2
            elbow_x = conn_x + 50
        else:
            card_x = canvas_w - card_w - 30
            conn_x = card_x
            conn_y = card_y + card_h // 2
            elbow_x = conn_x - 50

        elbow_y = ay
        line_points = [(ax, ay), (elbow_x, elbow_y), (conn_x, conn_y)]

        # 1. Dual-stroke line: outer dark halo + vibrant accent core
        draw.line(line_points, fill=halo_color, width=9, joint="curve")
        draw.line(line_points, fill=accent_color, width=5, joint="curve")

        # 2. Precision anchor pin on flower
        draw.ellipse((ax - 14, ay - 14, ax + 14, ay + 14), fill=halo_color)
        draw.ellipse((ax - 10, ay - 10, ax + 10, ay + 10), fill=accent_color)
        draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=(255, 255, 255, 255))

        # 3. Callout badge card with rounded corners
        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_w, card_y + card_h),
            radius=14,
            fill=badge_bg,
            outline=badge_border,
            width=2,
        )

        # 4. Clear, large text inside badge
        draw.text(
            (card_x + 20, card_y + 14),
            feature_text,
            fill=text_label_color,
            font=title_font,
        )
        draw.text(
            (card_x + 20, card_y + 14 + f_h + 10),
            val_text,
            fill=text_val_color,
            font=val_font,
        )

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
