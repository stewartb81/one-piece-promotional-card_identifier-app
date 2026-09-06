import streamlit as st
import tensorflow as tf
import json

from PIL import ImageOps

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class EfficientNetPreprocess(layers.Layer):

    def call(self, inputs):
        return preprocess_input(inputs)


# Load trained model
model = tf.keras.models.load_model(
    "efficientnet_final.keras",
    custom_objects={
        "EfficientNetPreprocess": EfficientNetPreprocess
    }
)


# Load class names
with open("class_names.json", "r") as file:
    class_names = json.load(file)


# Load card information
with open("card_information.json", "r") as file:
    card_information = json.load(file)


# Page setup
st.title("One Piece Promotional Card Identifier")

st.write(
    "AI-powered card recognition. Upload a card image and the model will attempt to identify it."
)


# Upload image
uploaded_file = st.file_uploader(
    "Choose a card image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Load uploaded image
    image = tf.keras.utils.load_img(
        uploaded_file
    )


    # Correct phone camera orientation
    display_image = ImageOps.exif_transpose(
        image
    )


    # Create resized copy for model input
    model_image = display_image.resize(
        (180, 180)
    )


    image_array = tf.keras.utils.img_to_array(
        model_image
    )


    image_array = tf.expand_dims(
        image_array,
        0
    )


    # Make prediction
    with st.spinner("Analysing image..."):

        prediction = model.predict(
            image_array
        )


    # Find top 3 predictions
    top_3 = tf.argsort(
        prediction[0],
        direction="DESCENDING"
    )[:3].numpy()


    # Best prediction
    predicted_class = tf.argmax(
        prediction[0]
    ).numpy()


    predicted_card = class_names[predicted_class]


    confidence = float(
        tf.reduce_max(prediction[0]) * 100
    )


    # Create columns
    col1, col2 = st.columns(2)


    # Left column - uploaded image and prediction details
    with col1:

        st.image(
            display_image,
            caption="Uploaded image",
            use_container_width=True
        )


        st.write(
            f"Confidence: {confidence:.2f}%"
        )


        st.progress(
            confidence / 100
        )


        if confidence >= 80:

            st.write(
                "Prediction:",
                predicted_card
            )


        elif confidence >= 50:

            st.write(
                "Possible match:",
                predicted_card
            )


    # Right column - reference image and confidence message
    with col2:

        if confidence >= 50:

            reference_image = f"card_images/{predicted_card}.jpg"


            st.image(
                reference_image,
                caption="Reference image",
                use_container_width=True
            )


        if confidence >= 80:

            st.success(
                "🟢 High confidence - This is very likely your card."
            )


        elif confidence >= 50:

            st.warning(
                "🟡 Medium confidence - This may be your card, "
                "but the model is not completely certain."
            )


        else:

            st.error(
                "🔴 Unable to identify - The model could not "
                "confidently identify this image as a known "
                "One Piece card. Please upload a clearer card image."
            )


    # Card origin information
    if confidence >= 80:

        st.write(
            "### Where did this card come from?"
        )


        st.write(
            "**Soruce:**",
            card_information[predicted_card]["event"]
        )


        st.write(
            "**Distribution:**",
            card_information[predicted_card]["description"]
        )


    # Top predictions
    if confidence >= 50:

        st.write(
            "### Top 3 Predictions"
        )


        for i in top_3:

            st.write(
                f"{class_names[i]} "
                f"({prediction[0][i] * 100:.2f}%)"
            )


# Project information and ethical considerations
with st.expander(
    "About this project, limitations and ethical considerations"
):

    st.caption(
        """
        This application is a prototype developed as part of a university
        project. It uses machine learning to assist with identifying
        One Piece promotional trading cards.

        **Model limitations**

        The model has been trained on a limited dataset of 20 promotional cards.
        Predictions are not guaranteed to be correct and should be treated as
        a guide rather than a definitive identification.

        Performance may be reduced when images are unclear, low quality,
        taken from unusual angles or when the uploaded card is outside the
        trained dataset.

        **Data privacy**

        Uploaded images are processed only for identification purposes and
        are not permanently stored by this prototype.
        """
    )