```python
import numpy as np
import joblib
import pandas as pd
import streamlit as st


# Load trained ML model
try:
    classifier = joblib.load("classifier.pkl")
except Exception as e:
    st.error("Unable to load classifier.pkl")
    st.error(f"Error: {e}")
    st.stop()


# Prediction function
def predict_note_authentication(variance, skewness, curtosis, entropy):
    prediction = classifier.predict(
        [[
            float(variance),
            float(skewness),
            float(curtosis),
            float(entropy)
        ]]
    )

    return prediction[0]


# Main Streamlit application
def main():

    st.set_page_config(
        page_title="Bank Authenticator",
        page_icon="🏦",
        layout="centered"
    )

    st.title("🏦 Bank Note Authentication")

    st.markdown(
        """
        <div style="
            background-color:tomato;
            padding:10px;
            border-radius:10px;
        ">
            <h2 style="color:white;text-align:center;">
                Streamlit Bank Authenticator ML App
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("Enter the features of the bank note:")

    # Input fields
    variance = st.number_input(
        "Variance",
        value=0.0,
        format="%.6f"
    )

    skewness = st.number_input(
        "Skewness",
        value=0.0,
        format="%.6f"
    )

    curtosis = st.number_input(
        "Curtosis",
        value=0.0,
        format="%.6f"
    )

    entropy = st.number_input(
        "Entropy",
        value=0.0,
        format="%.6f"
    )

    # Prediction button
    if st.button("🔍 Predict", use_container_width=True):

        try:
            result = predict_note_authentication(
                variance,
                skewness,
                curtosis,
                entropy
            )

            st.success(f"Prediction: {result}")

            if result == 0:
                st.info("The bank note is predicted to be **Genuine**.")
            elif result == 1:
                st.warning("The bank note is predicted to be **Forged**.")

        except Exception as e:
            st.error("Prediction failed.")
            st.error(f"Error: {e}")

    # About section
    if st.button("ℹ️ About", use_container_width=True):

        st.write("### About this project")
        st.write(
            "This application uses Machine Learning to authenticate "
            "bank notes based on four features:"
        )

        st.write("- Variance")
        st.write("- Skewness")
        st.write("- Curtosis")
        st.write("- Entropy")

        st.write("Built with Python, Scikit-learn and Streamlit.")


if __name__ == "__main__":
    main()
```

    
    
