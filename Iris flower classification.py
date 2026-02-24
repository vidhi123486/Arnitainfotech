import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Load Dataset & Train Model
# ------------------------------
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("🌼 Iris Flower Classification App")
st.write("A simple ML model deployed using **Streamlit**.")

# Sidebar sliders for feature input
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 6.9, 4.35)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)[0]
prediction_proba = model.predict_proba(input_data_scaled)
species = iris.target_names[prediction]

# ------------------------------
# OUTPUT BOX
# ------------------------------
st.subheader("🔍 Prediction Result")
st.success(f"🌸 **Predicted Species:** {species}")

# Probability box
st.subheader("📊 Prediction Probability")
st.info(
    {
        iris.target_names[i]: float(prediction_proba[0][i])
        for i in range(3)
    }
)

# Option to show dataset
if st.checkbox("Show Raw Iris Data"):
    st.write(iris.data)
