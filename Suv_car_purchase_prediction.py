import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


suv_car_df = pd.read_csv('suv_data.csv')

x = suv_car_df.iloc[:, [2,3]]
y = suv_car_df.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()

model.fit(x_train, y_train)
train_y_pred = model.predict(x_train)
test_y_pred = model.predict(x_test)




# Website
def app():
    # Image and Title
    image_path = 'car_img.jpg'
    st.image(image_path, width=350)
    
    st.title("SUV Car Purchasing Prediction")
    st.write("This app predicts whether a customer has purchased an SUV car based on their age and salary.")

    # Sliders for user input
    age = st.slider("Select Age: ", min_value=18, max_value=100, step=1, value=30)
    salary = st.slider("Select Salary: ", min_value=10000, max_value=200000, step=1000, value=50000)

    # Making prediction
    x_new = [[age, salary]]
    x_new_scaled = scaler.transform(x_new)
    y_new = model.predict(x_new_scaled)

    if y_new == 1:
        st.success("✅ This person **has bought** an SUV car.")
    else:
        st.error("❌ This person **has not bought** the Car.")


# Run App
if __name__ == "__main__":
    app()

