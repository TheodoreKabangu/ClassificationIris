#Imports
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import streamlit as st
import pickle

# Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
#create the model
model = load_model()

# Function to make predictions
def predict():
    # Set the title of the Streamlit app
    st.title("Classification des fleurs Iris")
    st.write("Cette application prédit la classe d'une fleur Iris en fonction de ses caractéristiques. Elle utilise un modèle de classification entraîné sur le jeu de données Iris.")
    st.sidebar.header("Dimensions de la fleur Iris")
    sepal_length= st.sidebar.slider("Longueur du sépale (cm)", 4.0, 8.0, 4.0)
    sepal_width= st.sidebar.slider("Largeur du sépale (cm)", 2.0, 5.0, 2.0)
    petal_length= st.sidebar.slider("Longueur du pétale (cm)", 1.0, 7.0, 1.0)
    petal_width= st.sidebar.slider("Largeur du pétale (cm)", 0.1, 2.5, 0.1)
    bouton_predict = st.sidebar.button("Prédire")
    classe = "Inconnu"
    data = pd.DataFrame({})
    if bouton_predict:
        # Predictions
        X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        X= X.astype(float)
        
        data = pd.DataFrame({
            "Longueur sépale": X[:, 0],
            "Largeur sépale": X[:, 1],
            'Longueur pétale': X[:, 2],
            "Largeur pétale": X[:, 3]
        })
        
        prediction = model.predict(X)    
        if prediction == 0:
            classe = "Setosa"
        elif prediction == 1:
            classe = "Versicolor"
        elif prediction == 2:
            classe = "Virginica"

    st.write("### Caractéristiques de la fleur Iris")
    st.write(data)
    st.write("### Prédiction")  
    st.write(f"La fleur Iris est classée comme : {classe}")


def visualiser():
    st.title("Visualisation des données Iris")
    st.write("Cette section permet de visualiser les données du jeu de données Iris.")
    # Load the dataset
    X = load_iris().data
    y = load_iris().target
    iris_data = pd.DataFrame(X, columns=load_iris().feature_names)
    iris_data['species'] = pd.Categorical.from_codes(y, load_iris().target_names)
    #st.write(iris_data.head())
    st.bar_chart(iris_data['species'].value_counts(ascending=False))

viz= visualiser()