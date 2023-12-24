import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import sys
from randomForest import RandomForest
from randomForest import DecisionTree
import base64
from PIL import Image
import pandas as pd
import pickle as cPickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings

# Load background image
def add_bg_from_local(image_file):
    with open("C:/Users/iq22b/OneDrive/Documents/Randoom_Forest/image/pic3.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('blue_bg.png')


def main():

    selected = option_menu( 
        menu_title=None,
        options=["About","Prediction","ALL"],
        icons=["None","None","None"],
        menu_icon="None",
        default_index=0,
        orientation="horizontal",
        styles = {
            
        }
    )
    
    if selected == "About":
        st.title("About The Project")

        image = Image.open("C:/Users/iq22b/OneDrive/Documents/Randoom_Forest/image/pic2.jpg")
        st.image(image, caption='random picture')

        '''
        The cardiovascular system in the human body consists of the heart and blood vessels. 
        The current consumerism and technology-driven culture, which is associated with longer work hours, 
        longer commutes, and less leisure time for recreational activities, may explain the significant and 
        steady increase in CVD rates over the last few decades. 
        
        According to WHO in 2019, an estimated 17.9 million people (representing 32% of all global) died from CVDs. 
        Physical inactivity, a high-calorie diet, saturated fats, and sugars are specifically linked to the development 
        of atherosclerosis and other metabolic disturbances such as metabolic syndrome, diabetes mellitus, 
        and hypertension, all of which are common in people with CVDs (Curry et al., 2018). 
        Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, 
        poor diet and obesity, physical inactivity, and excessive alcohol consumption. Early detection as possible is 
        crucial in order to arrange treatment such as counselling, self-care support and medication (Riegel et al., 2017).
        
        
        '''
        
        st.title("Objective")
        '''
        There are three objectives to this research project. These objectives must be stated to know whether the guidelines have been followed and the objectives have been successfully achieved. The objectives are as follows: 
        1.	To investigate random forest algorithms for predicting the probability getting heart diseases.
        2.	To develop a prototype that can predict a patient's potential for heart disease.
        3.	To evaluate the accuracy of the prediction of heart diseases

        '''
               
        image = Image.open("C:/Users/iq22b/OneDrive/Documents/Randoom_Forest/image/table.jpg")
        st.image(image, caption='Definition of each feature')
        
        
        st.title("Accuracy")        
        image = Image.open("C:/Users/iq22b/OneDrive/Documents/Randoom_Forest/image/fc.png")
        st.image(image, caption='Classification Evaluation Metrics')
        
        image = Image.open("C:/Users/iq22b/OneDrive/Documents/Randoom_Forest/image/fcc.png")
        st.image(image, caption='Confusion Metrics')
        
        image = Image.open("C:/Users/iq22b/OneDrive/Documents/Randoom_Forest/image/9.png")
        st.image(image, caption='ROC')
        




    if selected == "Prediction":
        tab1, tab2, tab3 = st.tabs(["RF", "KNN", "SVC"])
        with tab1:
        
            # Set the title and description
            st.title('Heart Disease Prediction using RF')
            st.write('Enter the following information to predict heart disease:')
            
            # Load the trained model
            model_path = "C:/Users/iq22b/OneDrive/Desktop/sem6/csp650/model/TrainModel3"
            model = joblib.load(model_path)

            # Define the feature labels
            feature_labels = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]

            # Divide the feature labels into three sections: left, center, right
            left_labels = feature_labels[:4]
            center_labels = feature_labels[4:9]
            right_labels = feature_labels[9:]
            
            # Create three columns for input fields: left, center, right
            col1, col2, col3 = st.columns(3)

            # Create input fields for user to enter data in the three columns
            user_inputs_left = []
            for label in left_labels:
                value = col1.number_input(label, step=0.00)
                user_inputs_left.append(value)

            user_inputs_center = []
            for label in center_labels:
                value = col2.number_input(label, step=0.00)
                user_inputs_center.append(value)

            user_inputs_right = []
            for label in right_labels:
                value = col3.number_input(label, step=0.00)
                user_inputs_right.append(value)

            user_inputs = user_inputs_left + user_inputs_center + user_inputs_right

            def heart_disease_prediction(features):
                # Preprocess the input features
                features = np.array(features).reshape(1, -1)

                # Make prediction
                prediction = model.predict(features)

                # Convert prediction to readable output
                if prediction[0] == 0:
                    result = st.success('No heart disease')
                else:
                    result = st.error('Heart Disease')

                return result


            # Make prediction when button is clicked
            if st.button('Predict'):
                prediction_result = heart_disease_prediction(user_inputs)
                return ('Prediction:', prediction_result)
        
        with tab2:
            # Load the KNN model
            with open("C:/Users/iq22b/OneDrive/Desktop/sem6/csp650/model/ModelKNN", 'rb') as f:
                knn_scores = cPickle.load(f)

            # Load the heart disease dataset
            dataset = pd.read_csv("C:/Users/iq22b/OneDrive/Desktop/jupyter/heart.csv")
            y = dataset['target']
            X = dataset.drop(['target'], axis=1)

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

            # Create a Streamlit app
            st.title('Heart Disease Prediction Using KNN')

            # Create input fields for user to input 13 features
            st.write("Enter the values for the 13 features:")
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            user_input = []
            for feature in feature_names:
                user_input.append(st.number_input(f"Enter {feature}:"))

            # Create a predict button
            if st.button('Predict KNN'):
                # Prepare user input as a DataFrame
                user_input_df = pd.DataFrame([user_input], columns=feature_names)

                # Load the KNN model
                k = knn_scores.index(max(knn_scores)) + 1  # Choose k with highest accuracy from the saved scores
                knn_classifier = KNeighborsClassifier(n_neighbors=k)
                
                # Make a prediction
                knn_classifier.fit(X_train, y_train)
                prediction = knn_classifier.predict(user_input_df)

                # Display prediction
                if prediction[0] == 1:
                    st.error('Prediction: Heart Disease')
                else:
                    st.success('Prediction: No Heart Disease')


        with tab3:
            # Load the trained SVC models
            with open("C:/Users/iq22b/OneDrive/Desktop/sem6/csp650/model/ModelSVC.pickle", 'rb') as f:
                svc_models = cPickle.load(f)

            # Streamlit GUI
            st.title('Heart Disease Prediction Using SVC')

            # Input form for user to enter features
            feature_names = [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ]

            user_input = []

            for feature in feature_names:
                user_input.append(st.number_input(f"Enter {feature}:", key=feature))

            # Choose a kernel for prediction
            selected_kernel = st.selectbox("Select Kernel for Prediction:", ["linear", "poly", "rbf", "sigmoid"])

            # Make prediction
            if st.button('Predict SVC'):
                user_input = np.array(user_input).reshape(1, -1)
                svc_model = svc_models[selected_kernel]
                prediction = svc_model.predict(user_input)

                if prediction[0] == 1:
                    st.error("Prediction: Heart Disease")
                else:
                    st.success("Prediction: No Heart Disease")
                    


    if selected == "ALL":
        # Set the title and description
        st.title('HD Prediction using RF,KNN & SVC')
        st.write('Enter the following information to predict heart disease:')
        
        # Load the trained model
        model_path = "C:/Users/iq22b/OneDrive/Desktop/sem6/csp650/model/TrainModel3"
        model = joblib.load(model_path)

        # Load the trained model knn
        model_pathknn = "C:/Users/iq22b/OneDrive/Desktop/sem6/csp650/model/ModelKNN"
        knn_scores = joblib.load(model_pathknn)
        
        # Load the trained model svc
        model_pathsvc = "C:/Users/iq22b/OneDrive/Desktop/sem6/csp650/model/ModelSVC.pickle"
        svc_models = joblib.load(model_pathsvc)
        
        # Load the heart disease dataset
        dataset = pd.read_csv("C:/Users/iq22b/OneDrive/Desktop/jupyter/heart.csv")
        y = dataset['target']
        X = dataset.drop(['target'], axis=1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        
        # Define the feature labels
        feature_labels = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        # Divide the feature labels into three sections: left, center, right
        left_labels = feature_labels[:4]
        center_labels = feature_labels[4:9]
        right_labels = feature_labels[9:]
        
        # Create three columns for input fields: left, center, right
        col1, col2, col3 = st.columns(3)

        # Create input fields for user to enter data in the three columns
        user_inputs_left = []
        for label in left_labels:
            value = col1.number_input(label, step=0.00)
            user_inputs_left.append(value)

        user_inputs_center = []
        for label in center_labels:
            value = col2.number_input(label, step=0.00)
            user_inputs_center.append(value)

        user_inputs_right = []
        for label in right_labels:
            value = col3.number_input(label, step=0.00)
            user_inputs_right.append(value)

        user_inputs = user_inputs_left + user_inputs_center + user_inputs_right
        
        # Choose a kernel for prediction
        selected_kernel = st.selectbox("Select Kernel for Prediction:", ["linear", "poly", "rbf", "sigmoid"])
        

        def heart_disease_prediction(features):
            # Preprocess the input features
            features = np.array(features).reshape(1, -1)

            # Make prediction
            prediction = model.predict(features)

            # Convert prediction to readable output
            if prediction[0] == 0:
                result = "No heart disease"
            else:
                result = "Heart disease"

            return result
            

        def knn_prediction(features):
            # Prepare user input as a DataFrame
            user_input_df = pd.DataFrame([user_inputs], columns=feature_labels)

            # Load the KNN model
            k = knn_scores.index(max(knn_scores)) + 1  # Choose k with highest accuracy from the saved scores
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            
            # Make a prediction
            knn_classifier.fit(X_train, y_train)
            prediction = knn_classifier.predict(user_input_df)

            # Convert prediction to readable output
            if prediction[0] == 0:
                result = "No heart disease"
            else:
                result = "Heart disease"

            return result
            
            
        def svc_prediction(features):
            user_input = np.array(user_inputs).reshape(1, -1)
            svc_model = svc_models[selected_kernel]
            prediction = svc_model.predict(user_input)

            # Convert prediction to readable output
            if prediction[0] == 0:
                result = "No heart disease"
            else:
                result = "Heart disease"

            return result




        # Make prediction when button is clicked
        if st.button('Predict'):
            prediction_result = heart_disease_prediction(user_inputs)
            st.write('Prediction RF:', prediction_result)
            
            prediction_resultknn = knn_prediction(user_inputs)
            st.write('Prediction KNN:', prediction_resultknn)
            
            prediction_resultsvc = svc_prediction(user_inputs)
            st.write('Prediction SVC:', prediction_resultsvc)

if __name__ == '__main__':
    main()