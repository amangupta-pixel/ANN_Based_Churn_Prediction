End-to-End Customer Churn Prediction with ANN & Streamlit

Project Description

Objective: Developed a machine learning model to predict customer churn for a banking institution, enabling proactive customer retention strategies.

Data Preprocessing:
Engineered features for the model by applying one-hot encoding to the 'Geography' column and label encoding to the 'Gender' column to convert categorical data into a machine-readable format.

Utilized a standard scaler on the numerical features to normalize the data, ensuring that all features contribute equally to the model's performance.

Serialized the trained encoders and scaler objects using Pickle, ensuring the same data transformations could be applied consistently during model inference in the web application.

Model Development:
Built, trained, and evaluated an Artificial Neural Network (ANN) using the TensorFlow and Keras libraries.

The model was trained on preprocessed customer data to learn patterns associated with customer churn.

Saved the trained model architecture and weights into an HDF5 file (model.h5) for efficient loading and prediction.

Web Application & Deployment:
Developed an interactive and user-friendly web application using Streamlit to serve the churn prediction model.

The application features a clean UI with sliders, dropdowns, and number inputs for users to enter customer details.

Implemented a prediction pipeline within the app that takes user input, preprocesses it in real-time using the saved scaler and encoders, and feeds it to the loaded Keras model.

The application displays the churn probability and a clear, interpretable outcome ("Likely to Churn" or "Not Likely to Churn") based on the model's prediction.
