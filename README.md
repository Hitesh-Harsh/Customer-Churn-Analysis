# Customer Churn Prediction

### Background and Context
In industries like banking, understanding customer churn—customers leaving for competitors—is crucial for improving service and retention strategies. This project aims to predict customer churn using machine learning techniques, specifically a neural network-based classifier built with TensorFlow.

### Objective
The goal of this project is to develop a model that can predict whether a bank customer will leave within the next 6 months based on various customer attributes.

### Data Description
The dataset used in this project is sourced from Kaggle and contains 10,000 samples with the following features:
- **CustomerId**: Unique identifier for each customer
- **Surname**: Customer's last name
- **CreditScore**: Customer's credit history score
- **Geography**: Customer's location
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years the customer has been with the bank
- **NumOfProducts**: Number of bank products owned by the customer
- **Balance**: Customer's account balance
- **HasCrCard**: Whether the customer has a credit card (Yes/No)
- **EstimatedSalary**: Estimated salary of the customer
- **IsActiveMember**: Whether the customer is an active member of the bank (Yes/No)
- **Exited**: Whether the customer left the bank within 6 months (0 = No, 1 = Yes)

### How to Use
To use the churn prediction application:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository

2. **Install Dependencies**

     To install the necessary Python libraries, run the following command in your terminal:

    ```bash
            pip install -r requirements.txt

3. **Run the Streamlit App**

    To run the Streamlit application locally, use the following command in your terminal:

    ```bash
            streamlit run app.py

4. **Use the interface to input customer details such as geography, gender, age, credit score, account balance, etc**

    Click on the "Predict" button to see the churn prediction for the customer based on the input data.

5. **Technologies Used**
    Python: Programming language used for the application.
    TensorFlow: Machine learning framework used to build and train the neural network model.
    Streamlit: Open-source app framework used to create interactive web applications for machine learning and data science.
6. **Files Included**
    model.h5: Trained neural network model file.
    label_encoder_gender.pkl: Pickled object for gender label encoding.
    onehot_encoder_geo.pkl: Pickled object for geography one-hot encoding.
    scaler.pkl: Pickled object for data scaling.

**Future Improvements**
Include more features or refine existing ones to improve prediction accuracy.
Enhance the UI for better user interaction and experience.
Deploy the application to a web server for wider accessibility.