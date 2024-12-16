# Algerian Forest Fire Prediction

This project predicts the **Fire Weather Index (FWI)**, a critical metric for assessing forest fire danger, using the **Algerian Forest Fire Dataset**. The workflow involves data cleaning, Exploratory Data Analysis (EDA), Feature Engineering (FE), and selecting the best regression model. The project is deployed using Flask to provide a user-friendly interface for input and prediction.

---

## Features
- **Dataset**: Algerian Forest Fire Dataset.
- **Data Preprocessing**: Includes data cleaning, EDA, and feature engineering.
- **Model Selection**: 
  - Ridge Regression
  - Lasso Regression
  - Elastic Net Regression
  
  Ridge Regression was selected as the final model based on performance metrics.
- **Models Created**:
  - `standardscalar.pkl`: For feature scaling.
  - `ridge.pkl`: Ridge regression model.
- **Deployment**: The project is deployed on a live server using Flask, with `index.html` and `home.html` to handle user input and display predictions.

---

## Dataset
The Algerian Forest Fire dataset contains data relevant to forest fire occurrences. Key preprocessing steps included:
- Cleaning missing and erroneous data.
- Analyzing correlations and distributions through EDA.
- Applying feature engineering techniques to enhance model performance.

---

## Workflow
1. **Data Cleaning**:
   - Removed missing or erroneous entries.
     - *This ensures that the dataset is free from inconsistencies that might affect the analysis.*
   - Formatted the data for analysis.
     - *This step prepares the data for easier integration with EDA and modeling tools.*

2. **Exploratory Data Analysis (EDA)**:
   - Visualized relationships between features.
     - *This helps identify patterns and correlations, aiding in feature selection.*
   - Identified important predictors for FWI.
     - *Focused on variables that significantly impact the Fire Weather Index.*

3. **Feature Engineering (FE)**:
   - Normalized and scaled features using `StandardScaler`.
     - *Standardization ensures consistent feature ranges, improving model convergence.*
   - Tested multiple regression techniques.
     - *Compared algorithms to identify the most suitable model for predictions.*

4. **Model Training**:
   - Compared Ridge, Lasso, and Elastic Net regression.
     - *Each technique offers different regularization approaches, balancing bias and variance.*
   - Ridge Regression was finalized for its superior performance.
     - *Chosen due to its ability to handle multicollinearity and provide accurate predictions.*

5. **Model Deployment**:
   - Flask-based web application.
     - *Flask simplifies the process of creating a lightweight server for predictions.*
   - HTML templates (`index.html` and `home.html`) for input and output.
     - *These templates create a user-friendly interface for interaction.*

---

## Files in the Repository
- `Model_Training.ipynb`: Jupyter notebook detailing data cleaning, EDA, and model training.
  - *Contains the step-by-step workflow for preprocessing and model selection.*
- `Regression_Project.ipynb`: Notebook containing additional insights and validation for the regression models.
  - *Focuses on model evaluation and further analysis.*
- `standardscalar.pkl`: Feature scaling model.
  - *Saves the trained `StandardScaler` for consistent preprocessing.*
- `ridge.pkl`: Trained Ridge regression model.
  - *Serialized model for making predictions on new data.*
- `index.html`: HTML template for user input.
  - *Provides fields for users to input feature values.*
- `home.html`: HTML template for displaying predictions.
  - *Displays the predicted FWI in a user-friendly format.*
- `app.py`: Flask application script for deployment.
  - *Connects the model to the web interface for real-time predictions.*

---

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RajxPatil/Algerian_forest_fire.git
   cd Algerian_forest_fire
   ```

2. **Install Dependencies**:
   Create a virtual environment and install the required libraries:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```
   - *Starts the Flask server on `http://127.0.0.1:5000/`.*

4. **Input Features**:
   - Use the web interface to input feature values.
     - *Enter the required data points for the model to process.*
   - Get predictions for the Fire Weather Index (FWI).
     - *Displays the predicted FWI based on the entered features.*

---

## Results
- The final Ridge regression model predicts FWI with high accuracy.
- Feature scaling significantly improved model performance.
  - *Ensures that all features contribute proportionally to the predictions.*
- The live server enables easy access for making predictions.
  - *Allows seamless interaction for users without requiring local installations.*

---

## Dependencies
- Python 3.x
  - *Programming language for data processing and application development.*
- Flask
  - *Web framework for deploying the model.*
- Pandas
  - *Data manipulation and analysis.*
- NumPy
  - *Numerical computations.*
- Scikit-learn
  - *Machine learning library for regression and preprocessing.*
- Jupyter Notebook
  - *Interactive environment for developing and documenting the workflow.*

---

## Future Enhancements
- Extend the model to include real-time weather data.
  - *Integrate APIs to fetch live weather conditions for more dynamic predictions.*
- Implement advanced visualization for better insights.
  - *Use libraries like Matplotlib and Seaborn to create informative plots.*
- Explore cloud deployment options for broader accessibility.
  - *Deploy on platforms like AWS, Azure, or Heroku to make the application globally available.*

---

## Credits
Developed by **Raj Patil** using the Algerian Forest Fire Dataset and deployed with Flask.
