import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd  # Importing pandas to load your dataset

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('placement.csv')

# Assuming 'cgpa' and 'placement_exam_marks' are columns in your dataset
x = df['cgpa']
y = df['placement_exam_marks']

# Reshape x and y
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create and train the model
lm = LinearRegression()
lm.fit(x_train, y_train)

# Save the model
joblib.dump(lm, 'models/linear_regression_model.joblib')

# Save other necessary variables
variables_to_save = {
    'x_train': x_train,
    'x_test': x_test,
    'y_train': y_train,
    'y_test': y_test,
    'slope': lm.coef_,
    'intercept': lm.intercept_,
}

joblib.dump(variables_to_save, 'models/model_variables.joblib')
