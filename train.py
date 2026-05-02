import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils import load_data, preprocess_data
import matplotlib.pyplot as plt

# Load data
df = load_data('data/housing.csv')

# Preprocess
X, y, encoder = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump((model, encoder), f)

print("Model trained and saved!")

import matplotlib.pyplot as plt

def plot_data(df):
    plt.scatter(df['area'], df['price'])
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Area vs Price")
    plt.show()

#train the model, run with `python train.py` and then run the app with `python -m streamlit run app.py` in the terminal to see the visualization.