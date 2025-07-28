import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load and prepare data
def load_data():
    diabetes = pd.read_csv('diabetes.csv')
    X = diabetes.drop('Outcome', axis=1)
    y = diabetes['Outcome']
    return train_test_split(X, y, test_size=0.4, random_state=80)

# Train all models
def train_models(x_train, x_test, y_train, y_test):
    # KNN
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(x_train, y_train)
    
    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(x_train, y_train)

    # MLP
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    mlp = MLPClassifier(random_state=0, max_iter=500)
    mlp.fit(x_train_scaled, y_train)

    return {
        "knn": knn,
        "tree": tree,
        "mlp": mlp,
        "scaler": scaler
    }
