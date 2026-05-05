import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
import glob

# Try to find the actual CSV file in the directory
csv_files = glob.glob("*.csv")
if not csv_files:
    print("No CSV files found.")
else:
    df = pd.read_csv(csv_files[0])
    print(f"Loaded {csv_files[0]} shape: {df.shape}")
    
    target = "Risk_Type"
    if target not in df.columns:
        print(f"{target} not found. Columns: {df.columns.tolist()}")
    else:
        features = [c for c in df.columns if c != target]
        
        X = df[features].copy()
        y = df[target].copy()
        
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
        X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True)).fillna(0)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        print("Before balancing:", np.bincount(y_train))
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print("After balancing:", np.bincount(y_train))
        
        model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        print(f"Train Acc: {accuracy_score(y_train, train_pred):.4f}")
        print(f"Test Acc:  {accuracy_score(y_test, test_pred):.4f}")
        
        # What if we reverse the label encoder?
        print("Test actual:", y_test[:10])
        print("Test pred:  ", test_pred[:10])
