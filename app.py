"""
Microplastic Risk Analysis System
Complete Dashboard with Home, Preprocessing, Feature Selection, 
Modeling, Cross Validation, and Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from imblearn.over_sampling import SMOTE
import warnings
import io

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Microplastic Risk Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.markdown("# 🔬 Microplastic Risk Analysis")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", 
     "⚙️ Preprocessing", 
     "🎯 Feature Selection", 
     "🤖 Modeling", 
     "🔄 Cross Validation",
     "📊 Visualization"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Status")
if 'data' in st.session_state and st.session_state.data is not None:
    st.sidebar.success(f"✅ Data Loaded\n{st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
else:
    st.sidebar.warning("⚠️ No data loaded")

# ==================== DATA LOADING ====================
def load_data(uploaded_file):
    """Load dataset with multiple encoding support"""
    try:
        if uploaded_file.name.endswith('.csv'):
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except:
                    continue
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported format. Please upload CSV or Excel.")
            return None
        
        st.session_state.data = df
        return df
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


# ==================== HOME PAGE ====================
def home_page():
    st.markdown('<h1 class="main-header">🏠 Home - Dataset Overview</h1>', unsafe_allow_html=True)
    
    # File upload section
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 Upload Dataset (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Generate Sample Data", use_container_width=True):
            # ============================================
            # OBJECTIVE 1: Load and explore the dataset
            # ============================================
            # [INSERT YOUR CODE HERE]
            # Example:
            np.random.seed(42)
            n = 500
            data = {
                'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
                'MP_Count_per_L': np.random.poisson(lam=50, size=n),
                'Particle_Size_um': np.random.normal(100, 30, n),
                'Risk_Score': np.random.uniform(0, 100, n),
                'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n),
                'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n),
                'Polymer_Type': np.random.choice(['PE', 'PP', 'PS', 'PET', 'PVC'], n),
                'Water_Source': np.random.choice(['River', 'Lake', 'Ocean', 'Groundwater'], n),
                'pH': np.random.normal(7, 0.5, n),
                'Temperature_C': np.random.normal(20, 5, n),
                'Location': np.random.choice(['Urban', 'Rural', 'Industrial', 'Coastal'], n)
            }
            st.session_state.data = pd.DataFrame(data)
            st.success("✅ Sample dataset generated!")
            st.rerun()
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success(f"✅ Dataset loaded! {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display data if loaded
    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data
        
        # ============================================
        # OBJECTIVE 1: Load and explore the dataset
        # ============================================
        st.markdown("---")
        st.markdown('<h2 class="section-header">📊 Objective 1: Load and Explore Dataset</h2>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        # Example code:
        # - Display shape
        # - Display info
        # - Display first 5 rows
        # - Display descriptive statistics
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        st.markdown('<h3 class="sub-header">📋 Table: First 5 Rows</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        
        st.markdown('<h3 class="sub-header">📊 Dataset Info</h3>', unsafe_allow_html=True)
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.markdown('<h3 class="sub-header">📈 Descriptive Statistics</h3>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)


# ==================== PREPROCESSING PAGE ====================
def preprocessing_page():
    st.markdown('<h1 class="main-header">⚙️ Data Preprocessing</h1>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please load data from Home page first!")
        return
    
    df = st.session_state.data.copy()
    
    # Preprocessing Options
    preprocessing_tabs = st.tabs([
        "🧹 Missing Values", 
        "🎯 Outlier Detection", 
        "📏 Feature Scaling", 
        "🔄 Encoding",
        "📊 Preprocessing Summary"
    ])
    
    # Tab 1: Missing Values
    with preprocessing_tabs[0]:
        st.markdown('<h2 class="section-header">🧹 Handle Missing Values</h2>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        # Example:
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values,
            'Data Type': df.dtypes.values
        })
        st.markdown("### Missing Values Table")
        st.dataframe(missing_data[missing_data['Missing Count'] > 0], use_container_width=True)
        
        if st.button("🔧 Fill Missing Values (Median/Mode)"):
            # [INSERT YOUR CODE HERE]
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            st.session_state.processed_data = df
            st.success("✅ Missing values handled!")
    
    # Tab 2: Outlier Detection
    with preprocessing_tabs[1]:
        st.markdown('<h2 class="section-header">🎯 Outlier Detection & Handling</h2>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)
        
        if selected_col:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.boxplot(df[selected_col].dropna())
                ax.set_title(f'Box Plot - {selected_col}')
                ax.set_ylabel(selected_col)
                st.pyplot(fig)
            
            with col2:
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[selected_col] < lower) | (df[selected_col] > upper)]
                
                outlier_stats = pd.DataFrame({
                    'Metric': ['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outliers Found'],
                    'Value': [f'{Q1:.2f}', f'{Q3:.2f}', f'{IQR:.2f}', 
                             f'{lower:.2f}', f'{upper:.2f}', len(outliers)]
                })
                st.dataframe(outlier_stats, use_container_width=True, hide_index=True)
    
    # Tab 3: Feature Scaling
    with preprocessing_tabs[2]:
        st.markdown('<h2 class="section-header">📏 Feature Scaling</h2>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if st.button("Apply StandardScaler"):
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
            st.session_state.processed_data = df
            st.success("✅ Scaling applied!")
            st.dataframe(df[numeric_cols].head(), use_container_width=True)
    
    # Tab 4: Encoding
    with preprocessing_tabs[3]:
        st.markdown('<h2 class="section-header">🔄 Categorical Encoding</h2>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write(f"Categorical columns: {categorical_cols}")
        
        if st.button("Apply One-Hot Encoding"):
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            st.session_state.processed_data = df_encoded
            st.success(f"✅ Encoding applied! New shape: {df_encoded.shape}")
            st.dataframe(df_encoded.head(), use_container_width=True)
    
    # Tab 5: Preprocessing Summary
    with preprocessing_tabs[4]:
        st.markdown('<h2 class="section-header">📊 Preprocessing Summary</h2>', unsafe_allow_html=True)
        
        if st.session_state.processed_data is not None:
            processed_df = st.session_state.processed_data
            st.markdown(f"**Original Shape:** {st.session_state.data.shape}")
            st.markdown(f"**Processed Shape:** {processed_df.shape}")
            st.dataframe(processed_df.describe(), use_container_width=True)
        else:
            st.info("No preprocessing applied yet.")


# ==================== FEATURE SELECTION PAGE ====================
def feature_selection_page():
    st.markdown('<h1 class="main-header">🎯 Feature Selection</h1>', unsafe_allow_html=True)
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    if data is None:
        st.warning("⚠️ Please load data first!")
        return
    
    df = data.copy()
    
    st.markdown('<h2 class="section-header">📊 Objective: Feature Selection using Mutual Information</h2>', unsafe_allow_html=True)
    
    # Select target variable
    target_col = st.selectbox("Select Target Variable", df.columns.tolist())
    
    # ============================================
    # OBJECTIVE 3: Feature selection using mutual information
    # ============================================
    if st.button("🔍 Calculate Feature Importance", type="primary"):
        
        # [INSERT YOUR CODE HERE]
        # Your mutual information code goes here
        # Example:
        
        # Prepare features
        feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # Method 1: Mutual Information
        st.markdown('<h3 class="sub-header">📈 Mutual Information Scores</h3>', unsafe_allow_html=True)
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'Feature': feature_cols,
            'Mutual Information': mi_scores
        }).sort_values('Mutual Information', ascending=False)
        
        st.dataframe(mi_df, use_container_width=True, hide_index=True)
        
        # Plot
        fig = px.bar(mi_df.head(20), x='Mutual Information', y='Feature', 
                    orientation='h', title='Mutual Information Scores')
        st.plotly_chart(fig, use_container_width=True)
        
        # Method 2: Chi-Squared
        st.markdown('<h3 class="sub-header">🔢 Chi-Squared Scores</h3>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        X_scaled = X - X.min() + 1
        chi2_scores, p_values = chi2(X_scaled, y)
        chi2_df = pd.DataFrame({
            'Feature': feature_cols,
            'Chi2 Score': chi2_scores,
            'P-Value': p_values
        }).sort_values('Chi2 Score', ascending=False)
        
        st.dataframe(chi2_df, use_container_width=True, hide_index=True)
        
        fig = px.bar(chi2_df.head(20), x='Chi2 Score', y='Feature',
                    orientation='h', title='Chi-Squared Scores')
        st.plotly_chart(fig, use_container_width=True)
        
        # Method 3: Random Forest Importance
        st.markdown('<h3 class="sub-header">🌲 Random Forest Feature Importance</h3>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(rf_df, use_container_width=True, hide_index=True)
        
        fig = px.bar(rf_df.head(20), x='Importance', y='Feature',
                    orientation='h', title='Random Forest Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
        
        # Select top features
        st.markdown('<h3 class="sub-header">✅ Top Selected Features</h3>', unsafe_allow_html=True)
        top_k = st.slider("Select top K features", 5, min(50, len(feature_cols)), 10)
        top_features = rf_df.head(top_k)['Feature'].tolist()
        
        st.success(f"Top {top_k} Features: {', '.join(top_features)}")
        st.session_state.selected_features = top_features


# ==================== MODELING PAGE ====================
def modeling_page():
    st.markdown('<h1 class="main-header">🤖 Model Training</h1>', unsafe_allow_html=True)
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    if data is None:
        st.warning("⚠️ Please load data first!")
        return
    
    df = data.copy()
    
    # ============================================
    # OBJECTIVE: Model Training
    # ============================================
    st.markdown('<h2 class="section-header">Objective: Train Classification Models</h2>', unsafe_allow_html=True)
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        target_col = st.selectbox("Target Variable", df.columns.tolist())
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    with col3:
        use_smote = st.checkbox("Use SMOTE", value=True)
    
    # Feature selection
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    selected_features = st.multiselect(
        "Select Features",
        feature_cols,
        default=st.session_state.get('selected_features', feature_cols[:5])
    )
    
    if st.button("🚀 Train Models", type="primary"):
        if len(selected_features) == 0:
            st.error("Select at least one feature!")
            return
        
        # [INSERT YOUR CODE HERE]
        # Your model training code goes here
        
        X = df[selected_features].fillna(0)
        y = df[target_col]
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # SMOTE
        if use_smote:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                st.info(f"✅ SMOTE applied. Training samples: {X_train.shape[0]}")
            except Exception as e:
                st.warning(f"SMOTE not applied: {e}")
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        results = []
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                })
            except Exception as e:
                st.error(f"Error with {name}: {e}")
        
        # Results table
        if results:
            results_df = pd.DataFrame(results)
            st.markdown('<h3 class="sub-header">📊 Model Performance Table</h3>', unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Visualization
            fig = px.bar(results_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                        barmode='group', title='Model Performance Comparison')
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model
            best_model = results_df.loc[results_df['F1 Score'].idxmax()]
            st.success(f"🏆 Best Model: {best_model['Model']} (F1: {best_model['F1 Score']:.4f})")
            
            st.session_state.models_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test


# ==================== CROSS VALIDATION PAGE ====================
def cross_validation_page():
    st.markdown('<h1 class="main-header">🔄 Cross Validation</h1>', unsafe_allow_html=True)
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    if data is None:
        st.warning("⚠️ Please load data first!")
        return
    
    df = data.copy()
    
    # ============================================
    # OBJECTIVE: Cross Validation
    # ============================================
    st.markdown('<h2 class="section-header">Objective: Cross Validation & Model Evaluation</h2>', unsafe_allow_html=True)
    
    target_col = st.selectbox("Target Variable", df.columns.tolist(), key='cv_target')
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    selected_features = st.multiselect(
        "Select Features for CV",
        feature_cols,
        default=st.session_state.get('selected_features', feature_cols[:5]),
        key='cv_features'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
    with col2:
        cv_metric = st.selectbox("Evaluation Metric", ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
    
    if st.button("🔄 Run Cross Validation", type="primary"):
        if len(selected_features) == 0:
            st.error("Select at least one feature!")
            return
        
        # [INSERT YOUR CODE HERE]
        # Your cross validation code goes here
        
        X = df[selected_features].fillna(0)
        y = df[target_col]
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Cross validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=cv_metric)
                cv_results.append({
                    'Model': name,
                    'Mean Score': scores.mean(),
                    'Std Score': scores.std(),
                    'Min Score': scores.min(),
                    'Max Score': scores.max()
                })
            except Exception as e:
                st.error(f"Error with {name}: {e}")
        
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            st.markdown('<h3 class="sub-header">📊 Cross Validation Results</h3>', unsafe_allow_html=True)
            st.dataframe(cv_df, use_container_width=True, hide_index=True)
            
            # Visualization
            fig = go.Figure()
            for name, model in models.items():
                if name in [r['Model'] for r in cv_results]:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=cv_metric)
                    fig.add_trace(go.Box(y=scores, name=name, boxmean='sd'))
            
            fig.update_layout(
                title=f'Cross Validation Scores ({cv_folds}-Fold)',
                yaxis_title=cv_metric,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model
            best_cv = cv_df.loc[cv_df['Mean Score'].idxmax()]
            st.success(f"🏆 Best CV Model: {best_cv['Model']} (Mean {cv_metric}: {best_cv['Mean Score']:.4f} ± {best_cv['Std Score']:.4f})")


# ==================== VISUALIZATION PAGE ====================
def visualization_page():
    st.markdown('<h1 class="main-header">📊 Visualization Dashboard</h1>', unsafe_allow_html=True)
    
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    if data is None:
        st.warning("⚠️ Please load data first!")
        return
    
    df = data.copy()
    
    # ============================================
    # OBJECTIVE 2: Analyze Risk Score Distribution
    # ============================================
    st.markdown('<h2 class="section-header">📊 Objective: Analyze Risk Score Distribution</h2>', unsafe_allow_html=True)
    
    if 'Risk_Score' in df.columns:
        # [INSERT YOUR CODE HERE]
        # Your Risk Score distribution code
        
        risk_tabs = st.tabs(["📊 Histogram", "📦 Box Plot", "📈 Combined View"])
        
        with risk_tabs[0]:
            st.markdown('<h3 class="sub-header">Histogram of Risk Score</h3>', unsafe_allow_html=True)
            
            # [INSERT YOUR CODE HERE]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='Risk_Score', kde=True, bins=30, color='skyblue', edgecolor='black')
            ax.set_title('Distribution of Risk Score', fontsize=16, fontweight='bold')
            ax.set_xlabel('Risk Score', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.axvline(df['Risk_Score'].mean(), color='red', linestyle='--', label=f'Mean: {df["Risk_Score"].mean():.2f}')
            ax.axvline(df['Risk_Score'].median(), color='green', linestyle='--', label=f'Median: {df["Risk_Score"].median():.2f}')
            ax.legend()
            st.pyplot(fig)
        
        with risk_tabs[1]:
            st.markdown('<h3 class="sub-header">Box Plot of Risk Score</h3>', unsafe_allow_html=True)
            
            # [INSERT YOUR CODE HERE]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, y='Risk_Score', color='lightblue')
            ax.set_title('Box Plot of Risk Score', fontsize=16, fontweight='bold')
            ax.set_ylabel('Risk Score', fontsize=12)
            st.pyplot(fig)
        
        with risk_tabs[2]:
            st.markdown('<h3 class="sub-header">Combined View</h3>', unsafe_allow_html=True)
            
            # [INSERT YOUR CODE HERE]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            sns.histplot(data=df, x='Risk_Score', kde=True, bins=30, color='skyblue', edgecolor='black', ax=ax1)
            ax1.set_title('Histogram', fontsize=14)
            ax1.set_xlabel('Risk Score')
            
            sns.boxplot(data=df, y='Risk_Score', color='lightblue', ax=ax2)
            ax2.set_title('Box Plot', fontsize=14)
            ax2.set_ylabel('Risk Score')
            
            plt.suptitle('Risk Score Distribution Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Statistics table
        st.markdown('<h3 class="sub-header">📈 Risk Score Statistics</h3>', unsafe_allow_html=True)
        
        # [INSERT YOUR CODE HERE]
        risk_stats = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Q1', 'Q3', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                len(df['Risk_Score'].dropna()),
                f"{df['Risk_Score'].mean():.2f}",
                f"{df['Risk_Score'].median():.2f}",
                f"{df['Risk_Score'].std():.2f}",
                f"{df['Risk_Score'].min():.2f}",
                f"{df['Risk_Score'].quantile(0.25):.2f}",
                f"{df['Risk_Score'].quantile(0.75):.2f}",
                f"{df['Risk_Score'].max():.2f}",
                f"{df['Risk_Score'].skew():.2f}",
                f"{df['Risk_Score'].kurtosis():.2f}"
            ]
        })
        st.dataframe(risk_stats, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ 'Risk_Score' column not found in dataset!")
    
    # Additional visualizations
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Additional Visualizations</h2>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols)
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
    
    color_col = st.selectbox("Color by (optional)", ['None'] + df.columns.tolist())
    
    if st.button("Generate Plot"):
        # [INSERT YOUR CODE HERE]
        if color_col != 'None':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                           title=f'{y_col} vs {x_col}')
        else:
            fig = px.scatter(df, x=x_col, y=y_col, 
                           title=f'{y_col} vs {x_col}')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🔥 Correlation Heatmap</h3>', unsafe_allow_html=True)
    
    if st.button("Generate Correlation Heatmap"):
        # [INSERT YOUR CODE HERE]
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
        st.pyplot(fig)


# ==================== MAIN FUNCTION ====================
def main():
    """Main function to run the app"""
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "⚙️ Preprocessing":
        preprocessing_page()
    elif page == "🎯 Feature Selection":
        feature_selection_page()
    elif page == "🤖 Modeling":
        modeling_page()
    elif page == "🔄 Cross Validation":
        cross_validation_page()
    elif page == "📊 Visualization":
        visualization_page()


if __name__ == "__main__":
    main()
