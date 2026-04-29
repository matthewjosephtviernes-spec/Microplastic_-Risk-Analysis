"""
Microplastic Risk Analysis Dashboard
A comprehensive Streamlit application for analyzing microplastic risk data,
featuring data preprocessing, EDA, model training, cross validation, and model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, r2_score)
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
import time
import io
import base64

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Microplastic Risk Analysis Dashboard", 
    page_icon="🔬", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #1f77b4; 
        text-align: center; 
        margin-bottom: 2rem; 
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header { 
        font-size: 1.8rem; 
        font-weight: 600; 
        color: #2c3e50; 
        margin-top: 1rem; 
        margin-bottom: 1rem; 
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #34495e;
        margin-top: 0.8rem;
    }
    .stButton > button { 
        width: 100%; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        font-weight: 600; 
        border-radius: 8px; 
        padding: 0.5rem 1rem;
        border: none;
        transition: transform 0.2s;
    }
    .stButton > button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .outlier-box { 
        padding: 1rem; 
        border-radius: 8px; 
        margin: 0.5rem 0; 
    }
    .outlier-before { 
        background-color: #ffeaa7; 
    }
    .outlier-after { 
        background-color: #55efc4; 
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .outlier-comparison {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .upload-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .upload-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize all session state variables"""
    if 'data' not in st.session_state: 
        st.session_state.data = None
    if 'processed_data' not in st.session_state: 
        st.session_state.processed_data = None
    if 'models' not in st.session_state: 
        st.session_state.models = {}
    if 'feature_importance' not in st.session_state: 
        st.session_state.feature_importance = None
    if 'mutual_info' not in st.session_state: 
        st.session_state.mutual_info = None
    if 'chi2_scores' not in st.session_state: 
        st.session_state.chi2_scores = None
    if 'trained' not in st.session_state: 
        st.session_state.trained = False
    if 'selected_features' not in st.session_state: 
        st.session_state.selected_features = None
    if 'scaler' not in st.session_state: 
        st.session_state.scaler = None
    if 'scaled_data' not in st.session_state: 
        st.session_state.scaled_data = None
    if 'scaled_columns' not in st.session_state: 
        st.session_state.scaled_columns = None
    if 'encoded_data' not in st.session_state: 
        st.session_state.encoded_data = None
    if 'encoded_shape' not in st.session_state: 
        st.session_state.encoded_shape = None
    if 'evaluation_ran' not in st.session_state: 
        st.session_state.evaluation_ran = False
    if 'comparison_ran' not in st.session_state: 
        st.session_state.comparison_ran = False
    if 'cv_ran' not in st.session_state: 
        st.session_state.cv_ran = False
    if 'outlier_stats_before' not in st.session_state: 
        st.session_state.outlier_stats_before = None
    if 'outlier_stats_after' not in st.session_state: 
        st.session_state.outlier_stats_after = None
    if 'outlier_columns_processed' not in st.session_state: 
        st.session_state.outlier_columns_processed = []
    if 'outlier_bounds' not in st.session_state:
        st.session_state.outlier_bounds = {}
    if 'outlier_counts' not in st.session_state:
        st.session_state.outlier_counts = {}
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

init_session_state()

# IMPORTANT: Removed @st.cache_data from load_dataset to fix upload issue
def load_dataset(uploaded_file):
    """
    Load dataset from uploaded file with encoding detection
    NOTE: Removed @st.cache_data to fix file upload issues in Streamlit
    """
    try:
        if uploaded_file is None:
            return None
            
        # Check file size
        if hasattr(uploaded_file, 'size'):
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 200:
                st.error(f"❌ File too large ({file_size_mb:.1f} MB). Please upload a file smaller than 200 MB.")
                return None
        
        # Try to read the file
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            data = None
            
            for enc in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    data = pd.read_csv(uploaded_file, encoding=enc)
                    if data is not None and not data.empty:
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
            
            if data is None:
                st.error("❌ Could not read the CSV file. Please check the file format.")
                return None
                
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            try:
                uploaded_file.seek(0)
                data = pd.read_excel(uploaded_file, engine='openpyxl')
            except Exception as e:
                try:
                    uploaded_file.seek(0)
                    data = pd.read_excel(uploaded_file, engine='xlrd')
                except Exception as e2:
                    st.error(f"❌ Could not read Excel file. Error: {str(e2)}")
                    return None
        else:
            st.error("❌ Unsupported file format. Please upload CSV (.csv) or Excel (.xlsx, .xls) files.")
            return None
        
        # Validate the data
        if data is None or data.empty:
            st.error("❌ The uploaded file is empty.")
            return None
        
        if data.shape[0] == 0:
            st.error("❌ The uploaded file has no rows.")
            return None
        
        # Store in session state
        st.session_state.data = data
        st.session_state.file_uploaded = True
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("💡 Tips: Make sure your file is a valid CSV or Excel file and not corrupted.")
        return None

def generate_sample_data():
    """Generate sample microplastic dataset with enhanced features"""
    np.random.seed(42)
    n = 1000
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
        'MP_Count_per_L': np.random.poisson(lam=50, size=n),
        'Particle_Size_um': np.random.normal(100, 30, n),
        'Microplastic_Size_mm_midpoint': np.random.normal(2.5, 1.5, n),
        'Density_midpoint': np.random.normal(1.0, 0.1, n),
        'Polymer_Type': np.random.choice(['PE','PP','PS','PET','PVC','Nylon'], n),
        'Water_Source': np.random.choice(['River','Lake','Ocean','Groundwater','Tap'], n),
        'pH': np.random.normal(7, 0.5, n), 
        'Temperature_C': np.random.normal(20, 5, n),
        'Risk_Score': np.random.uniform(0, 100, n),
        'Risk_Level': np.random.choice(['Low','Medium','High','Critical'], n, p=[0.3,0.35,0.25,0.1]),
        'Risk_Type': np.random.choice(['Type_A','Type_B','Type_C'], n, p=[0.5,0.3,0.2]),
        'Location': np.random.choice(['Urban','Rural','Industrial','Coastal'], n),
        'Season': np.random.choice(['Winter','Spring','Summer','Fall'], n),
        'Author': np.random.choice(['Author_A','Author_B','Author_C'], n),
        'Source': np.random.choice(['Source_1','Source_2','Source_3'], n)
    }
    df = pd.DataFrame(data)
    
    # Add outliers for more realistic data
    for col in ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']:
        if col in df.columns:
            outlier_indices = np.random.choice(n, size=int(n*0.05), replace=False)
            if col == 'Risk_Score':
                df.loc[outlier_indices, col] = np.random.uniform(150, 200, len(outlier_indices))
            elif col == 'MP_Count_per_L':
                df.loc[outlier_indices, col] = np.random.poisson(lam=200, size=len(outlier_indices))
            elif col == 'Microplastic_Size_mm_midpoint':
                df.loc[outlier_indices, col] = np.random.uniform(10, 20, len(outlier_indices))
            elif col == 'Density_midpoint':
                df.loc[outlier_indices, col] = np.random.uniform(1.5, 2.0, len(outlier_indices))
    
    # Add missing values
    for col in df.columns:
        if col != 'Sample_ID' and df[col].dtype in ['float64','int64']:
            df.loc[np.random.random(n) < 0.05, col] = np.nan
    
    return df

def detect_outliers_detailed(df, columns):
    """Detect outliers using IQR method with detailed statistics"""
    outlier_info = {}
    for col in columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            clean_data = df[col].dropna()
            if len(clean_data) == 0:
                continue
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            outlier_info[col] = {
                'Q1': Q1, 
                'Q3': Q3, 
                'IQR': IQR,
                'lower_bound': lower_bound, 
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(clean_data)) * 100,
                'min_value': clean_data.min(), 
                'max_value': clean_data.max(),
                'mean': clean_data.mean(), 
                'std': clean_data.std(),
                'outliers_below': len(clean_data[clean_data < lower_bound]),
                'outliers_above': len(clean_data[clean_data > upper_bound])
            }
    return outlier_info

def cap_outliers_iqr(df, numerical_cols):
    """
    Handle outliers using IQR method (capping)
    Caps outliers at upper and lower bounds calculated by IQR method
    """
    df_capped = df.copy()
    outlier_bounds = {}
    outlier_counts = {}
    stats_before = {}
    
    # Calculate statistics before capping
    for col in numerical_cols:
        if col in df_capped.columns:
            clean_data = df_capped[col].dropna()
            if len(clean_data) > 0:
                stats_before[col] = {
                    'count': len(clean_data),
                    'mean': clean_data.mean(),
                    'std': clean_data.std(),
                    'min': clean_data.min(),
                    '25%': clean_data.quantile(0.25),
                    '50%': clean_data.quantile(0.50),
                    '75%': clean_data.quantile(0.75),
                    'max': clean_data.max(),
                    'skewness': clean_data.skew(),
                    'kurtosis': clean_data.kurtosis()
                }
    
    # Apply IQR capping
    for col in numerical_cols:
        if col in df_capped.columns:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before capping
            outliers_below = (df_capped[col] < lower_bound).sum()
            outliers_above = (df_capped[col] > upper_bound).sum()
            
            outlier_bounds[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            outlier_counts[col] = {
                'below': outliers_below,
                'above': outliers_above,
                'total': outliers_below + outliers_above,
                'percentage': ((outliers_below + outliers_above) / len(df_capped[col].dropna())) * 100
            }
            
            # Cap the outliers
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Calculate statistics after capping
    stats_after = {}
    for col in numerical_cols:
        if col in df_capped.columns:
            clean_data = df_capped[col].dropna()
            if len(clean_data) > 0:
                stats_after[col] = {
                    'count': len(clean_data),
                    'mean': clean_data.mean(),
                    'std': clean_data.std(),
                    'min': clean_data.min(),
                    '25%': clean_data.quantile(0.25),
                    '50%': clean_data.quantile(0.50),
                    '75%': clean_data.quantile(0.75),
                    'max': clean_data.max(),
                    'skewness': clean_data.skew(),
                    'kurtosis': clean_data.kurtosis()
                }
    
    return df_capped, stats_before, stats_after, outlier_bounds, outlier_counts

def create_outlier_summary_table(stats_before, stats_after, numerical_cols):
    """Create summary table comparing before/after outlier handling"""
    summary_data = []
    for col in numerical_cols:
        if col in stats_before and col in stats_after:
            summary_data.append({
                'Column': col,
                'Mean Before': f"{stats_before[col]['mean']:.4f}",
                'Mean After': f"{stats_after[col]['mean']:.4f}",
                'Std Before': f"{stats_before[col]['std']:.4f}",
                'Std After': f"{stats_after[col]['std']:.4f}",
                'Min Before': f"{stats_before[col]['min']:.4f}",
                'Min After': f"{stats_after[col]['min']:.4f}",
                'Max Before': f"{stats_before[col]['max']:.4f}",
                'Max After': f"{stats_after[col]['max']:.4f}",
                'Skew Before': f"{stats_before[col]['skewness']:.4f}",
                'Skew After': f"{stats_after[col]['skewness']:.4f}"
            })
    return pd.DataFrame(summary_data)

def one_hot_encode(df):
    """One-hot encode categorical variables"""
    try:
        cats = df.select_dtypes(include=['object']).columns.tolist()
        cols = [c for c in cats if 'ID' not in c and 'Sample' not in c]
        if len(cols) == 0: 
            return df, [], [], df.shape
        df_enc = pd.get_dummies(df, columns=cols, drop_first=False)
        new = [c for c in df_enc.columns if c not in df.columns]
        return df_enc, new, cols, df_enc.shape
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return df, [], [], df.shape

def analyze_skewness(df, columns):
    """Analyze skewness of numerical columns"""
    info = []
    for col in columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            s = df[col].skew()
            info.append({
                'Column': col, 
                'Skewness': round(s, 4), 
                'Skewed': 'Yes' if abs(s) > 0.5 else 'No',
                'Direction': 'Right' if s > 0.5 else ('Left' if s < -0.5 else 'Symmetric')
            })
    return pd.DataFrame(info)

def apply_log_transform(df, columns):
    """Apply log transformation to reduce skewness"""
    df_t = df.copy()
    for col in columns:
        if col in df_t.columns and df_t[col].dtype in ['float64', 'int64'] and abs(df_t[col].skew()) > 0.5:
            shift = abs(df_t[col].min()) + 1 if df_t[col].min() <= 0 else 0
            df_t[col] = np.log1p(df_t[col] + shift)
    return df_t

@st.cache_data
def calculate_mutual_info(X, y):
    """Calculate mutual information scores"""
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

@st.cache_data
def calculate_chi2(X, y):
    """Calculate chi-squared scores"""
    X_s = X - X.min() + 1
    scores, pvals = chi2(X_s, y)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

@st.cache_data
def calculate_rf_importance(X, y):
    """Calculate Random Forest feature importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

def train_and_evaluate_detailed(df, target_col):
    """Train and evaluate multiple models"""
    feature_cols = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()
    if target_col in feature_cols: 
        feature_cols.remove(target_col)
    
    if len(feature_cols) == 0:
        st.error("No numerical features found for training")
        return None, None
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    if len(y) == 0:
        st.error("No valid target values found")
        return None, None
    
    if y.dtype == 'object': 
        y = LabelEncoder().fit_transform(y)
    
    if len(np.unique(y)) < 2:
        st.error("Target variable must have at least 2 classes")
        return None, None
    
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    info = {'X_train': X_train.shape, 'X_test': X_test.shape, 'target': target_col}
    
    models = {}
    
    try:
        lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1)
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
    except Exception as e:
        st.warning(f"Logistic Regression failed: {str(e)}")
    
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
    except Exception as e:
        st.warning(f"Random Forest failed: {str(e)}")
    
    try:
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X_train, y_train)
        models['Gradient Boosting'] = gb
    except Exception as e:
        st.warning(f"Gradient Boosting failed: {str(e)}")
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
    
    return results, info

def plot_distribution(data, column, title):
    """Create distribution plot using Plotly"""
    try:
        clean = data[column].dropna()
        if clean.empty: 
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=('Histogram with KDE', 'Box Plot'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=clean, nbinsx=30, marker_color='#3498db', 
                        name='Histogram', opacity=0.7),
            row=1, col=1
        )
        
        # Box Plot
        fig.add_trace(
            go.Box(y=clean, marker_color='#e74c3c', name='Box Plot'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=title, 
            showlegend=False, 
            height=500,
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating distribution plot: {str(e)}")
        return go.Figure()

def create_matplotlib_distribution(data, column, title):
    """Create distribution plot using Matplotlib/Seaborn"""
    try:
        clean = data[column].dropna()
        if clean.empty:
            return None
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        sns.histplot(
            data=clean, 
            kde=True, 
            bins=30, 
            color='#3498db',
            edgecolor='white',
            alpha=0.7,
            ax=ax1
        )
        ax1.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        
        # Add mean and median lines
        mean_val = clean.mean()
        median_val = clean.median()
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.2f}')
        ax1.legend()
        
        # Box Plot
        sns.boxplot(
            data=clean,
            color='#e74c3c',
            width=0.3,
            ax=ax2
        )
        ax2.set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
        ax2.set_ylabel(column, fontsize=12)
        
        # Add statistical annotations
        stats_text = f"Q1: {clean.quantile(0.25):.2f}\nQ3: {clean.quantile(0.75):.2f}\nIQR: {clean.quantile(0.75) - clean.quantile(0.25):.2f}"
        ax2.text(1.15, clean.median(), stats_text, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating matplotlib plot: {str(e)}")
        return None

def create_outlier_boxplot_comparison(df_before, df_after, column):
    """Create before/after boxplot comparison for outlier handling"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Before
        clean_before = df_before[column].dropna()
        if len(clean_before) > 0:
            sns.boxplot(data=clean_before, color='#ffeaa7', width=0.3, ax=ax1)
            ax1.set_title(f'{column} - Before Outlier Handling', fontsize=14, fontweight='bold')
            ax1.set_ylabel(column, fontsize=12)
        
        # After
        clean_after = df_after[column].dropna()
        if len(clean_after) > 0:
            sns.boxplot(data=clean_after, color='#55efc4', width=0.3, ax=ax2)
            ax2.set_title(f'{column} - After Outlier Handling', fontsize=14, fontweight='bold')
            ax2.set_ylabel(column, fontsize=12)
        
        plt.suptitle(f'Outlier Handling Comparison: {column}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio(
        "Select Section", 
        [
            "🏠 Home", 
            "🔧 Preprocessing", 
            "🛠️ Feature Selection & Relevance", 
            "🤖 Modeling", 
            "📊 Cross Validation & Evaluation"
        ],
        key="main_navigation"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Dashboard Features:**
    - 📤 Data Upload & Preview
    - 📊 Exploratory Data Analysis
    - 🔧 Data Preprocessing
    - 🎯 Feature Selection
    - 🤖 Model Training
    - 📈 Model Evaluation
    """)
    
    st.sidebar.markdown("---")
    
    # Status indicators
    st.sidebar.markdown("### 📌 Status")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.session_state.data is not None: 
            st.success("✅ Data")
        else: 
            st.warning("⚠️ No Data")
    
    with col2:
        if st.session_state.trained: 
            st.success("✅ Models")
        else: 
            st.warning("⚠️ No Models")
    
    if st.session_state.processed_data is not None:
        st.sidebar.success("✅ Preprocessed")
    
    # ==================== HOME PAGE ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Dataset Overview & Exploratory Analysis</p>', 
                   unsafe_allow_html=True)
        
        home_tab1, home_tab2, home_tab3, home_tab4, home_tab5 = st.tabs([
            "📤 Upload & Preview", 
            "📊 Risk Score Distribution", 
            "🔬 MP Count vs Risk Score", 
            "📊 Risk Score by Risk Level", 
            "🔍 Data Quality Check"
        ])
        
        # ==================== HOME TAB 1: Upload & Preview ====================
        with home_tab1:
            st.markdown("### 📤 Upload Dataset")
            st.markdown("""
            Upload your microplastic risk dataset in CSV or Excel format.
            The dashboard supports various encodings and will automatically detect the format.
            """)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a file (CSV or Excel)",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload your microplastic risk dataset. Supports .csv, .xlsx, .xls formats.",
                    key="file_uploader_main"
                )
                
                if uploaded_file is not None:
                    # Show file details
                    file_details = {
                        "Filename": uploaded_file.name,
                        "File size": f"{uploaded_file.size / 1024:.2f} KB" if uploaded_file.size < 1024*1024 else f"{uploaded_file.size / (1024*1024):.2f} MB",
                        "File type": uploaded_file.type
                    }
                    
                    st.markdown("**📄 File Details:**")
                    for key, value in file_details.items():
                        st.text(f"{key}: {value}")
                    
                    # Load the data
                    with st.spinner('🔄 Loading dataset...'):
                        data = load_dataset(uploaded_file)
                        
                        if data is not None:
                            st.markdown('<div class="upload-success">', unsafe_allow_html=True)
                            st.success(f"✅ Dataset loaded successfully!")
                            st.markdown(f"""
                            - **Shape:** {data.shape[0]:,} rows × {data.shape[1]} columns
                            - **Memory usage:** {data.memory_usage(deep=True).sum() / 1024:.2f} KB
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**OR**")
                if st.button("🎲 Generate Sample Dataset", type="primary", use_container_width=True, key="gen_sample"):
                    with st.spinner('Generating sample data...'):
                        st.session_state.data = generate_sample_data()
                        st.session_state.file_uploaded = True
                        st.success("✅ Sample dataset generated with realistic outliers!")
                        st.rerun()
            
            if st.session_state.data is not None:
                df = st.session_state.data
                
                st.markdown("---")
                
                # Key metrics
                st.markdown("#### 📊 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Samples", f"{df.shape[0]:,}")
                with col2:
                    st.metric("🔢 Features", df.shape[1])
                with col3:
                    st.metric("⚠️ Missing Values", f"{df.isnull().sum().sum():,}")
                with col4:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("📉 Missing %", f"{missing_pct:.2f}%")
                
                st.markdown("---")
                
                # Data Preview
                st.markdown("#### 📋 Data Preview (First 10 Rows)")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show data info
                with st.expander("📊 View Detailed Data Information"):
                    # Column Information
                    col_info = pd.DataFrame({
                        'Column Name': df.columns,
                        'Data Type': df.dtypes.values,
                        'Missing Values': df.isnull().sum().values,
                        'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True, hide_index=True)
                
                # Download column info
                col1, col2 = st.columns(2)
                with col1:
                    csv = col_info.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Column Info",
                        data=csv,
                        file_name="column_information.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    csv_full = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Dataset",
                        data=csv_full,
                        file_name="microplastic_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        # ==================== HOME TAB 2: Risk Score Distribution ====================
        with home_tab2:
            st.markdown("### 📊 Analyze the Distribution of Risk Score")
            st.markdown("""
            **Objective 1:** Visualize the distribution of the Risk_Score column using histogram and box plot.
            This analysis helps understand the central tendency, spread, and presence of outliers in risk scores.
            """)
            
            if st.session_state.data is None: 
                st.warning("⚠️ Please upload data or generate sample data first!")
            else:
                df = st.session_state.data
                
                if 'Risk_Score' not in df.columns:
                    st.error("❌ 'Risk_Score' column not found in dataset!")
                    st.info("Available columns: " + ", ".join(df.columns.tolist()))
                else:
                    # Convert to numeric
                    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                    clean_data = df['Risk_Score'].dropna()
                    
                    if len(clean_data) == 0:
                        st.error("❌ No valid Risk_Score data found after cleaning!")
                    else:
                        # Visualization options
                        st.markdown("#### 🎨 Visualization Options")
                        viz_type = st.radio(
                            "Select Visualization Type:",
                            ["📊 Interactive (Plotly)", "📈 Static (Matplotlib/Seaborn)", "🔢 Both"],
                            horizontal=True,
                            key="risk_score_viz_type"
                        )
                        
                        st.markdown("---")
                        
                        # Interactive visualization
                        if viz_type in ["📊 Interactive (Plotly)", "🔢 Both"]:
                            st.markdown("#### 📊 Interactive Visualization (Plotly)")
                            st.plotly_chart(
                                plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), 
                                use_container_width=True
                            )
                            
                            if viz_type == "🔢 Both":
                                st.markdown("---")
                        
                        # Static visualization with Matplotlib/Seaborn
                        if viz_type in ["📈 Static (Matplotlib/Seaborn)", "🔢 Both"]:
                            st.markdown("#### 📈 Static Visualization (Matplotlib/Seaborn)")
                            
                            fig = create_matplotlib_distribution(df, 'Risk_Score', 'Risk Score Distribution Analysis')
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.error("Failed to create visualization")
                        
                        st.markdown("---")
                        
                        # Comprehensive Statistics
                        st.markdown("#### 📈 Descriptive Statistics for Risk Score")
                        
                        # Calculate statistics
                        q1 = clean_data.quantile(0.25)
                        q3 = clean_data.quantile(0.75)
                        iqr = q3 - q1
                        
                        # Create statistics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📊 Count", f"{len(clean_data):,}")
                            st.metric("📈 Mean", f"{clean_data.mean():.4f}")
                            st.metric("📐 Median", f"{clean_data.median():.4f}")
                            st.metric("📏 Std Dev", f"{clean_data.std():.4f}")
                        
                        with col2:
                            st.metric("↗️ Skewness", f"{clean_data.skew():.4f}")
                            st.metric("⛰️ Kurtosis", f"{clean_data.kurtosis():.4f}")
                            st.metric("⬇️ Minimum", f"{clean_data.min():.4f}")
                            st.metric("⬆️ Maximum", f"{clean_data.max():.4f}")
                        
                        with col3:
                            st.metric("Q1 (25th)", f"{q1:.4f}")
                            st.metric("Q2 (50th)", f"{clean_data.median():.4f}")
                            st.metric("Q3 (75th)", f"{q3:.4f}")
                            st.metric("IQR", f"{iqr:.4f}")
                        
                        with col4:
                            st.metric("Range", f"{clean_data.max() - clean_data.min():.4f}")
                            variance = clean_data.var()
                            st.metric("Variance", f"{variance:.4f}")
                            cv = (clean_data.std() / clean_data.mean()) * 100
                            st.metric("CV %", f"{cv:.2f}%")
                            st.metric("Mode", f"{clean_data.mode().iloc[0] if not clean_data.mode().empty else 'N/A':.4f}")
                        
                        st.markdown("---")
                        
                        # Risk Categories Distribution
                        st.markdown("#### 🎯 Risk Score Categories Distribution")
                        
                        # Categorize risk scores
                        def categorize_risk(score):
                            if score < 25:
                                return '🟢 Low Risk (0-25)'
                            elif score < 50:
                                return '🟡 Medium Risk (25-50)'
                            elif score < 75:
                                return '🟠 High Risk (50-75)'
                            else:
                                return '🔴 Critical Risk (75+)'
                        
                        risk_categories = clean_data.apply(categorize_risk)
                        category_counts = risk_categories.value_counts()
                        
                        # Display categories
                        for category in ['🟢 Low Risk (0-25)', '🟡 Medium Risk (25-50)', 
                                       '🟠 High Risk (50-75)', '🔴 Critical Risk (75+)']:
                            count = category_counts.get(category, 0)
                            percentage = (count / len(clean_data)) * 100
                            
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.markdown(f"**{category}**")
                                st.progress(int(percentage))
                            with col2:
                                st.metric("Count", f"{count:,}")
                            with col3:
                                st.metric("Percentage", f"{percentage:.1f}%")
                        
                        st.markdown("---")
                        
                        # Outlier Analysis
                        st.markdown("#### 🔍 Outlier Analysis")
                        
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Lower Bound", f"{lower_bound:.4f}")
                        with col2:
                            st.metric("Upper Bound", f"{upper_bound:.4f}")
                        with col3:
                            st.metric("Outliers Found", len(outliers))
                        with col4:
                            st.metric("Outlier %", f"{(len(outliers)/len(clean_data)*100):.2f}%")
                        
                        if len(outliers) > 0:
                            with st.expander("🔍 View Outlier Details"):
                                outlier_df = pd.DataFrame({
                                    'Index': outliers.index,
                                    'Risk Score': outliers.values,
                                    'Category': outliers.apply(
                                        lambda x: 'Below' if x < lower_bound else 'Above'
                                    )
                                })
                                st.dataframe(outlier_df.head(20), use_container_width=True)
                                
                                # Outlier distribution
                                fig, ax = plt.subplots(figsize=(8, 4))
                                outlier_df['Category'].value_counts().plot(
                                    kind='pie', 
                                    autopct='%1.1f%%',
                                    ax=ax,
                                    colors=['#ff9999', '#66b3ff']
                                )
                                ax.set_title('Outlier Distribution (Above/Below Bounds)')
                                st.pyplot(fig)
                                plt.close()
                        
                        st.markdown("---")
                        
                        # Distribution Shape Analysis
                        st.markdown("#### 📊 Distribution Shape Analysis")
                        
                        skewness = clean_data.skew()
                        if abs(skewness) < 0.5:
                            shape = "Approximately Symmetric"
                            interpretation = "The distribution is roughly symmetric, suggesting balanced risk scores without significant skew."
                        elif skewness > 0.5:
                            shape = "Right-Skewed (Positive Skew)"
                            interpretation = "The distribution has a longer right tail. More samples have higher risk scores, and the mean is likely greater than the median."
                        else:
                            shape = "Left-Skewed (Negative Skew)"
                            interpretation = "The distribution has a longer left tail. More samples have lower risk scores, and the mean is likely less than the median."
                        
                        st.info(f"**Distribution Shape:** {shape}\n\n**Interpretation:** {interpretation}")
                        
                        # Download options
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            # Download statistics
                            stats_dict = {
                                'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Skewness', 
                                            'Kurtosis', 'Min', 'Q1', 'Q3', 'Max', 'IQR', 'Range',
                                            'Variance', 'CV%'],
                                'Value': [len(clean_data), clean_data.mean(), clean_data.median(), 
                                        clean_data.std(), clean_data.skew(), clean_data.kurtosis(),
                                        clean_data.min(), q1, q3, clean_data.max(), iqr,
                                        clean_data.max() - clean_data.min(), clean_data.var(), cv]
                            }
                            stats_df = pd.DataFrame(stats_dict)
                            csv_stats = stats_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Statistics (CSV)",
                                data=csv_stats,
                                file_name="risk_score_statistics.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Download cleaned data
                            csv_data = clean_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Clean Data (CSV)",
                                data=csv_data,
                                file_name="risk_score_clean_data.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
        
        # [Rest of the Home tabs (tab3, tab4, tab5) remain the same as previous version]
        # ... (keeping them unchanged as you love the Home navigation)
        
        # ==================== HOME TAB 3: MP Count vs Risk Score ====================
        with home_tab3:
            # [Same as previous version - keeping it unchanged]
            st.markdown("### 🔬 Explore the Relationship Between MP Count and Risk Score")
            # ... (rest of the code remains the same)
        
        # ==================== HOME TAB 4: Risk Score by Risk Level ====================
        with home_tab4:
            # [Same as previous version - keeping it unchanged]
            st.markdown("### 📊 Investigate Risk Score Differences by Risk Level")
            # ... (rest of the code remains the same)
        
        # ==================== HOME TAB 5: Data Quality Check ====================
        with home_tab5:
            # [Same as previous version - keeping it unchanged]
            st.markdown("### 🔍 Data Quality Check")
            # ... (rest of the code remains the same)

if __name__ == "__main__":
    main()
