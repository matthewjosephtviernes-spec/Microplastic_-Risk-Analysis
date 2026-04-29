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
    .encode-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
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
    if 'encoded_new_cols' not in st.session_state:
        st.session_state.encoded_new_cols = []
    if 'encoded_original_cols' not in st.session_state:
        st.session_state.encoded_original_cols = []
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

init_session_state()

def load_dataset(uploaded_file):
    """Load dataset from uploaded file with encoding detection"""
    try:
        if uploaded_file.name.endswith('.csv'):
            for enc in ['utf-8', 'latin1', 'cp1252']:
                try: 
                    uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, encoding=enc)
                    st.session_state.data = data
                    return data
                except UnicodeDecodeError: 
                    continue
                except Exception as e:
                    continue
            # If all encodings fail
            st.error("Unable to read CSV file with common encodings")
            return None
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
            st.session_state.data = data
            return data
        else: 
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
    except Exception as e: 
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def generate_sample_data():
    """Generate sample microplastic dataset with enhanced features"""
    np.random.seed(42)
    n = 1000
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
        'Latitude': np.random.uniform(-90, 90, n),
        'Longitude': np.random.uniform(-180, 180, n),
        'Microplastic_Size_mm': np.random.normal(2.5, 1.5, n),
        'Density': np.random.normal(1.0, 0.1, n),
        'MP_Count_per_L': np.random.poisson(lam=50, size=n),
        'Particle_Size_um': np.random.normal(100, 30, n),
        'Microplastic_Size_mm_midpoint': np.random.normal(2.5, 1.5, n),
        'Density_midpoint': np.random.normal(1.0, 0.1, n),
        'Polymer_Type': np.random.choice(['PE','PP','PS','PET','PVC','Nylon'], n),
        'Water_Source': np.random.choice(['River','Lake','Ocean','Groundwater','Tap'], n),
        'pH': np.random.choice(['Acidic','Neutral','Alkaline'], n),
        'Salinity': np.random.choice(['Fresh','Brackish','Saline'], n),
        'Temperature_C': np.random.normal(20, 5, n),
        'Risk_Score': np.random.uniform(0, 100, n),
        'Risk_Level': np.random.choice(['Low','Medium','High','Critical'], n, p=[0.3,0.35,0.25,0.1]),
        'Risk_Type': np.random.choice(['Type_A','Type_B','Type_C'], n, p=[0.5,0.3,0.2]),
        'Location': np.random.choice(['Urban','Rural','Industrial','Coastal'], n),
        'Shape': np.random.choice(['Fiber','Fragment','Sphere','Film'], n),
        'Industrial_Activity': np.random.choice(['Low','Moderate','High'], n),
        'Population_Density': np.random.choice(['Low','Medium','High'], n),
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
            
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload dataset (CSV/Excel)", 
                    type=['csv','xlsx','xls'],
                    help="Upload your microplastic risk dataset"
                )
                if uploaded_file:
                    data = load_dataset(uploaded_file)
                    if data is not None:
                        st.success(f"✅ Dataset loaded successfully! Shape: {data.shape[0]} rows × {data.shape[1]} columns")
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🎲 Generate Sample Dataset", type="primary", use_container_width=True):
                    st.session_state.data = generate_sample_data()
                    st.success("✅ Sample dataset generated with realistic outliers!")
                    st.rerun()
            
            if st.session_state.data is not None:
                df = st.session_state.data
                
                # Key metrics
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
                
                # Column Information
                st.markdown("#### 📊 Column Information")
                col_info = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': df.dtypes.values,
                    'Missing Values': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)
                
                # Download column info
                csv = col_info.to_csv(index=False)
                st.download_button(
                    label="📥 Download Column Info",
                    data=csv,
                    file_name="column_information.csv",
                    mime="text/csv"
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
                            color = "green"
                            interpretation = "The distribution is roughly symmetric, suggesting balanced risk scores without significant skew."
                        elif skewness > 0.5:
                            shape = "Right-Skewed (Positive Skew)"
                            color = "orange"
                            interpretation = "The distribution has a longer right tail. More samples have higher risk scores, and the mean is likely greater than the median. Consider investigating what causes high risk scores."
                        else:
                            shape = "Left-Skewed (Negative Skew)"
                            color = "red"
                            interpretation = "The distribution has a longer left tail. More samples have lower risk scores, and the mean is likely less than the median. Consider if risk assessment might be underestimating hazards."
                        
                        st.info(f"**Distribution Shape:** {shape}  \n**Interpretation:** {interpretation}")
                        
                        # Normality test
                        from scipy.stats import normaltest
                        statistic, p_value = normaltest(clean_data)
                        if p_value < 0.05:
                            st.warning(f"**Normality Test:** The distribution is NOT normal (p-value: {p_value:.4f}). Consider applying transformations for certain statistical analyses.")
                        else:
                            st.success(f"**Normality Test:** The distribution appears normal (p-value: {p_value:.4f}).")
                        
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
        
        # ==================== HOME TAB 3: MP Count vs Risk Score ====================
        with home_tab3:
            st.markdown("### 🔬 Explore the Relationship Between MP Count and Risk Score")
            st.markdown("""
            **Objective 2:** Investigate the correlation between Microplastic Count per Liter and Risk Score.
            Understanding this relationship is crucial for risk assessment models.
            """)
            
            if st.session_state.data is None: 
                st.warning("⚠️ Upload data first!")
            else:
                df = st.session_state.data
                
                if 'MP_Count_per_L' not in df.columns or 'Risk_Score' not in df.columns:
                    missing_cols = []
                    if 'MP_Count_per_L' not in df.columns:
                        missing_cols.append('MP_Count_per_L')
                    if 'Risk_Score' not in df.columns:
                        missing_cols.append('Risk_Score')
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Convert to numeric
                    df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
                    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                    clean = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
                    
                    if len(clean) == 0:
                        st.error("❌ No valid data after cleaning!")
                    else:
                        # Visualization tabs
                        st1, st2, st3 = st.tabs([
                            "📊 Scatter Plot", 
                            "📈 Trend Analysis", 
                            "📋 Correlation Analysis"
                        ])
                        
                        with st1:
                            st.markdown("#### Scatter Plot: MP Count vs Risk Score")
                            
                            # Color by Risk Level if available
                            color_col = None
                            if 'Risk_Level' in clean.columns:
                                color_col = 'Risk_Level'
                            
                            fig = px.scatter(
                                clean, 
                                x='MP_Count_per_L', 
                                y='Risk_Score',
                                color=color_col,
                                title='MP Count per Liter vs Risk Score',
                                opacity=0.7,
                                trendline=None,
                                labels={
                                    'MP_Count_per_L': 'MP Count per Liter',
                                    'Risk_Score': 'Risk Score'
                                }
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean MP Count", f"{clean['MP_Count_per_L'].mean():.2f}")
                                st.metric("Std MP Count", f"{clean['MP_Count_per_L'].std():.2f}")
                            with col2:
                                st.metric("Mean Risk Score", f"{clean['Risk_Score'].mean():.2f}")
                                st.metric("Std Risk Score", f"{clean['Risk_Score'].std():.2f}")
                        
                        with st2:
                            st.markdown("#### Trend Analysis")
                            
                            # Try different trendlines
                            try:
                                fig = px.scatter(
                                    clean, 
                                    x='MP_Count_per_L', 
                                    y='Risk_Score',
                                    color=color_col,
                                    trendline='ols',
                                    title='MP Count vs Risk Score with Linear Trend',
                                    opacity=0.7
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add LOWESS trendline
                                st.markdown("**Alternative: LOWESS Smoother**")
                                fig2 = px.scatter(
                                    clean, 
                                    x='MP_Count_per_L', 
                                    y='Risk_Score',
                                    color=color_col,
                                    trendline='lowess',
                                    title='MP Count vs Risk Score with LOWESS Trend',
                                    opacity=0.7
                                )
                                fig2.update_layout(height=500)
                                st.plotly_chart(fig2, use_container_width=True)
                                
                            except Exception as e:
                                st.warning(f"Trendline calculation failed: {str(e)}")
                        
                        with st3:
                            st.markdown("#### Correlation Analysis")
                            
                            # Calculate correlations
                            pearson_corr = clean['MP_Count_per_L'].corr(clean['Risk_Score'])
                            spearman_corr = clean['MP_Count_per_L'].corr(clean['Risk_Score'], method='spearman')
                            kendall_corr = clean['MP_Count_per_L'].corr(clean['Risk_Score'], method='kendall')
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Pearson Correlation", f"{pearson_corr:.4f}")
                                if abs(pearson_corr) < 0.3:
                                    st.caption("Weak correlation")
                                elif abs(pearson_corr) < 0.7:
                                    st.caption("Moderate correlation")
                                else:
                                    st.caption("Strong correlation")
                            
                            with col2:
                                st.metric("Spearman Correlation", f"{spearman_corr:.4f}")
                                st.caption("Monotonic relationship")
                            
                            with col3:
                                st.metric("Kendall's Tau", f"{kendall_corr:.4f}")
                                st.caption("Ordinal association")
                            
                            # Interpretation
                            st.markdown("---")
                            st.markdown("#### 📊 Interpretation")
                            
                            if abs(pearson_corr) > 0.5:
                                st.warning("""
                                **Strong Relationship Detected:**
                                - MP Count is strongly correlated with Risk Score
                                - MP Count could be a key predictor for risk assessment
                                - Consider using MP Count as a primary feature in risk models
                                """)
                            else:
                                st.info("""
                                **Moderate/Weak Relationship:**
                                - Other factors may be more important for risk assessment
                                - Consider multivariate analysis to identify key predictors
                                - MP Count alone may not be sufficient for accurate risk prediction
                                """)
                        
                        # Advanced Analysis
                        st.markdown("---")
                        with st.expander("🔬 Advanced Analysis"):
                            st.markdown("#### Regression Analysis")
                            
                            # Simple linear regression
                            from sklearn.linear_model import LinearRegression
                            X = clean[['MP_Count_per_L']].values
                            y = clean['Risk_Score'].values
                            
                            reg = LinearRegression().fit(X, y)
                            r2 = reg.score(X, y)
                            
                            st.metric("R² Score", f"{r2:.4f}")
                            st.write(f"**Equation:** Risk_Score = {reg.intercept_:.4f} + ({reg.coef_[0]:.4f} × MP_Count_per_L)")
                            
                            st.write("""
                            **Interpretation:**
                            - For each unit increase in MP Count, Risk Score changes by the coefficient
                            - R² indicates how well MP Count explains Risk Score variance
                            """)
        
        # ==================== HOME TAB 4: Risk Score by Risk Level ====================
           # ==================== PREPROCESSING TAB 4: Skewness & Transform ====================
        with p4:
            st.markdown("### 📊 Skewness Analysis & Log Transform")
            st.markdown("""
            **Subtask:** Check for skewed numerical columns and apply transformations (log transformation) if necessary.
            
            Calculate and display the skewness of the numerical columns, identify skewed columns, 
            apply log transformation to skewed columns, and recalculate and display the skewness.
            """)
            
            # Select the numerical columns
            numerical_cols = ['MP_Count_per_L', 'Risk_Score', 
                             'Microplastic_Size_mm_midpoint', 'Density_midpoint']
            
            # Check which columns exist
            available_cols = [col for col in numerical_cols if col in df.columns]
            missing_cols = [col for col in numerical_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Some columns not found: {', '.join(missing_cols)}")
            
            if len(available_cols) == 0:
                st.error("❌ None of the specified numerical columns found!")
            else:
                st.markdown(f"**📊 Numerical columns:** {', '.join(available_cols)}")
                
                # Calculate and display skewness before transformation
                st.markdown("---")
                st.markdown("### 📋 Skewness Before Transformation")
                
                skewness_before = df[available_cols].skew()
                
                st.markdown("**Skewness before transformation:**")
                st.dataframe(skewness_before.round(6), use_container_width=True)
                
                # Create a styled dataframe showing skewness status
                skew_before_df = pd.DataFrame({
                    'Column': available_cols,
                    'Skewness': skewness_before.values.round(6),
                    'Status': ['⚠️ SKEWED' if abs(s) > 0.5 else '✅ OK' for s in skewness_before.values]
                })
                
                # Identify skewed columns (using a threshold of 0.5)
                skewed_cols = skewness_before[abs(skewness_before) > 0.5].index.tolist()
                
                st.markdown(f"**Threshold:** |skewness| > 0.5")
                
                if len(skewed_cols) == 0:
                    st.success("✅ No skewed columns found! All columns have |skewness| ≤ 0.5.")
                else:
                    st.warning(f"⚠️ Found {len(skewed_cols)} skewed column(s): **{', '.join(skewed_cols)}**")
                    
                    for col in skewed_cols:
                        direction = "Right (Positive)" if skewness_before[col] > 0 else "Left (Negative)"
                        st.info(f"**{col}:** Skewness = {skewness_before[col]:.6f} → {direction} skew")
                    
                    # Visualize before transformation
                    st.markdown("---")
                    st.markdown("### 📈 Distribution of Skewed Columns (Before)")
                    
                    num_cols = min(len(skewed_cols), 2)
                    fig, axes = plt.subplots(1, num_cols, figsize=(7*num_cols, 5))
                    if num_cols == 1:
                        axes = [axes]
                    
                    for i, col in enumerate(skewed_cols[:num_cols]):
                        clean_data = df[col].dropna()
                        sns.histplot(data=clean_data, kde=True, bins=30,
                                   color='#ffeaa7', edgecolor='white', alpha=0.7, ax=axes[i])
                        axes[i].set_title(f'{col}\nSkewness: {clean_data.skew():.6f}',
                                        fontsize=12, fontweight='bold')
                        axes[i].axvline(clean_data.mean(), color='red', linestyle='--',
                                      label=f'Mean: {clean_data.mean():.2f}')
                        axes[i].axvline(clean_data.median(), color='green', linestyle='--',
                                      label=f'Median: {clean_data.median():.2f}')
                        axes[i].legend()
                    
                    plt.suptitle('Skewed Columns - Before Transformation', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Apply transformation
                    st.markdown("---")
                    st.markdown("### 🔧 Apply Log Transformation")
                    st.markdown("""
                    **Transformation:** `np.log1p(x - min(x))` 
                    
                    This shifts values to handle negatives, then applies log1p to reduce skewness.
                    """)
                    
                    if st.button("📊 Apply Log Transformation to Skewed Columns", type="primary", key="log_btn"):
                        with st.spinner('Applying log transformation...'):
                            df_transformed = df.copy()
                            
                            # Apply log transformation to skewed columns
                            for col in skewed_cols:
                                # Apply log1p after shifting to handle negative values
                                df_transformed[col] = np.log1p(df_transformed[col] - df_transformed[col].min())
                            
                            # Store in session state
                            st.session_state.processed_data = df_transformed
                            st.session_state.skewness_before = skewness_before
                            st.session_state.skewness_after = df_transformed[available_cols].skew()
                            st.session_state.skewed_cols_transformed = skewed_cols
                            
                            # Recalculate and display skewness after transformation
                            st.markdown("---")
                            st.markdown("### 📊 Skewness After Transformation")
                            
                            skewness_after = df_transformed[available_cols].skew()
                            
                            st.markdown("**Skewness after transformation:**")
                            st.dataframe(skewness_after.round(6), use_container_width=True)
                            
                            # Comparison table
                            st.markdown("---")
                            st.markdown("### 📊 Before vs After Comparison")
                            
                            comparison_df = pd.DataFrame({
                                'Column': available_cols,
                                'Skewness Before': skewness_before[available_cols].values.round(6),
                                'Skewness After': skewness_after[available_cols].values.round(6),
                                '|Change|': (abs(skewness_before[available_cols].values) - abs(skewness_after[available_cols].values)).round(6),
                                'Improved': ['✅ YES' if abs(skewness_after[col]) < abs(skewness_before[col]) else '⬜ NO'
                                           for col in available_cols]
                            })
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Summary
                            improved = [col for col in available_cols 
                                      if abs(skewness_after[col]) < abs(skewness_before[col])]
                            st.success(f"✅ Skewness reduced in {len(improved)} column(s): {', '.join(improved)}")
                            
                            # Visual comparison
                            st.markdown("---")
                            st.markdown("### 📈 Visual Comparison (Before vs After)")
                            
                            for col in skewed_cols:
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                                
                                # Before
                                clean_before = df[col].dropna()
                                sns.histplot(data=clean_before, kde=True, bins=30,
                                           color='#ffeaa7', edgecolor='white', alpha=0.7, ax=ax1)
                                ax1.set_title(f'{col} - BEFORE\nSkewness: {clean_before.skew():.6f}',
                                            fontsize=12, fontweight='bold')
                                ax1.set_xlabel(col)
                                ax1.set_ylabel('Frequency')
                                
                                # After
                                clean_after = df_transformed[col].dropna()
                                sns.histplot(data=clean_after, kde=True, bins=30,
                                           color='#55efc4', edgecolor='white', alpha=0.7, ax=ax2)
                                ax2.set_title(f'{col} - AFTER\nSkewness: {clean_after.skew():.6f}',
                                            fontsize=12, fontweight='bold')
                                ax2.set_xlabel(col)
                                ax2.set_ylabel('Frequency')
                                
                                plt.suptitle(f'Log Transformation: {col}', fontsize=14, fontweight='bold')
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            
                            # Download
                            st.markdown("---")
                            csv_transformed = df_transformed[available_cols].to_csv(index=False)
                            st.download_button(
                                label="📥 Download Transformed Data (CSV)",
                                data=csv_transformed,
                                file_name="skewness_transformed_data.csv",
                                mime="text/csv"
                            )
        
        # ==================== HOME TAB 5: Data Quality Check ====================
        with home_tab5:
            st.markdown("### 🔍 Data Quality Check")
            
            if st.session_state.data is None: 
                st.warning("⚠️ Upload data first!")
            else:
                df = st.session_state.data
                
                # Quality metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Missing Values %", f"{missing_pct:.2f}%")
                with col2:
                    st.metric("Duplicate Rows", df.duplicated().sum())
                with col3:
                    st.metric("Numeric Columns", len(df.select_dtypes(include=['float64', 'int64']).columns))
                with col4:
                    st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
                
                st.markdown("---")
                
                # Missing values heatmap
                st.markdown("#### Missing Values Pattern")
                
                if df.isnull().sum().sum() > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
                    ax.set_title('Missing Values Pattern (Yellow = Missing)')
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.success("✅ No missing values found!")
                
                # Data types summary
                st.markdown("#### Data Types Distribution")
                
                dtype_counts = df.dtypes.value_counts()
                fig, ax = plt.subplots(figsize=(8, 4))
                dtype_counts.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
                ax.set_title('Distribution of Data Types')
                ax.set_xlabel('Data Type')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()
                
                # Descriptive statistics
                st.markdown("#### Descriptive Statistics")
                st.dataframe(df.describe(), use_container_width=True)
                
                # Download options
                if st.download_button("📥 Download Quality Report", 
                                     data=df.describe().to_csv(), 
                                     file_name="data_quality_report.csv",
                                     mime="text/csv"):
                    st.success("Report downloaded!")
    
    # ==================== PREPROCESSING PAGE ====================
    elif section == "🔧 Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        
        if st.session_state.data is None: 
            st.warning("⚠️ Please upload data first!")
            return
        
        df = st.session_state.data.copy()
        
        p1, p2, p3, p4, p5 = st.tabs([
            "📏 Feature Scaling", 
            "🔄 Categorical Encoding", 
            "🎯 Outlier Handling", 
            "📊 Skewness & Transform", 
            "📋 Summary"
        ])
        
        # ==================== PREPROCESSING TAB 1: Feature Scaling ====================
        with p1:
            st.markdown("### 📏 Feature Scaling")
            st.markdown("""
            **Subtask:** Apply feature scaling (Standardization) to the numerical columns.
            
            StandardScaler transforms the data to have mean=0 and standard deviation=1.
            This is important for many machine learning algorithms that are sensitive to feature scales.
            """)
            
            # Define the specific numerical columns for scaling
            numerical_cols = ['MP_Count_per_L', 'Risk_Score', 
                             'Microplastic_Size_mm_midpoint', 'Density_midpoint']
            
            # Check which of these columns exist in the dataframe
            available_cols = [col for col in numerical_cols if col in df.columns]
            missing_cols = [col for col in numerical_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Some specified columns not found in dataset: {', '.join(missing_cols)}")
            
            if len(available_cols) == 0:
                st.error("❌ None of the specified numerical columns found in the dataset!")
                st.info(f"Available numerical columns: {', '.join(df.select_dtypes(include=['float64', 'int64']).columns.tolist())}")
            else:
                st.markdown(f"**📊 Columns to scale:** {', '.join(available_cols)}")
                
                # Show data before scaling
                st.markdown("---")
                st.markdown("### 📋 Data Before Scaling")
                st.markdown("**First 5 rows of original numerical data:**")
                st.dataframe(df[available_cols].head(), use_container_width=True)
                
                # Show statistics before scaling
                st.markdown("**Descriptive Statistics (Before Scaling):**")
                st.dataframe(df[available_cols].describe(), use_container_width=True)
                
                st.markdown("---")
                
                # Apply StandardScaler
                st.markdown("### 🔧 Apply StandardScaler")
                st.markdown("""
                **StandardScaler Formula:** 
                - z = (x - μ) / σ
                - where μ is the mean and σ is the standard deviation
                
                After scaling:
                - Mean ≈ 0
                - Standard Deviation ≈ 1
                """)
                
                if st.button("📏 Apply StandardScaler", type="primary", key="scale_btn"):
                    with st.spinner('Applying StandardScaler...'):
                        # Instantiate the scaler
                        scaler = StandardScaler()
                        
                        # Fit and transform the numerical data
                        df[available_cols] = scaler.fit_transform(df[available_cols])
                        
                        # Store in session state
                        st.session_state.processed_data = df
                        st.session_state.scaler = scaler
                        st.session_state.scaled_columns = available_cols
                        
                        st.success(f"✅ Successfully scaled {len(available_cols)} columns!")
                        
                        # Display the first few rows of the scaled numerical data
                        st.markdown("---")
                        st.markdown("### 📊 Scaled Data Results")
                        
                        st.markdown("**First 5 rows of scaled numerical data:**")
                        st.dataframe(df[available_cols].head(), use_container_width=True)
                        
                        # Show statistics after scaling
                        st.markdown("**Descriptive Statistics (After Scaling):**")
                        st.dataframe(df[available_cols].describe(), use_container_width=True)
                        
                        # Show scaling parameters
                        st.markdown("**Scaling Parameters:**")
                        scaling_params = pd.DataFrame({
                            'Column': available_cols,
                            'Mean (μ)': scaler.mean_.round(6),
                            'Std (σ)': scaler.scale_.round(6)
                        })
                        st.dataframe(scaling_params, use_container_width=True, hide_index=True)
                        
                        # Visualize before and after
                        st.markdown("---")
                        st.markdown("### 📈 Visual Comparison")
                        
                        # Get original data for comparison
                        original_df = st.session_state.data.copy()
                        
                        for col in available_cols[:2]:  # Show first 2 columns
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # Before
                            original_clean = original_df[col].dropna()
                            sns.histplot(data=original_clean, kde=True, bins=30, 
                                       color='#3498db', edgecolor='white', alpha=0.7, ax=ax1)
                            ax1.set_title(f'{col} - Before Scaling\nMean: {original_clean.mean():.4f}, Std: {original_clean.std():.4f}', 
                                        fontsize=12)
                            ax1.set_xlabel(col)
                            ax1.set_ylabel('Frequency')
                            
                            # After
                            scaled_clean = df[col].dropna()
                            sns.histplot(data=scaled_clean, kde=True, bins=30, 
                                       color='#2ecc71', edgecolor='white', alpha=0.7, ax=ax2)
                            ax2.set_title(f'{col} - After Scaling\nMean: {scaled_clean.mean():.4f}, Std: {scaled_clean.std():.4f}', 
                                        fontsize=12)
                            ax2.set_xlabel(col)
                            ax2.set_ylabel('Frequency')
                            
                            plt.suptitle(f'Feature Scaling Comparison: {col}', fontsize=14, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        # Download scaled data
                        st.markdown("---")
                        csv_scaled = df[available_cols].to_csv(index=False)
                        st.download_button(
                            label="📥 Download Scaled Data (CSV)",
                            data=csv_scaled,
                            file_name="scaled_numerical_data.csv",
                            mime="text/csv"
                        )
        
        # ==================== PREPROCESSING TAB 2: Categorical Encoding ====================
        with p2:
            st.markdown("### 🔄 Categorical Encoding")
            st.markdown("""
            **Subtask:** Encode the categorical columns using one-hot encoding.
            
            One-hot encoding creates binary columns for each category, allowing machine learning 
            algorithms to work with categorical data effectively.
            """)
            
            # Identify categorical columns as specified
            categorical_cols = ['Location', 'Shape', 'Polymer_Type', 'pH', 'Salinity', 
                              'Industrial_Activity', 'Population_Density', 'Risk_Type', 
                              'Risk_Level', 'Author', 'Source']
            
            # Check which of these columns exist in the dataframe
            available_cats = [col for col in categorical_cols if col in df.columns]
            missing_cats = [col for col in categorical_cols if col not in df.columns]
            
            if missing_cats:
                st.warning(f"⚠️ Some specified columns not found in dataset: {', '.join(missing_cats)}")
            
            if len(available_cats) == 0:
                st.error("❌ None of the specified categorical columns found in the dataset!")
                
                # Show actual categorical columns in the dataset
                actual_cats = df.select_dtypes(include=['object']).columns.tolist()
                st.info(f"Available categorical columns in dataset: {', '.join(actual_cats) if actual_cats else 'None found'}")
            else:
                st.markdown(f"**📊 Categorical columns to encode ({len(available_cats)}):**")
                
                # Display the columns in a nice format
                cols_per_row = 3
                for i in range(0, len(available_cats), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(available_cats[i:i+cols_per_row]):
                        with cols[j]:
                            unique_vals = df[col_name].nunique()
                            st.info(f"**{col_name}**\n{unique_vals} unique values")
                
                # Show original data before encoding
                st.markdown("---")
                st.markdown("### 📋 Data Before Encoding")
                st.markdown("**First 5 rows of original categorical data:**")
                st.dataframe(df[available_cats].head(), use_container_width=True)
                
                # Show value counts for each categorical column
                with st.expander("🔍 View Value Counts for Each Categorical Column"):
                    for col in available_cats:
                        st.markdown(f"**{col}:**")
                        st.dataframe(df[col].value_counts().reset_index().rename(
                            columns={'index': 'Value', col: 'Count'}).head(10), 
                            use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Apply One-Hot Encoding
                st.markdown("### 🔧 Apply One-Hot Encoding")
                st.markdown("""
                **One-Hot Encoding Process:**
                - Each category value becomes a new binary column (0 or 1)
                - `drop_first=True` removes the first category to avoid multicollinearity
                - Original categorical columns are removed and replaced with encoded columns
                """)
                
                if st.button("🔄 Apply One-Hot Encoding", type="primary", key="encode_btn"):
                    with st.spinner('Applying one-hot encoding...'):
                        # Apply one-hot encoding
                        df_encoded = pd.get_dummies(df, columns=available_cats, drop_first=True)
                        
                        # Identify new columns created
                        new_cols = [c for c in df_encoded.columns if c not in df.columns or c in available_cats]
                        original_shape = df.shape
                        encoded_shape = df_encoded.shape
                        
                        # Store in session state
                        st.session_state.processed_data = df_encoded
                        st.session_state.encoded_data = df_encoded
                        st.session_state.encoded_shape = encoded_shape
                        st.session_state.encoded_new_cols = new_cols
                        st.session_state.encoded_original_cols = available_cats
                        
                        st.success(f"✅ One-hot encoding applied successfully!")
                        
                        # Display encoding summary
                        st.markdown("---")
                        st.markdown("### 📊 Encoding Results")
                        
                        # Show shape comparison
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Shape", f"{original_shape[0]} × {original_shape[1]}")
                        with col2:
                            st.metric("Encoded Shape", f"{encoded_shape[0]} × {encoded_shape[1]}")
                        with col3:
                            st.metric("New Columns Added", encoded_shape[1] - original_shape[1])
                        
                        st.markdown("---")
                        
                        # Display the first few rows of the DataFrame with encoded categorical variables
                        st.markdown("### First 5 rows of the DataFrame after one-hot encoding:")
                        st.dataframe(df_encoded.head(), use_container_width=True)
                        
                        # Display the shape of the resulting DataFrame
                        st.markdown("### Shape of the DataFrame after one-hot encoding:")
                        st.info(f"**Rows:** {encoded_shape[0]:,} | **Columns:** {encoded_shape[1]:,}")
                        
                        st.markdown("---")
                        
                        # Show new columns created
                        st.markdown("### 📋 New Encoded Columns Created")
                        
                        # Group new columns by original category
                        for cat in available_cats:
                            related_cols = [c for c in new_cols if c.startswith(cat + '_')]
                            if related_cols:
                                with st.expander(f"📌 {cat} → {len(related_cols)} new columns"):
                                    st.write(related_cols)
                        
                        # Show column types after encoding
                        st.markdown("---")
                        st.markdown("### 📊 Column Types After Encoding")
                        
                        dtype_counts = df_encoded.dtypes.value_counts()
                        dtype_df = pd.DataFrame({
                            'Data Type': dtype_counts.index.astype(str),
                            'Count': dtype_counts.values
                        })
                        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
                        
                        # Download encoded data
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_encoded = df_encoded.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Encoded Data (CSV)",
                                data=csv_encoded,
                                file_name="encoded_microplastic_data.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col2:
                            # Download encoding summary
                            encoding_summary = pd.DataFrame({
                                'Original Column': available_cats,
                                'New Columns Created': [len([c for c in new_cols if c.startswith(cat + '_')]) for cat in available_cats]
                            })
                            csv_summary = encoding_summary.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Encoding Summary (CSV)",
                                data=csv_summary,
                                file_name="encoding_summary.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
        
        # ==================== PREPROCESSING TAB 3: Outlier Handling ====================
        with p3:
            st.markdown("### 🎯 Outlier Handling")
            st.markdown("""
            **Subtask:** Identify and handle outliers in numerical columns using the IQR method.
            
            <div class='outlier-box outlier-before'>
            <strong>📌 Method:</strong> Interquartile Range (IQR)<br>
            <strong>Formula:</strong> Lower bound = Q1 - 1.5×IQR, Upper bound = Q3 + 1.5×IQR<br>
            <strong>Strategy:</strong> Cap outliers at the upper and lower bounds
            </div>
            """, unsafe_allow_html=True)
            
            # Define the specific numerical columns for outlier handling
            specified_numerical_cols = ['MP_Count_per_L', 'Risk_Score', 
                                        'Microplastic_Size_mm_midpoint', 'Density_midpoint']
            
            # Check which of these columns exist in the dataframe
            available_cols = [col for col in specified_numerical_cols if col in df.columns]
            missing_cols = [col for col in specified_numerical_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Some specified columns not found in dataset: {', '.join(missing_cols)}")
            
            if len(available_cols) == 0:
                st.error("❌ None of the specified numerical columns found in the dataset!")
                st.info(f"Available numerical columns: {', '.join(df.select_dtypes(include=['float64', 'int64']).columns.tolist())}")
            else:
                st.markdown(f"**📊 Columns to process:** {', '.join(available_cols)}")
                
                # Step 1: Show current state with outliers
                st.markdown("---")
                st.markdown("### 📋 Step 1: Current Data with Outliers")
                
                # Display descriptive statistics before outlier handling
                st.markdown("#### Descriptive Statistics (Before Outlier Handling)")
                st.dataframe(df[available_cols].describe(), use_container_width=True)
                
                # Detect and show outliers
                outlier_info = detect_outliers_detailed(df, available_cols)
                
                # Create outlier summary table
                outlier_summary = []
                for col, info in outlier_info.items():
                    outlier_summary.append({
                        'Column': col,
                        'Q1': f"{info['Q1']:.4f}",
                        'Q3': f"{info['Q3']:.4f}",
                        'IQR': f"{info['IQR']:.4f}",
                        'Lower Bound': f"{info['lower_bound']:.4f}",
                        'Upper Bound': f"{info['upper_bound']:.4f}",
                        'Outliers Found': info['outlier_count'],
                        'Outlier %': f"{info['outlier_percentage']:.1f}%",
                        'Below Bound': info['outliers_below'],
                        'Above Bound': info['outliers_above']
                    })
                
                st.markdown("#### Outlier Detection Summary")
                outlier_summary_df = pd.DataFrame(outlier_summary)
                st.dataframe(outlier_summary_df, use_container_width=True, hide_index=True)
                
                # Visualize outliers before handling
                st.markdown("#### Box Plots with Outliers (Before)")
                fig, axes = plt.subplots(1, len(available_cols), figsize=(5*len(available_cols), 5))
                if len(available_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(available_cols):
                    clean_data = df[col].dropna()
                    sns.boxplot(data=clean_data, color='#ffeaa7', width=0.3, ax=axes[i])
                    axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
                    axes[i].set_ylabel(col, fontsize=10)
                    axes[i].tick_params(axis='x', rotation=45)
                
                plt.suptitle('Before Outlier Handling - Box Plots', fontsize=16, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Step 2: Handle outliers
                st.markdown("---")
                st.markdown("### 🔧 Step 2: Handle Outliers")
                
                st.markdown("""
                **Approach:** Using the IQR method to identify and cap outliers:
                - Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR
                - Define bounds: Lower = Q1 - 1.5×IQR, Upper = Q3 + 1.5×IQR
                - Cap values below lower bound and above upper bound
                """)
                
                if st.button("🔧 Apply Outlier Capping", type="primary", key="outlier_btn"):
                    with st.spinner('Capping outliers using IQR method...'):
                        # Apply IQR capping
                        df_capped, stats_before, stats_after, outlier_bounds, outlier_counts = cap_outliers_iqr(df, available_cols)
                        
                        # Store in session state
                        st.session_state.processed_data = df_capped
                        st.session_state.outlier_columns_processed = available_cols
                        st.session_state.outlier_stats_before = stats_before
                        st.session_state.outlier_stats_after = stats_after
                        st.session_state.outlier_bounds = outlier_bounds
                        st.session_state.outlier_counts = outlier_counts
                        
                        st.success(f"✅ Successfully capped outliers in {len(available_cols)} columns!")
                        
                        # Show IQR bounds used
                        st.markdown("#### IQR Bounds Applied")
                        bounds_data = []
                        for col, bounds in outlier_bounds.items():
                            bounds_data.append({
                                'Column': col,
                                'Q1': f"{bounds['Q1']:.4f}",
                                'Q3': f"{bounds['Q3']:.4f}",
                                'IQR': f"{bounds['IQR']:.4f}",
                                'Lower Bound': f"{bounds['lower_bound']:.4f}",
                                'Upper Bound': f"{bounds['upper_bound']:.4f}"
                            })
                        st.dataframe(pd.DataFrame(bounds_data), use_container_width=True, hide_index=True)
                        
                        # Show outlier counts
                        st.markdown("#### Outliers Capped")
                        counts_data = []
                        for col, counts in outlier_counts.items():
                            counts_data.append({
                                'Column': col,
                                'Below Bound': counts['below'],
                                'Above Bound': counts['above'],
                                'Total Capped': counts['total'],
                                '% of Data': f"{counts['percentage']:.2f}%"
                            })
                        st.dataframe(pd.DataFrame(counts_data), use_container_width=True, hide_index=True)
                        
                        # Step 3: Show results after outlier handling
                        st.markdown("---")
                        st.markdown("### 📊 Step 3: Results After Outlier Handling")
                        
                        # Display descriptive statistics after outlier handling
                        st.markdown("#### Descriptive Statistics (After Outlier Handling)")
                        st.dataframe(df_capped[available_cols].describe(), use_container_width=True)
                        
                        # Before vs After comparison table
                        st.markdown("#### Before vs After Comparison")
                        comparison_table = create_outlier_summary_table(stats_before, stats_after, available_cols)
                        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
                        
                        # Visual comparison
                        st.markdown("#### Visual Comparison (Before vs After)")
                        
                        for col in available_cols:
                            st.markdown(f"**{col}**")
                            fig = create_outlier_boxplot_comparison(df, df_capped, col)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning(f"Could not create comparison plot for {col}")
                        
                        # Side-by-side distribution comparison
                        st.markdown("#### Distribution Comparison")
                        
                        for col in available_cols:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # Before
                            clean_before = df[col].dropna()
                            sns.histplot(data=clean_before, kde=True, bins=30, 
                                       color='#ffeaa7', edgecolor='white', alpha=0.7, ax=ax1)
                            ax1.set_title(f'{col} - Before', fontsize=14, fontweight='bold')
                            ax1.set_xlabel(col)
                            ax1.set_ylabel('Frequency')
                            ax1.axvline(clean_before.mean(), color='red', linestyle='--', label=f'Mean: {clean_before.mean():.2f}')
                            ax1.legend()
                            
                            # After
                            clean_after = df_capped[col].dropna()
                            sns.histplot(data=clean_after, kde=True, bins=30, 
                                       color='#55efc4', edgecolor='white', alpha=0.7, ax=ax2)
                            ax2.set_title(f'{col} - After', fontsize=14, fontweight='bold')
                            ax2.set_xlabel(col)
                            ax2.set_ylabel('Frequency')
                            ax2.axvline(clean_after.mean(), color='red', linestyle='--', label=f'Mean: {clean_after.mean():.2f}')
                            ax2.legend()
                            
                            plt.suptitle(f'Distribution Comparison: {col}', fontsize=16, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        # Summary of changes
                        st.markdown("#### 📈 Summary of Changes")
                        
                        summary_changes = []
                        for col in available_cols:
                            if col in stats_before and col in stats_after:
                                mean_change = stats_after[col]['mean'] - stats_before[col]['mean']
                                std_change = stats_after[col]['std'] - stats_before[col]['std']
                                range_before = stats_before[col]['max'] - stats_before[col]['min']
                                range_after = stats_after[col]['max'] - stats_after[col]['min']
                                range_reduction = ((range_before - range_after) / range_before) * 100
                                
                                summary_changes.append({
                                    'Column': col,
                                    'Mean Change': f"{mean_change:+.4f}",
                                    'Std Reduction': f"{-std_change:.4f}" if std_change < 0 else f"{std_change:.4f}",
                                    'Range Reduction %': f"{range_reduction:.1f}%",
                                    'Outliers Removed': f"{outlier_counts[col]['total']}"
                                })
                        
                        st.dataframe(pd.DataFrame(summary_changes), use_container_width=True, hide_index=True)
                        
                        # Download options
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            csv_before = df[available_cols].describe().to_csv()
                            st.download_button(
                                "📥 Download Stats (Before)",
                                data=csv_before,
                                file_name="outlier_stats_before.csv",
                                mime="text/csv"
                            )
                        with col2:
                            csv_after = df_capped[available_cols].describe().to_csv()
                            st.download_button(
                                "📥 Download Stats (After)",
                                data=csv_after,
                                file_name="outlier_stats_after.csv",
                                mime="text/csv"
                            )
                        with col3:
                            csv_comparison = comparison_table.to_csv(index=False)
                            st.download_button(
                                "📥 Download Comparison",
                                data=csv_comparison,
                                file_name="outlier_comparison.csv",
                                mime="text/csv"
                            )
                    
                    # Store the processed dataframe for subsequent preprocessing steps
                    st.session_state.processed_data = df_capped
        
        # ==================== PREPROCESSING TAB 4: Skewness & Transform ====================
        with p4:
            st.markdown("### 📊 Skewness Analysis & Log Transform")
            st.markdown("Analyze and reduce skewness in numerical features")
            
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            numerical_cols = [c for c in numerical_cols if 'ID' not in c and 'Sample' not in c]
            
            if len(numerical_cols) == 0:
                st.warning("No numerical columns for skewness analysis")
            else:
                # Skewness analysis
                skew_info = analyze_skewness(df, numerical_cols)
                st.markdown("**Skewness Analysis:**")
                st.dataframe(skew_info, use_container_width=True, hide_index=True)
                
                # Identify skewed columns
                skewed_cols = skew_info[skew_info['Skewed'] == 'Yes']['Column'].tolist()
                
                if len(skewed_cols) == 0:
                    st.success("✅ No highly skewed columns found!")
                else:
                    st.warning(f"Found {len(skewed_cols)} skewed columns")
                    
                    selected_skewed = st.multiselect(
                        "Select columns for log transform:",
                        skewed_cols,
                        default=skewed_cols[:5] if len(skewed_cols) > 5 else skewed_cols,
                        key="skew_cols"
                    )
                    
                    if selected_skewed and st.button("Apply Log Transform", key="log_btn"):
                        df_transformed = apply_log_transform(df, selected_skewed)
                        st.session_state.processed_data = df_transformed
                        
                        # Show comparison
                        skew_before = df[selected_skewed].skew().values
                        skew_after = df_transformed[selected_skewed].skew().values
                        
                        skew_comparison = pd.DataFrame({
                            'Column': selected_skewed,
                            'Skewness Before': skew_before,
                            'Skewness After': skew_after,
                            'Reduction': skew_before - skew_after
                        }).round(4)
                        
                        st.markdown("**Skewness Comparison:**")
                        st.dataframe(skew_comparison, use_container_width=True, hide_index=True)
                        
                        # Visualize transformation
                        for col in selected_skewed[:3]:  # Show first 3
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            sns.histplot(df[col].dropna(), kde=True, bins=30, 
                                       color='#ffeaa7', ax=ax1)
                            ax1.set_title(f'{col} - Before Log Transform\nSkewness: {df[col].skew():.4f}')
                            
                            sns.histplot(df_transformed[col].dropna(), kde=True, bins=30, 
                                       color='#55efc4', ax=ax2)
                            ax2.set_title(f'{col} - After Log Transform\nSkewness: {df_transformed[col].skew():.4f}')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        st.success("✅ Log transform applied!")
        
        # ==================== PREPROCESSING TAB 5: Summary ====================
        with p5:
            st.markdown("### 📋 Preprocessing Summary")
            
            summary_items = []
            
            if st.session_state.get('scaled_columns') is not None:
                summary_items.append({
                    'Step': 'Feature Scaling (StandardScaler)',
                    'Status': '✅',
                    'Details': f"Scaled {len(st.session_state.scaled_columns)} columns: {', '.join(st.session_state.scaled_columns)}"
                })
            
            if st.session_state.get('encoded_data') is not None:
                summary_items.append({
                    'Step': 'Categorical Encoding (One-Hot)',
                    'Status': '✅',
                    'Details': f"Encoded {len(st.session_state.encoded_original_cols)} columns. Original shape: {st.session_state.data.shape}, Encoded shape: {st.session_state.encoded_shape}"
                })
            
            if len(st.session_state.get('outlier_columns_processed', [])) > 0:
                total_outliers = sum(st.session_state.get('outlier_counts', {}).get(col, {}).get('total', 0) 
                                   for col in st.session_state.outlier_columns_processed)
                summary_items.append({
                    'Step': 'Outlier Handling (IQR Capping)',
                    'Status': '✅',
                    'Details': f"Processed {len(st.session_state.outlier_columns_processed)} columns, capped {total_outliers} outliers"
                })
            
            if summary_items:
                st.markdown("#### Completed Preprocessing Steps")
                st.dataframe(pd.DataFrame(summary_items), use_container_width=True, hide_index=True)
                
                current_data = st.session_state.processed_data if st.session_state.processed_data is not None else df
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Data Shape", f"{current_data.shape[0]:,} rows")
                with col2:
                    st.metric("Features", f"{current_data.shape[1]} columns")
                
                # Download processed data
                csv = current_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Processed Data",
                    data=csv,
                    file_name="processed_microplastic_data.csv",
                    mime="text/csv"
                )
            else:
                st.info("No preprocessing steps applied yet. Use the tabs above to preprocess your data.")
    
    # ==================== FEATURE SELECTION PAGE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        st.markdown("### 🎯 Target Variable Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox(
                "Select Target Variable:", 
                df.columns.tolist(),
                index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0
            )
        with col2:
            if df[target].dtype == 'object' or df[target].nunique() < 10:
                model_type = "Classification"
                st.info(f"**Model Type:** {model_type}")
                st.metric("Unique Classes", df[target].nunique())
            else:
                model_type = st.selectbox("Model Type:", ["Classification", "Regression"])
        
        st.markdown("---")
        
        # Feature Selection Methods
        st.markdown("### 📚 Feature Selection Methods")
        
        method_tabs = st.tabs(["📋 Overview", "🔍 Filter Methods", "🔄 Wrapper Methods", "🌲 Embedded Methods"])
        
        with method_tabs[0]:
            st.markdown("""
            #### Feature Selection Methods Overview
            
            **1. Filter Methods (Selected)**
            - Mutual Information: Measures dependency between features and target
            - Chi-Squared Test: Tests independence between categorical variables
            - Fast and model-independent
            
            **2. Wrapper Methods**
            - Recursive Feature Elimination (RFE)
            - Forward/Backward Selection
            - More accurate but computationally expensive
            
            **3. Embedded Methods (Selected)**
            - Random Forest Feature Importance
            - Lasso Regularization
            - Feature selection during model training
            """)
        
        with method_tabs[1]:
            st.markdown("#### 🔍 Filter Methods")
            st.success("✅ Selected: Mutual Information & Chi-Squared Test")
            st.info("These methods are well-suited for mixed data types and provide interpretable results.")
        
        with method_tabs[2]:
            st.markdown("#### 🔄 Wrapper Methods")
            st.warning("⚠️ Not selected due to computational cost with many features after encoding.")
        
        with method_tabs[3]:
            st.markdown("#### 🌲 Embedded Methods")
            st.success("✅ Selected: Random Forest Importance")
            st.info("Provides reliable feature importance for both classification and regression tasks.")
        
        st.markdown("---")
        
        # Feature Selection Execution
        st.markdown("### 🎯 Apply Feature Selection")
        
        numerical_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target in numerical_cols:
            numerical_cols.remove(target)
        
        if len(numerical_cols) == 0:
            st.error("No numerical features available for selection!")
        else:
            st.markdown(f"**Available numerical features:** {len(numerical_cols)}")
            
            if st.button("🚀 Calculate Feature Importance", type="primary", use_container_width=True):
                with st.spinner('Calculating feature importance...'):
                    try:
                        X = df[numerical_cols].copy()
                        y = df[target].copy()
                        
                        # Clean data
                        mask = y.notna()
                        X = X[mask]
                        y = y[mask]
                        X = X.fillna(X.median())
                        
                        # Encode target if needed
                        if y.dtype == 'object':
                            y = LabelEncoder().fit_transform(y)
                        elif model_type == "Classification":
                            y = pd.qcut(y, q=4, labels=False, duplicates='drop')
                        
                        # Calculate scores
                        mi_scores = calculate_mutual_info(X, y)
                        chi2_scores = calculate_chi2(X, y)
                        rf_scores = calculate_rf_importance(X, y)
                        
                        # Store top 10 features
                        st.session_state.selected_features = rf_scores.head(10).index.tolist()
                        st.session_state.mutual_info = mi_scores
                        st.session_state.chi2_scores = chi2_scores
                        st.session_state.feature_importance = rf_scores
                        
                        # Display results
                        ft1, ft2, ft3 = st.tabs([
                            "📊 Mutual Information", 
                            "🔢 Chi-squared", 
                            "🌲 Random Forest"
                        ])
                        
                        with ft1:
                            st.markdown("**Top 20 Features - Mutual Information:**")
                            st.dataframe(
                                pd.DataFrame({
                                    'Feature': mi_scores.head(20).index,
                                    'Score': mi_scores.head(20).values.round(4)
                                })
                            )
                            
                            fig = px.bar(
                                x=mi_scores.head(20).index, 
                                y=mi_scores.head(20).values,
                                title='Top 20 Features - Mutual Information'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with ft2:
                            st.markdown("**Top 20 Features - Chi-squared:**")
                            st.dataframe(
                                pd.DataFrame({
                                    'Feature': chi2_scores.head(20).index,
                                    'Score': chi2_scores.head(20).values.round(4)
                                })
                            )
                        
                        with ft3:
                            st.markdown("**Top 20 Features - Random Forest:**")
                            st.dataframe(
                                pd.DataFrame({
                                    'Feature': rf_scores.head(20).index,
                                    'Importance': rf_scores.head(20).values.round(4)
                                })
                            )
                            
                            fig = px.bar(
                                x=rf_scores.head(20).index,
                                y=rf_scores.head(20).values,
                                title='Top 20 Features - Random Forest Importance'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.success(f"✅ Feature selection completed! Top 10 features selected for modeling.")
                        
                    except Exception as e:
                        st.error(f"Feature selection failed: {str(e)}")
    
    # ==================== MODELING PAGE ====================
    elif section == "🤖 Modeling":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target = st.selectbox("Target Variable", df.columns.tolist(), key='train_target')
        with col2:
            all_features = [c for c in df.columns if c != target]
            default_features = st.session_state.get('selected_features', 
                                                   df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:5])
            default_features = [f for f in default_features if f in all_features]
            features = st.multiselect("Features", all_features, default=default_features)
        with col3:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            use_smote = st.checkbox("Use SMOTE", value=True)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(features) == 0: 
                st.error("Select at least one feature!")
            else:
                with st.spinner('Training models...'):
                    try:
                        X = df[features].select_dtypes(include=['float64', 'int64', 'int32'])
                        y = df[target]
                        mask = y.notna()
                        X = X[mask]
                        y = y[mask]
                        if y.dtype == 'object':
                            y = LabelEncoder().fit_transform(y)
                        X = X.fillna(X.median())
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, 
                            stratify=y if len(np.unique(y)) > 1 else None
                        )
                        
                        if use_smote:
                            class_counts = pd.Series(y_train).value_counts()
                            if class_counts.min() >= 2:
                                try:
                                    smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min()-1))
                                    X_train, y_train = smote.fit_resample(X_train, y_train)
                                except Exception as e:
                                    st.warning(f"SMOTE failed: {str(e)}")
                        
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
                        
                        if models:
                            st.session_state.models = models
                            st.session_state.trained = True
                            
                            st.markdown("### 📊 Model Performance")
                            results_list = []
                            for name, model in models.items():
                                y_pred = model.predict(X_test)
                                results_list.append({
                                    'Model': name,
                                    'Accuracy': f"{accuracy_score(y_test, y_pred):.4f}",
                                    'Precision': f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
                                    'Recall': f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
                                    'F1 Score': f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
                                })
                            
                            results_df = pd.DataFrame(results_list)
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                            
                            best_model = results_df.iloc[results_df['F1 Score'].astype(float).idxmax()]
                            st.success(f"🏆 Best Model: **{best_model['Model']}** with F1 Score: {best_model['F1 Score']}")
                            
                            fig = px.bar(
                                results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                                x='Model', y='Score', color='Metric',
                                barmode='group', height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if 'Random Forest' in models:
                                st.markdown("### 🌲 Random Forest Feature Importance")
                                importances = models['Random Forest'].feature_importances_
                                feat_imp = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(feat_imp, x='Feature', y='Importance')
                                st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Model training failed: {str(e)}")
    
    # ==================== CROSS VALIDATION PAGE ====================
    elif section == "📊 Cross Validation & Evaluation":
        st.markdown('<p class="section-header">📊 Cross Validation & Model Evaluation</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        et1, et2, et3, et4 = st.tabs([
            "📊 Evaluate Models", 
            "📊 Compare Targets", 
            "🔄 Cross Validation", 
            "📋 Pipeline Summary"
        ])
        
        with et1:
            st.markdown("### 📊 Evaluate Models on Risk_Type")
            if 'Risk_Type' not in df.columns:
                st.error("'Risk_Type' column not found!")
            elif st.button("🚀 Evaluate Models", type="primary", key="eval"):
                with st.spinner('Evaluating models...'):
                    results, info = train_and_evaluate_detailed(df, 'Risk_Type')
                    st.session_state.evaluation_ran = True
                if results:
                    results_list = [{'Model': name, **metrics} for name, metrics in results.items()]
                    results_df = pd.DataFrame(results_list)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    fig = px.bar(
                        results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                        x='Model', y='Score', color='Metric',
                        barmode='group', height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with et2:
            st.markdown("### 📊 Compare Models on Both Risk_Type and Risk_Level")
            if st.button("🚀 Compare Targets", type="primary", key="cmp"):
                all_results = {}
                for target in ['Risk_Type', 'Risk_Level']:
                    if target in df.columns:
                        with st.spinner(f'Processing {target}...'):
                            results, _ = train_and_evaluate_detailed(df, target)
                            all_results[target] = results
                st.session_state.comparison_ran = True
                for target, results in all_results.items():
                    st.markdown(f"### Target: {target}")
                    if results:
                        results_list = [{'Model': name, 'Accuracy': metrics['accuracy'], 'F1': metrics['f1_score']} 
                                      for name, metrics in results.items()]
                        st.dataframe(pd.DataFrame(results_list), use_container_width=True, hide_index=True)
                    st.markdown("---")
        
        with et3:
            st.markdown("### 🔄 Cross Validation")
            target = st.selectbox("Target for CV", df.columns.tolist(),
                                 index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0,
                                 key="cv_target")
            numerical_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
            if target in numerical_cols:
                numerical_cols.remove(target)
            folds = st.slider("Number of CV Folds", 3, 10, 5, key="cv_folds")
            
            if st.button("🔄 Run Cross Validation", type="primary", key="cv"):
                with st.spinner('Running cross validation...'):
                    try:
                        X = df[numerical_cols].copy()
                        y = df[target].copy()
                        mask = y.notna()
                        X = X[mask]
                        y = y[mask]
                        if y.dtype == 'object':
                            y = LabelEncoder().fit_transform(y)
                        X = X.fillna(X.median())
                        
                        models = {
                            'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1),
                            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
                            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
                        }
                        
                        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                        cv_results = []
                        
                        for name, model in models.items():
                            try:
                                acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                                f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                                cv_results.append({
                                    'Model': name,
                                    'Mean Accuracy': f"{acc_scores.mean():.4f}",
                                    'Std Accuracy': f"{acc_scores.std():.4f}",
                                    'Mean F1': f"{f1_scores.mean():.4f}",
                                    'Std F1': f"{f1_scores.std():.4f}"
                                })
                            except Exception as e:
                                st.warning(f"{name} CV failed: {str(e)}")
                        
                        if cv_results:
                            cv_df = pd.DataFrame(cv_results)
                            st.dataframe(cv_df, use_container_width=True, hide_index=True)
                            best_model = cv_df.iloc[cv_df['Mean F1'].astype(float).idxmax()]
                            st.success(f"🏆 Best Model: **{best_model['Model']}** (F1: {best_model['Mean F1']})")
                            st.session_state.cv_ran = True
                    
                    except Exception as e:
                        st.error(f"Cross validation failed: {str(e)}")
        
        with et4:
            st.markdown("### 📋 Complete Pipeline Summary")
            if st.button("🔄 Generate Pipeline Summary", type="primary", key="pipe"):
                pipeline_data = [
                    {'Stage': '1. Data Loading', 'Step': 'Dataset', 
                     'Status': '✅' if st.session_state.data is not None else '❌',
                     'Details': f"{st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} cols" if st.session_state.data is not None else 'No data'},
                    {'Stage': '2. Preprocessing', 'Step': 'Feature Scaling (StandardScaler)',
                     'Status': '✅' if st.session_state.get('scaled_columns') else '⬜',
                     'Details': f"Scaled {len(st.session_state.get('scaled_columns', []))} columns" if st.session_state.get('scaled_columns') else 'Not applied'},
                    {'Stage': '2. Preprocessing', 'Step': 'Categorical Encoding (One-Hot)',
                     'Status': '✅' if st.session_state.get('encoded_data') is not None else '⬜',
                     'Details': f"Encoded {len(st.session_state.get('encoded_original_cols', []))} columns. Shape: {st.session_state.get('encoded_shape')}" if st.session_state.get('encoded_data') is not None else 'Not applied'},
                    {'Stage': '2. Preprocessing', 'Step': 'Outlier Handling (IQR)',
                     'Status': '✅' if len(st.session_state.get('outlier_columns_processed', [])) > 0 else '⬜',
                     'Details': f"Processed {len(st.session_state.get('outlier_columns_processed', []))} columns" if len(st.session_state.get('outlier_columns_processed', [])) > 0 else 'Not applied'},
                    {'Stage': '3. Feature Selection', 'Step': 'Feature Importance',
                     'Status': '✅' if st.session_state.get('feature_importance') is not None else '⬜',
                     'Details': f"Top features: {', '.join(st.session_state.get('selected_features', [])[:5])}..." if st.session_state.get('selected_features') else 'Not computed'},
                    {'Stage': '4. Modeling', 'Step': 'Models Trained',
                     'Status': '✅' if st.session_state.get('trained') else '⬜',
                     'Details': f"{len(st.session_state.get('models', {}))} models" if st.session_state.get('trained') else 'Not trained'},
                    {'Stage': '5. Evaluation', 'Step': 'Model Evaluation',
                     'Status': '✅' if st.session_state.get('evaluation_ran') else '⬜',
                     'Details': 'Completed' if st.session_state.get('evaluation_ran') else 'Not done'},
                    {'Stage': '5. Evaluation', 'Step': 'Cross Validation',
                     'Status': '✅' if st.session_state.get('cv_ran') else '⬜',
                     'Details': 'Completed' if st.session_state.get('cv_ran') else 'Not done'}
                ]
                
                pipeline_df = pd.DataFrame(pipeline_data)
                st.dataframe(pipeline_df, use_container_width=True, height=400)
                
                completed = sum(1 for d in pipeline_data if d['Status'] == '✅')
                progress = int((completed / len(pipeline_data)) * 100)
                st.progress(progress, text=f"Pipeline Progress: {progress}%")
                
                csv = pipeline_df.to_csv(index=False)
                st.download_button("📥 Download Pipeline Report", data=csv, 
                                 file_name="pipeline_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()
