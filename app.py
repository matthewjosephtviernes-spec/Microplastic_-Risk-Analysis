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
        with home_tab4:
            st.markdown("### 📊 Investigate Risk Score Differences by Risk Level")
            st.markdown("""
            **Objective 3:** Analyze how Risk Score varies across different Risk Level categories.
            This helps validate the risk level classification system.
            """)
            
            if st.session_state.data is None: 
                st.warning("⚠️ Upload data first!")
            else:
                df = st.session_state.data
                
                if 'Risk_Score' not in df.columns or 'Risk_Level' not in df.columns:
                    missing_cols = []
                    if 'Risk_Score' not in df.columns:
                        missing_cols.append('Risk_Score')
                    if 'Risk_Level' not in df.columns:
                        missing_cols.append('Risk_Level')
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Convert and clean
                    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                    clean = df.dropna(subset=['Risk_Score', 'Risk_Level'])
                    clean['Risk_Level'] = clean['Risk_Level'].astype(str)
                    
                    if len(clean) == 0:
                        st.error("❌ No valid data after cleaning!")
                    else:
                        # Visualization
                        st.markdown("#### Box Plot: Risk Score by Risk Level")
                        
                        fig = px.box(
                            clean, 
                            x='Risk_Level', 
                            y='Risk_Score', 
                            color='Risk_Level',
                            title='Risk Score Distribution by Risk Level',
                            points='outliers'
                        )
                        fig.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Violin plot
                        st.markdown("#### Violin Plot: Distribution Shape by Risk Level")
                        fig2 = px.violin(
                            clean,
                            x='Risk_Level',
                            y='Risk_Score',
                            color='Risk_Level',
                            title='Distribution Shape by Risk Level',
                            box=True,
                            points='all'
                        )
                        fig2.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Statistics by Risk Level
                        st.markdown("#### 📊 Statistics by Risk Level")
                        
                        stats_by_level = clean.groupby('Risk_Level')['Risk_Score'].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(4)
                        
                        stats_by_level.columns = ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
                        st.dataframe(stats_by_level, use_container_width=True)
                        
                        # ANOVA test
                        from scipy.stats import f_oneway
                        
                        risk_levels = clean['Risk_Level'].unique()
                        if len(risk_levels) >= 2:
                            groups = [clean[clean['Risk_Level'] == level]['Risk_Score'].values 
                                     for level in risk_levels]
                            f_stat, p_value = f_oneway(*groups)
                            
                            st.markdown(f"""
                            #### Statistical Test (ANOVA)
                            - **F-statistic:** {f_stat:.4f}
                            - **P-value:** {p_value:.4f}
                            - **Conclusion:** {'Significant differences exist between risk levels' if p_value < 0.05 else 'No significant differences between risk levels'}
                            """)
        
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
    # ==================== FEATURE SELECTION PAGE ====================
    # ==================== FEATURE SELECTION PAGE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        # ─────────────────────────────────────────────────────────────
        # STEP 1: Understand the Goal
        # ─────────────────────────────────────────────────────────────
        st.markdown("## 🎯 Step 1: Understand the Goal")
        st.markdown("Clarify the target variable for classification/prediction and the type of model.")
        
        col1, col2 = st.columns(2)
        with col1:
            default_idx = df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0
            target = st.selectbox("Select Target Variable:", df.columns.tolist(), index=default_idx)
        with col2:
            model_type = "Classification" if (df[target].dtype == 'object' or df[target].nunique() < 10) else st.selectbox("Model Type:", ["Classification", "Regression"])
        
        st.divider()
        
        # Target Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            st.metric("Variable Type", "Categorical" if df[target].dtype == 'object' else "Numerical")
        with col2: 
            st.metric("Unique Values", df[target].nunique())
        with col3: 
            st.metric("Missing Values", df[target].isnull().sum())
        with col4: 
            st.metric("Total Samples", len(df))
        
        # Target Distribution
        st.divider()
        st.markdown("### Target Distribution")
        
        if df[target].dtype == 'object' or df[target].nunique() < 10:
            target_counts = df[target].value_counts()
            fig = px.bar(x=target_counts.index.astype(str), y=target_counts.values,
                        title=f'Distribution of {target}', color=target_counts.index.astype(str))
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            clean_target = pd.to_numeric(df[target], errors='coerce').dropna()
            if len(clean_target) > 0:
                fig = plot_distribution(df, target, f'Distribution of {target}')
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: Explore Feature Selection Methods
        # ─────────────────────────────────────────────────────────────
        st.markdown("## 📚 Step 2: Explore Feature Selection Methods")
        st.markdown("Discuss and select appropriate feature selection methods based on data type and goal.")
        
        # Method tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Overview", "🔍 Filter Methods", "🔄 Wrapper Methods", "🌲 Embedded Methods", "✅ Selection"
        ])
        
        # ── TAB 1: Overview ──
        with tab1:
            st.markdown("### Comparison of Feature Selection Methods")
            st.markdown("""
            | Criterion | Filter Methods | Wrapper Methods | Embedded Methods |
            |-----------|---------------|-----------------|------------------|
            | Speed | ⚡ Very Fast | 🐢 Slow | ⚡ Fast |
            | Model Independent | ✅ Yes | ❌ No | ⚠️ Partial |
            | Overfitting Risk | ✅ Low | ⚠️ Higher | ✅ Lower |
            | Handles Non-linear | ⚠️ Limited | ✅ Yes | ✅ Yes |
            | Cost | 💰 Low | 💰💰💰 High | 💰💰 Medium |
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**🔍 Filter Methods**\n\nStatistical ranking\n\n✅ Fast\n✅ Model Independent\n✅ Scalable")
            with col2:
                st.warning("**🔄 Wrapper Methods**\n\nModel-based search\n\n✅ Accurate\n❌ Expensive\n❌ Can overfit")
            with col3:
                st.success("**🌲 Embedded Methods**\n\nBuilt-in selection\n\n✅ Balanced\n✅ Feature interactions\n✅ Interpretable")
        
        # ── TAB 2: Filter Methods ──
        with tab2:
            st.markdown("### 🔍 Filter Methods - Selected")
            col1, col2 = st.columns(2)
            with col1:
                st.success("✅ **Mutual Information**")
                st.markdown("- Measures dependency between features and target\n- Works with mixed data types\n- Captures non-linear relationships")
            with col2:
                st.success("✅ **Chi-Squared Test**")
                st.markdown("- Tests independence between categorical variables\n- Provides statistical significance\n- Good for one-hot encoded features")
            st.info("**Why selected:** Fast, model-independent, great for initial screening of 100+ features.")
        
        # ── TAB 3: Wrapper Methods ──
        with tab3:
            st.markdown("### 🔄 Wrapper Methods - Not Selected")
            st.warning("⚠️ Not suitable for our dataset")
            st.markdown("""
            **Reasons for not selecting:**
            - After one-hot encoding: 100+ features (too many)
            - RFE would require 100+ model trainings
            - Forward selection: O(n²) combinations
            - Exhaustive search: impossible (2¹⁰⁰)
            
            **Alternative:** Filter + Embedded = similar accuracy, much faster
            """)
        
        # ── TAB 4: Embedded Methods ──
        with tab4:
            st.markdown("### 🌲 Embedded Methods - Selected")
            st.success("✅ **Random Forest Feature Importance**")
            st.markdown("""
            **How it works:**
            - Trains multiple decision trees
            - Measures impurity reduction (Gini) per feature
            - Features used at top splits = higher importance
            
            **Why selected:**
            - Handles non-linear relationships
            - Captures feature interactions
            - Robust to outliers
            - Clear, interpretable scores
            """)
        
        # ── TAB 5: Final Selection ──
        with tab5:
            st.markdown("### ✅ Final Decision")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div style='background:#0984e3;color:white;padding:20px;border-radius:10px;text-align:center'><h3>🔍 Method 1</h3><h2>Mutual Information</h2><p>Filter Method</p><h4>✅ SELECTED</h4></div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div style='background:#6c5ce7;color:white;padding:20px;border-radius:10px;text-align:center'><h3>🔢 Method 2</h3><h2>Chi-Squared</h2><p>Filter Method</p><h4>✅ SELECTED</h4></div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div style='background:#00b894;color:white;padding:20px;border-radius:10px;text-align:center'><h3>🌲 Method 3</h3><h2>Random Forest</h2><p>Embedded Method</p><h4>✅ SELECTED</h4></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # STEP 3: Quick EDA
        # ─────────────────────────────────────────────────────────────
        st.markdown("## 📈 Step 3: Quick Exploratory Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Risk_Score' in df.columns:
                fig = plot_distribution(df, 'Risk_Score', 'Risk Score')
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                clean = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
                if not clean.empty:
                    fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                   title='MP Count vs Risk Score', opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # STEP 4: Implement Selected Methods (AUTO-RUN)
        # ─────────────────────────────────────────────────────────────
        st.markdown("## 🎯 Step 4: Implement Selected Method(s)")
        st.markdown("Apply Mutual Information, Chi-squared, and Random Forest to rank features.")
        
        # ── Configuration ──
        categorical_cols = ['Location', 'Shape', 'Polymer_Type', 'pH', 'Salinity',
                           'Industrial_Activity', 'Population_Density', 'Risk_Type',
                           'Risk_Level', 'Author', 'Source']
        available_cats = [c for c in categorical_cols if c in df.columns]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            feat_target = st.selectbox("Target for Feature Selection:", df.columns.tolist(),
                                       index=df.columns.tolist().index('Risk_Level') if 'Risk_Level' in df.columns else 0)
        with col2:
            n_top = st.slider("Top N features:", 5, 50, 20)
        with col3:
            st.metric("Categorical Cols", len(available_cats))
        
        st.divider()
        
        # ── Auto-run feature selection ──
        with st.spinner('Running feature selection automatically...'):
            try:
                # --- Prepare Data ---
                df_encoded = pd.get_dummies(df, columns=available_cats, drop_first=True)
                y = df[feat_target].copy()
                mask = y.notna()
                y = y[mask]
                
                original_cols = [c for c in df.columns if c != feat_target]
                ohe_cols = [c for c in df_encoded.columns if c not in original_cols]
                X = df_encoded[ohe_cols].loc[mask].fillna(0)
                
                if y.dtype == 'object':
                    y_encoded = LabelEncoder().fit_transform(y)
                else:
                    y_encoded = y
                
                st.success(f"✅ Data prepared: X = {X.shape[0]} rows × {X.shape[1]} features, y = {len(y_encoded):,} samples")
                
                # --- Method 1: Mutual Information ---
                st.divider()
                st.markdown("### 📊 Method 1: Mutual Information Scores")
                st.markdown("*Measures dependency between each feature and the target variable*")
                
                mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
                mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    mi_df = pd.DataFrame({
                        'Rank': range(1, n_top + 1),
                        'Feature': mi_series.head(n_top).index,
                        'Mutual Information Score': mi_series.head(n_top).values.round(6)
                    })
                    st.dataframe(mi_df, use_container_width=True, hide_index=True)
                with col2:
                    fig = px.bar(mi_df.head(15), x='Feature', y='Mutual Information Score',
                               title='Top 15 - Mutual Information', color='Mutual Information Score',
                               color_continuous_scale='Blues')
                    fig.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- Method 2: Chi-Squared ---
                st.divider()
                st.markdown("### 🔢 Method 2: Chi-Squared Test Scores")
                st.markdown("*Tests statistical independence between features and target*")
                
                chi2_scores, _ = chi2(X, y_encoded)
                chi2_series = pd.Series(chi2_scores, index=X.columns).sort_values(ascending=False)
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    chi2_df = pd.DataFrame({
                        'Rank': range(1, n_top + 1),
                        'Feature': chi2_series.head(n_top).index,
                        'Chi-Squared Score': chi2_series.head(n_top).values.round(4)
                    })
                    st.dataframe(chi2_df, use_container_width=True, hide_index=True)
                with col2:
                    fig = px.bar(chi2_df.head(15), x='Feature', y='Chi-Squared Score',
                               title='Top 15 - Chi-Squared', color='Chi-Squared Score',
                               color_continuous_scale='Reds')
                    fig.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- Method 3: Random Forest ---
                st.divider()
                st.markdown("### 🌲 Method 3: Random Forest Feature Importances")
                st.markdown("*Model-based importance from Random Forest Classifier*")
                
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(X, y_encoded)
                rf_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    rf_df = pd.DataFrame({
                        'Rank': range(1, n_top + 1),
                        'Feature': rf_series.head(n_top).index,
                        'Importance': rf_series.head(n_top).values.round(6)
                    })
                    st.dataframe(rf_df, use_container_width=True, hide_index=True)
                with col2:
                    fig = px.bar(rf_df.head(15), x='Feature', y='Importance',
                               title='Top 15 - Random Forest', color='Importance',
                               color_continuous_scale='Greens')
                    fig.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- Combined Summary ---
                st.divider()
                st.markdown("### 📋 Combined Feature Selection Summary")
                
                top_features = rf_series.head(n_top).index.tolist()
                combined = pd.DataFrame({
                    'Feature': top_features,
                    'RF Importance': [round(rf_series.get(f, 0), 6) for f in top_features],
                    'MI Score': [round(mi_series.get(f, 0), 6) for f in top_features],
                    'Chi2 Score': [round(chi2_series.get(f, 0), 4) for f in top_features]
                })
                st.dataframe(combined, use_container_width=True, hide_index=True)
                
                # Store for modeling
                st.session_state.feature_importance = rf_series
                st.session_state.mutual_info = mi_series
                st.session_state.chi2_scores = chi2_series
                st.session_state.selected_features = rf_series.head(10).index.tolist()
                
                st.success(f"✅ **Top 10 features stored for modeling:** {', '.join(rf_series.head(10).index.tolist())}")
                
                # Download
                csv_combined = combined.to_csv(index=False)
                st.download_button("📥 Download Feature Selection Report", csv_combined,
                                 "feature_selection_report.csv", "text/csv")
            
            except Exception as e:
                st.error(f"❌ Feature selection error: {str(e)}")
                st.info("Please check that your data has categorical columns and the target variable is valid.")
    # ==================== MODELING PAGE ====================
    # ==================== MODELING PAGE ====================
    elif section == "🤖 Modeling":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        # =============================================================
        # CONFIGURATION (Always visible)
        # =============================================================
        st.markdown("## ⚙️ Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target = st.selectbox(
                "Target Variable:", 
                df.columns.tolist(),
                index=df.columns.tolist().index('Risk_Level') if 'Risk_Level' in df.columns else 0,
                key="model_target"
            )
        with col2:
            all_features = [c for c in df.columns if c != target]
            
            default_features = st.session_state.get('selected_features', None)
            if default_features is not None:
                default_features = [f for f in default_features if f in all_features]
            else:
                default_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:10]
                default_features = [f for f in default_features if f in all_features and f != target]
            
            features = st.multiselect("Features:", all_features, default=default_features, key="model_features")
        with col3:
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05, key="model_test_size")
        
        # Models info
        st.markdown("**Models to train:** Logistic Regression | Random Forest | Gradient Boosting")
        
        st.divider()
        
        # =============================================================
        # SINGLE BUTTON - Does everything
        # =============================================================
        if len(features) == 0:
            st.error("Please select at least one feature.")
        else:
            if st.button("🚀 Train & Evaluate Models", type="primary", use_container_width=True, key="run_all"):
                
                # ── Data Preparation ──
                with st.spinner('Preparing data...'):
                    X_selected = df[features].copy()
                    y = df[target].copy()
                    
                    mask = y.notna()
                    X_selected = X_selected[mask]
                    y = y[mask]
                    
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    # FIX: Force all features to numeric
                    X_selected = X_selected.apply(pd.to_numeric, errors='coerce')
                    X_selected = X_selected.fillna(X_selected.median())
                    
                    # Remove any columns that are all NaN
                    X_selected = X_selected.dropna(axis=1, how='all')
                    
                    if X_selected.shape[1] == 0:
                        st.error("No valid numerical features remaining after cleaning. Please select numerical features.")
                        st.stop()
                    
                    # Split with stratification check
                    unique, counts = np.unique(y, return_counts=True)
                    stratify_param = y if (len(unique) > 1 and all(c >= 2 for c in counts)) else None
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, test_size=test_size, random_state=42, stratify=stratify_param
                    )
                
                st.success("✅ Data prepared: " + str(X_train.shape[0]) + " train | " + str(X_test.shape[0]) + " test | " + str(X_train.shape[1]) + " features")
                
                st.divider()
                
                # ── Train Models ──
                st.markdown("### 🤖 Training Models")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Logistic Regression**")
                    lr_progress = st.progress(0)
                    try:
                        logistic_regression_model = LogisticRegression(random_state=42, max_iter=1000)
                        logistic_regression_model.fit(X_train, y_train)
                        lr_progress.progress(100)
                        st.success("✅ Trained")
                    except Exception as e:
                        lr_progress.progress(100)
                        st.error("Failed: " + str(e)[:50])
                        logistic_regression_model = None
                
                with col2:
                    st.markdown("**Random Forest**")
                    rf_progress = st.progress(0)
                    try:
                        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        random_forest_model.fit(X_train, y_train)
                        rf_progress.progress(100)
                        st.success("✅ Trained")
                    except Exception as e:
                        rf_progress.progress(100)
                        st.error("Failed: " + str(e)[:50])
                        random_forest_model = None
                
                with col3:
                    st.markdown("**Gradient Boosting**")
                    gb_progress = st.progress(0)
                    try:
                        gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        gradient_boosting_model.fit(X_train, y_train)
                        gb_progress.progress(100)
                        st.success("✅ Trained")
                    except Exception as e:
                        gb_progress.progress(100)
                        st.error("Failed: " + str(e)[:50])
                        gradient_boosting_model = None
                
                # Check if any model trained successfully
                trained_models = sum([
                    logistic_regression_model is not None,
                    random_forest_model is not None,
                    gradient_boosting_model is not None
                ])
                
                if trained_models == 0:
                    st.error("No models could be trained. Please check your feature selection.")
                    st.stop()
                
                st.divider()
                
                # ── Evaluate Models ──
                st.markdown("### 📊 Model Evaluation")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if logistic_regression_model is not None:
                        lr_pred = logistic_regression_model.predict(X_test)
                        lr_acc = accuracy_score(y_test, lr_pred)
                        lr_prec = precision_score(y_test, lr_pred, average='weighted')
                        lr_rec = recall_score(y_test, lr_pred, average='weighted')
                        lr_f1 = f1_score(y_test, lr_pred, average='weighted')
                        
                        st.markdown("**--- Logistic Regression ---**")
                        st.code(
                            "Accuracy:  " + str(round(lr_acc, 4)) + "\n" +
                            "Precision: " + str(round(lr_prec, 4)) + "\n" +
                            "Recall:    " + str(round(lr_rec, 4)) + "\n" +
                            "F1-Score:  " + str(round(lr_f1, 4))
                        )
                    else:
                        st.markdown("**--- Logistic Regression ---**")
                        st.warning("Not trained")
                        lr_acc = lr_prec = lr_rec = lr_f1 = 0
                
                with col2:
                    if random_forest_model is not None:
                        rf_pred = random_forest_model.predict(X_test)
                        rf_acc = accuracy_score(y_test, rf_pred)
                        rf_prec = precision_score(y_test, rf_pred, average='weighted')
                        rf_rec = recall_score(y_test, rf_pred, average='weighted')
                        rf_f1 = f1_score(y_test, rf_pred, average='weighted')
                        
                        st.markdown("**--- Random Forest ---**")
                        st.code(
                            "Accuracy:  " + str(round(rf_acc, 4)) + "\n" +
                            "Precision: " + str(round(rf_prec, 4)) + "\n" +
                            "Recall:    " + str(round(rf_rec, 4)) + "\n" +
                            "F1-Score:  " + str(round(rf_f1, 4))
                        )
                    else:
                        st.markdown("**--- Random Forest ---**")
                        st.warning("Not trained")
                        rf_acc = rf_prec = rf_rec = rf_f1 = 0
                
                with col3:
                    if gradient_boosting_model is not None:
                        gb_pred = gradient_boosting_model.predict(X_test)
                        gb_acc = accuracy_score(y_test, gb_pred)
                        gb_prec = precision_score(y_test, gb_pred, average='weighted')
                        gb_rec = recall_score(y_test, gb_pred, average='weighted')
                        gb_f1 = f1_score(y_test, gb_pred, average='weighted')
                        
                        st.markdown("**--- Gradient Boosting ---**")
                        st.code(
                            "Accuracy:  " + str(round(gb_acc, 4)) + "\n" +
                            "Precision: " + str(round(gb_prec, 4)) + "\n" +
                            "Recall:    " + str(round(gb_rec, 4)) + "\n" +
                            "F1-Score:  " + str(round(gb_f1, 4))
                        )
                    else:
                        st.markdown("**--- Gradient Boosting ---**")
                        st.warning("Not trained")
                        gb_acc = gb_prec = gb_rec = gb_f1 = 0
                
                st.divider()
                
                # ── Comparison Table ──
                st.markdown("### 📊 Model Performance Comparison")
                
                performance_data = []
                if logistic_regression_model is not None:
                    performance_data.append({
                        'Model': 'Logistic Regression',
                        'Accuracy': round(lr_acc, 4), 'Precision': round(lr_prec, 4),
                        'Recall': round(lr_rec, 4), 'F1-Score': round(lr_f1, 4)
                    })
                if random_forest_model is not None:
                    performance_data.append({
                        'Model': 'Random Forest',
                        'Accuracy': round(rf_acc, 4), 'Precision': round(rf_prec, 4),
                        'Recall': round(rf_rec, 4), 'F1-Score': round(rf_f1, 4)
                    })
                if gradient_boosting_model is not None:
                    performance_data.append({
                        'Model': 'Gradient Boosting',
                        'Accuracy': round(gb_acc, 4), 'Precision': round(gb_prec, 4),
                        'Recall': round(gb_rec, 4), 'F1-Score': round(gb_f1, 4)
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, use_container_width=True, hide_index=True)
                
                # Best model
                if len(performance_df) > 0:
                    best_idx = performance_df['F1-Score'].idxmax()
                    best_name = performance_df.iloc[best_idx]['Model']
                    best_f1 = performance_df.iloc[best_idx]['F1-Score']
                    
                    st.success("🏆 **Best Model: " + best_name + "** (F1-Score: " + str(best_f1) + ")")
                
                st.divider()
                
                # ── Charts ──
                if len(performance_df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Bar Chart Comparison")
                        fig = px.bar(
                            performance_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                            x='Model', y='Score', color='Metric',
                            barmode='group', height=350,
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 🎯 Radar Chart")
                        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        fig = go.Figure()
                        
                        if logistic_regression_model is not None:
                            fig.add_trace(go.Scatterpolar(
                                r=[lr_acc, lr_prec, lr_rec, lr_f1], theta=categories,
                                fill='toself', name='Logistic Regression'
                            ))
                        if random_forest_model is not None:
                            fig.add_trace(go.Scatterpolar(
                                r=[rf_acc, rf_prec, rf_rec, rf_f1], theta=categories,
                                fill='toself', name='Random Forest'
                            ))
                        if gradient_boosting_model is not None:
                            fig.add_trace(go.Scatterpolar(
                                r=[gb_acc, gb_prec, gb_rec, gb_f1], theta=categories,
                                fill='toself', name='Gradient Boosting'
                            ))
                        
                        fig.update_layout(polar=dict(radialaxis=dict(range=[0.9, 1.0])), height=350)
                        st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # ── Feature Importance & Confusion Matrix ──
                if len(performance_df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if random_forest_model is not None:
                            st.markdown("### 🌲 Feature Importance (Random Forest)")
                            importances = random_forest_model.feature_importances_
                            feat_imp = pd.DataFrame({
                                'Feature': features[:len(importances)],
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            fig = px.bar(feat_imp, x='Feature', y='Importance',
                                       color='Importance', color_continuous_scale='Greens', height=350)
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.markdown("### 🌲 Feature Importance")
                            st.warning("Random Forest not trained")
                    
                    with col2:
                        st.markdown("### 🔍 Confusion Matrix (" + best_name + ")")
                        if best_name == 'Logistic Regression' and logistic_regression_model is not None:
                            best_pred = lr_pred
                        elif best_name == 'Random Forest' and random_forest_model is not None:
                            best_pred = rf_pred
                        elif best_name == 'Gradient Boosting' and gradient_boosting_model is not None:
                            best_pred = gb_pred
                        else:
                            best_pred = None
                        
                        if best_pred is not None:
                            cm = confusion_matrix(y_test, best_pred)
                            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No predictions available")
                
                st.divider()
                
                # ── Classification Report ──
                if best_pred is not None:
                    with st.expander("📋 View Classification Report"):
                        report_text = classification_report(y_test, best_pred)
                        st.code(report_text, language='text')
                
                # Store in session
                models_dict = {}
                if logistic_regression_model is not None:
                    models_dict['Logistic Regression'] = logistic_regression_model
                if random_forest_model is not None:
                    models_dict['Random Forest'] = random_forest_model
                if gradient_boosting_model is not None:
                    models_dict['Gradient Boosting'] = gradient_boosting_model
                
                st.session_state.models = models_dict
                st.session_state.trained = True
                
                # Download
                if len(performance_df) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button("📥 Results CSV", performance_df.to_csv(index=False), "results.csv", "text/csv")
                    with col2:
                        if random_forest_model is not None:
                            st.download_button("📥 Feature Importance", feat_imp.to_csv(index=False), "features.csv", "text/csv")
                    with col3:
                        if best_pred is not None:
                            st.download_button("📥 Report", report_text, "report.txt", "text/plain")
    # ==================== CROSS VALIDATION & EVALUATION PAGE ====================
    # ==================== CROSS VALIDATION & EVALUATION PAGE ====================
    elif section == "📊 Cross Validation & Evaluation":
        st.markdown('<p class="section-header">📊 Cross Validation & Model Evaluation</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        # Tabs for each step - click instead of scroll
        cv_tab1, cv_tab2, cv_tab3, cv_tab4, cv_tab5 = st.tabs([
            "⚙️ Configure", 
            "📊 Evaluate Models", 
            "🔄 K-Fold CV", 
            "📋 Summary",
            "📈 Visualizations"
        ])
        
        # =============================================================
        # TAB 1: Configure
        # =============================================================
        with cv_tab1:
            st.markdown("### ⚙️ Configuration")
            st.markdown("Set up target variable, features, and train/test split.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                target = st.selectbox(
                    "Target Variable:", 
                    df.columns.tolist(),
                    index=df.columns.tolist().index('Risk_Level') if 'Risk_Level' in df.columns else 0,
                    key="cv_target"
                )
            with col2:
                all_features = [c for c in df.columns if c != target]
                
                default_features = st.session_state.get('selected_features', None)
                if default_features is not None:
                    default_features = [f for f in default_features if f in all_features]
                else:
                    default_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:10]
                    default_features = [f for f in default_features if f in all_features and f != target]
                
                features = st.multiselect("Features:", all_features, default=default_features, key="cv_features")
            with col3:
                test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05, key="cv_test_size")
            
            st.info("**Models:** Logistic Regression | Random Forest | Gradient Boosting")
            
            st.divider()
            
            if len(features) == 0:
                st.error("Please select at least one feature.")
            else:
                if st.button("🚀 Split Data & Train Models", type="primary", use_container_width=True, key="cv_split_train"):
                    with st.spinner('Preparing data and training models...'):
                        X = df[features].copy()
                        y = df[target].copy()
                        
                        mask = y.notna()
                        X = X[mask]
                        y = y[mask]
                        
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                        
                        X = X.apply(pd.to_numeric, errors='coerce')
                        X = X.fillna(X.median())
                        X = X.dropna(axis=1, how='all')
                        
                        if X.shape[1] == 0:
                            st.error("No valid numerical features.")
                            st.stop()
                        
                        unique, counts = np.unique(y, return_counts=True)
                        stratify_param = y if (len(unique) > 1 and all(c >= 2 for c in counts)) else None
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=stratify_param
                        )
                        
                        st.session_state.cv_X_train = X_train
                        st.session_state.cv_X_test = X_test
                        st.session_state.cv_y_train = y_train
                        st.session_state.cv_y_test = y_test
                    
                    st.success("✅ Data split: " + str(X_train.shape[0]) + " train | " + str(X_test.shape[0]) + " test")
                    
                    # Train models
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Logistic Regression**")
                        lr_pb = st.progress(0)
                        logistic_regression_model = LogisticRegression(random_state=42, max_iter=1000)
                        logistic_regression_model.fit(X_train, y_train)
                        lr_pb.progress(100)
                        st.success("✅ Trained")
                        st.session_state.cv_lr_model = logistic_regression_model
                    
                    with col2:
                        st.markdown("**Random Forest**")
                        rf_pb = st.progress(0)
                        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        random_forest_model.fit(X_train, y_train)
                        rf_pb.progress(100)
                        st.success("✅ Trained")
                        st.session_state.cv_rf_model = random_forest_model
                    
                    with col3:
                        st.markdown("**Gradient Boosting**")
                        gb_pb = st.progress(0)
                        gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        gradient_boosting_model.fit(X_train, y_train)
                        gb_pb.progress(100)
                        st.success("✅ Trained")
                        st.session_state.cv_gb_model = gradient_boosting_model
                    
                    st.success("All models trained! Go to **Evaluate Models** tab.")
        
        # =============================================================
        # TAB 2: Evaluate Models
        # =============================================================
        with cv_tab2:
            st.markdown("### 📊 Evaluate the Models")
            st.markdown("Evaluate each trained model on the testing data using accuracy, precision, recall, and F1-score.")
            
            if not hasattr(st.session_state, 'cv_lr_model'):
                st.warning("⚠️ Please complete Configuration tab first.")
            else:
                X_test = st.session_state.cv_X_test
                y_test = st.session_state.cv_y_test
                
                logistic_regression_model = st.session_state.cv_lr_model
                random_forest_model = st.session_state.cv_rf_model
                gradient_boosting_model = st.session_state.cv_gb_model
                
                if st.button("📊 Evaluate All Models", type="primary", use_container_width=True, key="cv_eval_btn"):
                    with st.spinner('Evaluating models...'):
                        
                        # Logistic Regression
                        st.markdown("### --- Logistic Regression Model Evaluation ---")
                        
                        lr_pred = logistic_regression_model.predict(X_test)
                        lr_acc = accuracy_score(y_test, lr_pred)
                        lr_prec = precision_score(y_test, lr_pred, average='weighted')
                        lr_rec = recall_score(y_test, lr_pred, average='weighted')
                        lr_f1 = f1_score(y_test, lr_pred, average='weighted')
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("Accuracy", str(round(lr_acc, 4)))
                        with col2: st.metric("Precision", str(round(lr_prec, 4)))
                        with col3: st.metric("Recall", str(round(lr_rec, 4)))
                        with col4: st.metric("F1-Score", str(round(lr_f1, 4)))
                        
                        st.code("Accuracy: " + str(round(lr_acc, 4)) + "\nPrecision: " + str(round(lr_prec, 4)) + "\nRecall: " + str(round(lr_rec, 4)) + "\nF1-Score: " + str(round(lr_f1, 4)) + "\n" + "-" * 40)
                        
                        st.markdown("---")
                        
                        # Random Forest
                        st.markdown("### --- RandomForestClassifier Model Evaluation ---")
                        
                        rf_pred = random_forest_model.predict(X_test)
                        rf_acc = accuracy_score(y_test, rf_pred)
                        rf_prec = precision_score(y_test, rf_pred, average='weighted')
                        rf_rec = recall_score(y_test, rf_pred, average='weighted')
                        rf_f1 = f1_score(y_test, rf_pred, average='weighted')
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("Accuracy", str(round(rf_acc, 4)))
                        with col2: st.metric("Precision", str(round(rf_prec, 4)))
                        with col3: st.metric("Recall", str(round(rf_rec, 4)))
                        with col4: st.metric("F1-Score", str(round(rf_f1, 4)))
                        
                        st.code("Accuracy: " + str(round(rf_acc, 4)) + "\nPrecision: " + str(round(rf_prec, 4)) + "\nRecall: " + str(round(rf_rec, 4)) + "\nF1-Score: " + str(round(rf_f1, 4)) + "\n" + "-" * 40)
                        
                        st.markdown("---")
                        
                        # Gradient Boosting
                        st.markdown("### --- GradientBoostingClassifier Model Evaluation ---")
                        
                        gb_pred = gradient_boosting_model.predict(X_test)
                        gb_acc = accuracy_score(y_test, gb_pred)
                        gb_prec = precision_score(y_test, gb_pred, average='weighted')
                        gb_rec = recall_score(y_test, gb_pred, average='weighted')
                        gb_f1 = f1_score(y_test, gb_pred, average='weighted')
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("Accuracy", str(round(gb_acc, 4)))
                        with col2: st.metric("Precision", str(round(gb_prec, 4)))
                        with col3: st.metric("Recall", str(round(gb_rec, 4)))
                        with col4: st.metric("F1-Score", str(round(gb_f1, 4)))
                        
                        st.code("Accuracy: " + str(round(gb_acc, 4)) + "\nPrecision: " + str(round(gb_prec, 4)) + "\nRecall: " + str(round(gb_rec, 4)) + "\nF1-Score: " + str(round(gb_f1, 4)) + "\n" + "-" * 40)
                        
                        st.markdown("---")
                        
                        # Comparison Table
                        st.markdown("### 📊 Model Performance Comparison")
                        
                        performance_df = pd.DataFrame({
                            'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
                            'Accuracy': [round(lr_acc, 4), round(rf_acc, 4), round(gb_acc, 4)],
                            'Precision': [round(lr_prec, 4), round(rf_prec, 4), round(gb_prec, 4)],
                            'Recall': [round(lr_rec, 4), round(rf_rec, 4), round(gb_rec, 4)],
                            'F1-Score': [round(lr_f1, 4), round(rf_f1, 4), round(gb_f1, 4)]
                        })
                        
                        st.dataframe(performance_df, use_container_width=True, hide_index=True)
                        
                        best_idx = performance_df['F1-Score'].idxmax()
                        best_name = performance_df.iloc[best_idx]['Model']
                        best_f1 = performance_df.iloc[best_idx]['F1-Score']
                        
                        st.success("🏆 **Best Model: " + best_name + "** (F1-Score: " + str(best_f1) + ")")
                        
                        # Store
                        st.session_state.cv_lr_pred = lr_pred
                        st.session_state.cv_rf_pred = rf_pred
                        st.session_state.cv_gb_pred = gb_pred
                        st.session_state.cv_performance_df = performance_df
                        st.session_state.cv_best_name = best_name
                        st.session_state.cv_evaluation_ran = True
                        
                        st.success("Evaluation complete! Go to **K-Fold CV** tab.")
        
        # =============================================================
        # TAB 3: K-Fold Cross Validation
        # =============================================================
        with cv_tab3:
            st.markdown("### 🔄 K-Fold Cross Validation")
            st.markdown("Perform Stratified K-Fold Cross Validation for robust performance estimates.")
            
            if not st.session_state.get('cv_evaluation_ran', False):
                st.warning("⚠️ Please complete Evaluate Models tab first.")
            else:
                X_train = st.session_state.cv_X_train
                y_train = st.session_state.cv_y_train
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    cv_folds = st.slider("Number of Folds (K):", 3, 10, 5, key="cv_k_slider")
                with col2:
                    st.metric("Training Samples", str(X_train.shape[0]))
                with col3:
                    st.metric("Features", str(X_train.shape[1]))
                
                if st.button("🔄 Run K-Fold Cross Validation", type="primary", use_container_width=True, key="cv_run_btn"):
                    with st.spinner('Running ' + str(cv_folds) + '-Fold Cross Validation...'):
                        
                        models_cv = {
                            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
                        }
                        
                        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        
                        cv_results = []
                        fold_details = {}
                        
                        col1, col2, col3 = st.columns(3)
                        
                        for i, (name, model) in enumerate(models_cv.items()):
                            with [col1, col2, col3][i]:
                                st.markdown("**" + name + "**")
                                progress = st.progress(0)
                                
                                try:
                                    acc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                                    progress.progress(25)
                                    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
                                    progress.progress(50)
                                    prec_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted', n_jobs=-1)
                                    progress.progress(75)
                                    rec_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_weighted', n_jobs=-1)
                                    progress.progress(100)
                                    
                                    cv_results.append({
                                        'Model': name,
                                        'Accuracy Mean': round(acc_scores.mean(), 4),
                                        'Accuracy Std': round(acc_scores.std(), 4),
                                        'Precision Mean': round(prec_scores.mean(), 4),
                                        'F1 Mean': round(f1_scores.mean(), 4),
                                        'F1 Std': round(f1_scores.std(), 4)
                                    })
                                    
                                    fold_details[name] = {'Accuracy': acc_scores, 'F1': f1_scores}
                                    
                                    st.success("✅ Done")
                                    st.metric("Mean F1", str(round(f1_scores.mean(), 4)))
                                except Exception as e:
                                    progress.progress(100)
                                    st.error("Failed")
                        
                        if len(cv_results) > 0:
                            st.divider()
                            st.markdown("### 📊 K-Fold Cross Validation Results")
                            
                            cv_df = pd.DataFrame(cv_results)
                            st.dataframe(cv_df, use_container_width=True, hide_index=True)
                            
                            best_cv_idx = cv_df['F1 Mean'].idxmax()
                            best_cv_name = cv_df.iloc[best_cv_idx]['Model']
                            best_cv_f1 = cv_df.iloc[best_cv_idx]['F1 Mean']
                            
                            st.success("🏆 **Best CV Model: " + best_cv_name + "** (F1: " + str(best_cv_f1) + ")")
                            
                            # Store
                            st.session_state.cv_results_df = cv_df
                            st.session_state.cv_fold_details = fold_details
                            st.session_state.cv_ran = True
                            
                            st.success("CV complete! Go to **Summary** or **Visualizations** tab.")
        
        # =============================================================
        # TAB 4: Summary
        # =============================================================
        with cv_tab4:
            st.markdown("### 📋 Cross Validation Summary")
            
            if not st.session_state.get('cv_ran', False):
                st.warning("⚠️ Please complete K-Fold CV tab first.")
            else:
                cv_df = st.session_state.cv_results_df
                performance_df = st.session_state.get('cv_performance_df', None)
                
                # Test set results
                if performance_df is not None:
                    st.markdown("#### 📊 Test Set Evaluation")
                    st.dataframe(performance_df, use_container_width=True, hide_index=True)
                    
                    best_idx = performance_df['F1-Score'].idxmax()
                    st.success("🏆 Best on Test: **" + performance_df.iloc[best_idx]['Model'] + "** (F1: " + str(performance_df.iloc[best_idx]['F1-Score']) + ")")
                
                st.divider()
                
                # CV results
                st.markdown("#### 🔄 Cross Validation Results")
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
                
                best_cv_idx = cv_df['F1 Mean'].idxmax()
                best_cv_name = cv_df.iloc[best_cv_idx]['Model']
                best_cv_f1 = cv_df.iloc[best_cv_idx]['F1 Mean']
                best_cv_acc = cv_df.iloc[best_cv_idx]['Accuracy Mean']
                
                st.success("🏆 Best CV Model: **" + best_cv_name + "**")
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Mean F1", str(best_cv_f1))
                with col2: st.metric("Mean Accuracy", str(best_cv_acc))
                with col3: st.metric("F1 Std", str(cv_df.iloc[best_cv_idx]['F1 Std']))
                
                st.divider()
                
                # Stability
                st.markdown("#### 📊 Model Stability")
                
                col1, col2, col3 = st.columns(3)
                for i, (_, row) in enumerate(cv_df.iterrows()):
                    with [col1, col2, col3][i]:
                        cv_score = round(row['F1 Std'] / row['F1 Mean'], 4) if row['F1 Mean'] > 0 else 0
                        emoji = "🟢" if cv_score < 0.02 else ("🟡" if cv_score < 0.05 else "🔴")
                        st.markdown(emoji + " **" + row['Model'] + "**")
                        st.metric("CV (Std/Mean)", str(cv_score))
                
                st.divider()
                
                # Final ranking
                st.markdown("#### 🏆 Final Model Ranking")
                
                cv_sorted = cv_df.sort_values('F1 Mean', ascending=False)
                summary = "Model Ranking (Cross Validation):\n" + "=" * 40 + "\n\n"
                for rank, (_, row) in enumerate(cv_sorted.iterrows(), 1):
                    summary += str(rank) + ". " + row['Model'] + "\n   F1: " + str(row['F1 Mean']) + " (±" + str(row['F1 Std']) + ")\n   Acc: " + str(row['Accuracy Mean']) + "\n\n"
                
                st.code(summary, language='text')
                
                # Download
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("📥 CV Results CSV", cv_df.to_csv(index=False), "cv_results.csv", "text/csv")
                with col2:
                    st.download_button("📥 Test Results CSV", performance_df.to_csv(index=False), "test_results.csv", "text/csv") if performance_df is not None else None
                with col3:
                    st.download_button("📥 Summary TXT", summary, "cv_summary.txt", "text/plain")
        
        # =============================================================
        # TAB 5: Visualizations
        # =============================================================
        with cv_tab5:
            st.markdown("### 📈 Visualizations & Feature Analysis")
            st.markdown("Comprehensive visual comparison and feature importance analysis.")
            
            if not st.session_state.get('cv_ran', False):
                st.warning("⚠️ Please complete K-Fold CV tab first.")
            else:
                cv_df = st.session_state.cv_results_df
                performance_df = st.session_state.get('cv_performance_df', None)
                fold_details = st.session_state.cv_fold_details
                
                # Sub-tabs
                viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5, viz_tab6 = st.tabs([
                    "📊 Performance Charts", 
                    "🌲 Feature Importance", 
                    "📊 Analyze Relevance",
                    "📈 Visualize Features",
                    "🔍 Confusion Matrices",
                    "📋 Summarize Findings"
                ])
                
                # ── Sub-Tab 1: Performance Charts ──
                with viz_tab1:
                    st.markdown("#### 📊 Test Set Performance Comparison")
                    
                    if performance_df is not None:
                        fig = px.bar(
                            performance_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                            x='Model', y='Score', color='Metric',
                            barmode='group', height=400, title='Test Set Performance',
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.markdown("#### 🔄 Cross Validation Comparison")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = go.Figure()
                        for _, row in cv_df.iterrows():
                            fig.add_trace(go.Bar(name=row['Model'], x=['Accuracy'], y=[row['Accuracy Mean']],
                                error_y=dict(type='data', array=[row['Accuracy Std']]),
                                text=str(row['Accuracy Mean']) + ' ± ' + str(row['Accuracy Std'])))
                        fig.update_layout(barmode='group', height=400, title='Accuracy with Error Bars')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure()
                        for _, row in cv_df.iterrows():
                            fig.add_trace(go.Bar(name=row['Model'], x=['F1 Score'], y=[row['F1 Mean']],
                                error_y=dict(type='data', array=[row['F1 Std']]),
                                text=str(row['F1 Mean']) + ' ± ' + str(row['F1 Std'])))
                        fig.update_layout(barmode='group', height=400, title='F1 Score with Error Bars')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.markdown("#### 🎯 Radar Chart")
                    if performance_df is not None:
                        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        fig = go.Figure()
                        for _, row in performance_df.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                                theta=categories, fill='toself', name=row['Model']))
                        fig.update_layout(polar=dict(radialaxis=dict(range=[0.9, 1.0])), height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    st.markdown("#### 📈 Best Model Fold-by-Fold")
                    
                    best_cv_name = cv_df.iloc[cv_df['F1 Mean'].idxmax()]['Model']
                    if best_cv_name in fold_details:
                        fd = fold_details[best_cv_name]
                        fold_df = pd.DataFrame({
                            'Fold': [str(i+1) for i in range(len(fd['Accuracy']))],
                            'Accuracy': fd['Accuracy'].round(4), 'F1 Score': fd['F1'].round(4)
                        })
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fold_df['Fold'], y=fold_df['Accuracy'],
                                mode='lines+markers', name='Accuracy', line=dict(color='#0984e3', width=3)))
                            fig.add_trace(go.Scatter(x=fold_df['Fold'], y=fold_df['F1 Score'],
                                mode='lines+markers', name='F1', line=dict(color='#00b894', width=3)))
                            fig.update_layout(title='Fold-by-Fold: ' + best_cv_name, yaxis_range=[0.5, 1.0], height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            st.dataframe(fold_df, use_container_width=True, hide_index=True)
                
                # ── Sub-Tab 2: Feature Importance ──
                with viz_tab2:
                    st.markdown("### 🌲 Feature Importance Extraction")
                    
                    logistic_regression_model = st.session_state.cv_lr_model
                    random_forest_model = st.session_state.cv_rf_model
                    gradient_boosting_model = st.session_state.cv_gb_model
                    
                    features_list = st.session_state.get('cv_features', [])
                    
                    if len(features_list) == 0:
                        features_list = [str(i) for i in range(len(random_forest_model.feature_importances_))]
                    
                    feature_relevance = {}
                    feature_relevance['Random Forest'] = random_forest_model.feature_importances_
                    feature_relevance['Gradient Boosting'] = gradient_boosting_model.feature_importances_
                    feature_relevance['Logistic Regression'] = np.mean(np.abs(logistic_regression_model.coef_), axis=0)
                    
                    feature_relevance_series = {}
                    for model_name, scores in feature_relevance.items():
                        feature_relevance_series[model_name] = pd.Series(
                            scores, index=features_list[:len(scores)]
                        ).sort_values(ascending=False)
                    
                    st.success("✅ Feature relevance extracted for all models.")
                    
                    model_tabs = st.tabs(["🌲 Random Forest", "🚀 Gradient Boosting", "📊 Logistic Regression"])
                    colors = ['Greens', 'Purples', 'Blues']
                    
                    for i, (model_name, series) in enumerate(feature_relevance_series.items()):
                        with model_tabs[i]:
                            actual_n = min(15, len(series))
                            feat_df = pd.DataFrame({
                                'Feature': series.head(actual_n).index.tolist(),
                                'Importance': series.head(actual_n).values.round(6).tolist()
                            })
                            col1, col2 = st.columns([3, 2])
                            with col1:
                                fig = px.bar(feat_df, x='Feature', y='Importance',
                                           title=model_name + ' Feature Importance',
                                           color='Importance', color_continuous_scale=colors[i])
                                fig.update_layout(xaxis_tickangle=-45, height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                st.dataframe(feat_df, use_container_width=True, hide_index=True)
                    
                    # Store for other tabs
                    st.session_state.viz_feature_relevance_series = feature_relevance_series
                
                # ── Sub-Tab 3: Analyze Feature Relevance ──
                with viz_tab3:
                    st.markdown("### 📊 Analyze Feature Relevance")
                    
                    if hasattr(st.session_state, 'viz_feature_relevance_series'):
                        feature_relevance_series = st.session_state.viz_feature_relevance_series
                    else:
                        logistic_regression_model = st.session_state.cv_lr_model
                        random_forest_model = st.session_state.cv_rf_model
                        gradient_boosting_model = st.session_state.cv_gb_model
                        features_list = st.session_state.get('cv_features', [])
                        if len(features_list) == 0:
                            features_list = [str(i) for i in range(len(random_forest_model.feature_importances_))]
                        
                        feature_relevance = {}
                        feature_relevance['Random Forest'] = random_forest_model.feature_importances_
                        feature_relevance['Gradient Boosting'] = gradient_boosting_model.feature_importances_
                        feature_relevance['Logistic Regression'] = np.mean(np.abs(logistic_regression_model.coef_), axis=0)
                        
                        feature_relevance_series = {}
                        for model_name, scores in feature_relevance.items():
                            feature_relevance_series[model_name] = pd.Series(
                                scores, index=features_list[:len(scores)]
                            ).sort_values(ascending=False)
                    
                    target_name = st.session_state.get('cv_target', 'Target')
                    n_top = 10
                    
                    st.markdown("### --- Top " + str(n_top) + " Features for '" + target_name + "' Models ---")
                    
                    for model_name, series in feature_relevance_series.items():
                        st.markdown("**" + model_name + ":**")
                        actual_n = min(n_top, len(series))
                        top_df = pd.DataFrame({
                            'Rank': range(1, actual_n + 1),
                            'Feature': series.head(actual_n).index.tolist(),
                            'Importance': series.head(actual_n).values.round(6).tolist()
                        })
                        st.dataframe(top_df, use_container_width=True, hide_index=True)
                        st.markdown("---")
                
                # ── Sub-Tab 4: Visualize Features ──
                with viz_tab4:
                    st.markdown("### 📈 Visualize Feature Importances")
                    st.markdown("Visualize top feature importances using Matplotlib/Seaborn bar plots.")
                    
                    if hasattr(st.session_state, 'viz_feature_relevance_series'):
                        feature_relevance_series = st.session_state.viz_feature_relevance_series
                    else:
                        logistic_regression_model = st.session_state.cv_lr_model
                        random_forest_model = st.session_state.cv_rf_model
                        gradient_boosting_model = st.session_state.cv_gb_model
                        features_list = st.session_state.get('cv_features', [])
                        if len(features_list) == 0:
                            features_list = [str(i) for i in range(len(random_forest_model.feature_importances_))]
                        
                        feature_relevance = {}
                        feature_relevance['Random Forest'] = random_forest_model.feature_importances_
                        feature_relevance['Gradient Boosting'] = gradient_boosting_model.feature_importances_
                        feature_relevance['Logistic Regression'] = np.mean(np.abs(logistic_regression_model.coef_), axis=0)
                        
                        feature_relevance_series = {}
                        for model_name, scores in feature_relevance.items():
                            feature_relevance_series[model_name] = pd.Series(
                                scores, index=features_list[:len(scores)]
                            ).sort_values(ascending=False)
                    
                    target_name = st.session_state.get('cv_target', 'Target')
                    n_top_viz = st.slider("Number of top features:", 5, 20, 10, key="viz_n_top")
                    
                    model_colors = {'Random Forest': '#27ae60', 'Gradient Boosting': '#8e44ad', 'Logistic Regression': '#2980b9'}
                    
                    st.markdown("### --- Top " + str(n_top_viz) + " Feature Relevance for '" + target_name + "' Models ---")
                    
                    for model_name, series in feature_relevance_series.items():
                        st.markdown("**" + model_name + "**")
                        actual_n = min(n_top_viz, len(series))
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        top_features = series.head(actual_n).index.tolist()
                        top_values = series.head(actual_n).values
                        
                        sns.barplot(x=top_features, y=top_values, color=model_colors.get(model_name, '#3498db'), ax=ax)
                        ax.set_title("Top " + str(actual_n) + " Feature Relevance for " + target_name + " (" + model_name + ")", fontsize=14, fontweight='bold')
                        ax.set_xlabel("Features", fontsize=12)
                        ax.set_ylabel("Relevance Score", fontsize=12)
                        ax.tick_params(axis='x', rotation=45)
                        
                        for j, v in enumerate(top_values):
                            ax.text(j, v + (v * 0.02), str(round(v, 4)), ha='center', va='bottom', fontsize=8)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        st.markdown("---")
                    
                    # Combined plot
                    st.markdown("### 📊 Combined Feature Importance Comparison")
                    rf_top = feature_relevance_series['Random Forest'].head(n_top_viz).index.tolist()
                    
                    fig, ax = plt.subplots(figsize=(14, 7))
                    x = range(len(rf_top))
                    width = 0.25
                    
                    for i, (model_name, series) in enumerate(feature_relevance_series.items()):
                        values = [series.get(f, 0) for f in rf_top]
                        if sum(values) > 0:
                            values = [v / sum(values) for v in values]
                        ax.bar([p + i * width for p in x], values, width, label=model_name, 
                              color=list(model_colors.values())[i], alpha=0.8)
                    
                    ax.set_title("Top " + str(n_top_viz) + " Feature Relevance Comparison - " + target_name, fontsize=14, fontweight='bold')
                    ax.set_xlabel("Features", fontsize=12)
                    ax.set_ylabel("Normalized Relevance Score", fontsize=12)
                    ax.set_xticks([p + width for p in x])
                    ax.set_xticklabels(rf_top, rotation=45, ha='right')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # ── Sub-Tab 5: Confusion Matrices ──
                with viz_tab5:
                    st.markdown("#### 🔍 Confusion Matrices")
                    
                    X_test = st.session_state.cv_X_test
                    y_test = st.session_state.cv_y_test
                    
                    if hasattr(st.session_state, 'cv_lr_pred'):
                        cm_tab1, cm_tab2, cm_tab3 = st.tabs(["Logistic Regression", "Random Forest", "Gradient Boosting"])
                        
                        with cm_tab1:
                            cm = confusion_matrix(y_test, st.session_state.cv_lr_pred)
                            fig = px.imshow(cm, text_auto=True, title='Logistic Regression',
                                          labels=dict(x='Predicted', y='Actual'), color_continuous_scale='Blues')
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("Classification Report"):
                                st.code(classification_report(y_test, st.session_state.cv_lr_pred), language='text')
                        
                        with cm_tab2:
                            cm = confusion_matrix(y_test, st.session_state.cv_rf_pred)
                            fig = px.imshow(cm, text_auto=True, title='Random Forest',
                                          labels=dict(x='Predicted', y='Actual'), color_continuous_scale='Greens')
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("Classification Report"):
                                st.code(classification_report(y_test, st.session_state.cv_rf_pred), language='text')
                        
                        with cm_tab3:
                            cm = confusion_matrix(y_test, st.session_state.cv_gb_pred)
                            fig = px.imshow(cm, text_auto=True, title='Gradient Boosting',
                                          labels=dict(x='Predicted', y='Actual'), color_continuous_scale='Purples')
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("Classification Report"):
                                st.code(classification_report(y_test, st.session_state.cv_gb_pred), language='text')
                
                # ── Sub-Tab 6: Summarize Findings ──
                with viz_tab6:
                    st.markdown("### 📋 Summarize Findings")
                    st.markdown("""
                    **Subtask:** Summarize the key findings about feature relevance from the analysis and visualization.
                    """)
                    
                    # Get data
                    if hasattr(st.session_state, 'viz_feature_relevance_series'):
                        feature_relevance_series = st.session_state.viz_feature_relevance_series
                    else:
                        logistic_regression_model = st.session_state.cv_lr_model
                        random_forest_model = st.session_state.cv_rf_model
                        gradient_boosting_model = st.session_state.cv_gb_model
                        features_list = st.session_state.get('cv_features', [])
                        if len(features_list) == 0:
                            features_list = [str(i) for i in range(len(random_forest_model.feature_importances_))]
                        
                        feature_relevance = {}
                        feature_relevance['Random Forest'] = random_forest_model.feature_importances_
                        feature_relevance['Gradient Boosting'] = gradient_boosting_model.feature_importances_
                        feature_relevance['Logistic Regression'] = np.mean(np.abs(logistic_regression_model.coef_), axis=0)
                        
                        feature_relevance_series = {}
                        for model_name, scores in feature_relevance.items():
                            feature_relevance_series[model_name] = pd.Series(
                                scores, index=features_list[:len(scores)]
                            ).sort_values(ascending=False)
                    
                    target_name = st.session_state.get('cv_target', 'Target')
                    best_cv_name = cv_df.iloc[cv_df['F1 Mean'].idxmax()]['Model']
                    best_f1 = cv_df.iloc[cv_df['F1 Mean'].idxmax()]['F1 Mean']
                    best_acc = cv_df.iloc[cv_df['F1 Mean'].idxmax()]['Accuracy Mean']
                    
                    n_top = 10
                    
                    # ── Key Findings Summary ──
                    st.markdown("## 📊 Key Findings Summary")
                    
                    # Best Model
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #d4edda, #c3e6cb); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #155724; margin-top: 0;'>🏆 Best Performing Model</h3>
                    <p style='color: #155724; font-size: 1.1rem;'>
                    <strong>""" + best_cv_name + """</strong> achieved the highest cross-validation performance:<br>
                    • F1 Score: """ + str(best_f1) + """ (±""" + str(cv_df.iloc[cv_df['F1 Mean'].idxmax()]['F1 Std']) + """)<br>
                    • Accuracy: """ + str(best_acc) + """ (±""" + str(cv_df.iloc[cv_df['F1 Mean'].idxmax()]['Accuracy Std']) + """)
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Feature Importance Findings
                    st.markdown("## 🌲 Feature Importance Findings")
                    
                    # Get common top features
                    all_top_sets = []
                    for series in feature_relevance_series.values():
                        actual = min(n_top, len(series))
                        all_top_sets.append(set(series.head(actual).index.tolist()))
                    
                    common_features = all_top_sets[0].intersection(*all_top_sets[1:]) if len(all_top_sets) > 1 else set()
                    
                    # Feature categories
                    risk_features = set()
                    density_features = set()
                    location_features = set()
                    source_features = set()
                    ph_features = set()
                    other_features = set()
                    
                    for series in feature_relevance_series.values():
                        actual = min(n_top, len(series))
                        for feat in series.head(actual).index.tolist():
                            feat_str = str(feat)
                            if 'Risk' in feat_str:
                                risk_features.add(feat_str)
                            elif 'Density' in feat_str or 'Population' in feat_str:
                                density_features.add(feat_str)
                            elif 'Location' in feat_str:
                                location_features.add(feat_str)
                            elif 'Source' in feat_str:
                                source_features.add(feat_str)
                            elif 'pH' in feat_str:
                                ph_features.add(feat_str)
                            else:
                                other_features.add(feat_str)
                    
                    # Display findings
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Feature Categories in Top " + str(n_top))
                        
                        cat_data = pd.DataFrame({
                            'Category': ['Risk Level', 'Population/Density', 'Location', 'Source', 'pH', 'Other'],
                            'Count': [len(risk_features), len(density_features), len(location_features), 
                                     len(source_features), len(ph_features), len(other_features)]
                        }).sort_values('Count', ascending=False)
                        
                        st.dataframe(cat_data, use_container_width=True, hide_index=True)
                        
                        fig = px.bar(cat_data, x='Category', y='Count', title='Feature Categories Distribution',
                                   color='Category', color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 🤝 Model Agreement")
                        st.markdown("**Common features across all models:** " + str(len(common_features)))
                        
                        if len(common_features) > 0:
                            st.markdown("**Universally important features:**")
                            for feat in list(common_features)[:5]:
                                st.markdown("• " + str(feat))
                        
                        # Top 3 dominant features
                        all_scores = {}
                        for feat in set().union(*all_top_sets):
                            all_scores[feat] = sum(series.get(feat, 0) for series in feature_relevance_series.values())
                        
                        top3 = sorted(all_scores, key=all_scores.get, reverse=True)[:3]
                        st.markdown("**Top 3 Most Dominant Features:**")
                        for i, feat in enumerate(top3, 1):
                            st.markdown(str(i) + ". **" + str(feat) + "**")
                    
                    st.divider()
                    
                    # Detailed Summary
                    st.markdown("## 📝 Detailed Summary")
                    
                    # Generate summary text
                    summary_text = (
                        "FEATURE RELEVANCE ANALYSIS SUMMARY\n" +
                        "=" * 50 + "\n\n" +
                        "Target Variable: " + target_name + "\n" +
                        "Best Model: " + best_cv_name + " (F1: " + str(best_f1) + ")\n\n" +
                        "KEY FINDINGS:\n" +
                        "-" * 30 + "\n\n"
                    )
                    
                    if len(risk_features) > len(density_features):
                        summary_text += (
                            "1. RISK FEATURES DOMINATE:\n" +
                            "   Risk-related features (especially Risk_Level encoded features)\n" +
                            "   consistently appear as the most important predictors across\n" +
                            "   all three models. This indicates that risk indicators are\n" +
                            "   strongly correlated with the target variable.\n\n"
                        )
                    
                    if len(density_features) > 0:
                        summary_text += (
                            "2. POPULATION DENSITY INFLUENCE:\n" +
                            "   Population density features (" + str(len(density_features)) + " in top " + str(n_top) + ")\n" +
                            "   show significant predictive power, suggesting environmental\n" +
                            "   and demographic factors play an important role.\n\n"
                        )
                    
                    if len(ph_features) > 0:
                        summary_text += (
                            "3. pH LEVELS MATTER:\n" +
                            "   pH features appear as important predictors, indicating that\n" +
                            "   water chemistry influences microplastic risk levels.\n\n"
                        )
                    
                    summary_text += (
                        "4. MODEL CONSISTENCY:\n" +
                        "   " + str(len(common_features)) + " features appear in the top " + str(n_top) + " for ALL models,\n" +
                        "   showing strong agreement on the most important predictors.\n" +
                        "   The models agree most on risk-encoded features.\n\n"
                    )
                    
                    summary_text += (
                        "5. BEST MODEL: " + best_cv_name + "\n" +
                        "   Achieved F1 Score of " + str(best_f1) + " with low variance\n" +
                        "   across CV folds, indicating reliable performance.\n\n" +
                        "=" * 50 + "\n"
                    )
                    
                    st.code(summary_text, language='text')
                    
                    st.divider()
                    
                    # Recommendations
                    st.markdown("## 💡 Recommendations")
                    
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #e8f4f8, #d1ecf1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
                    <h4 style='color: #0c5460; margin-top: 0;'>Based on the feature relevance analysis:</h4>
                    <ol style='color: #0c5460;'>
                        <li><strong>Focus on Risk Indicators:</strong> Risk-related features are consistently the strongest predictors across all models.</li>
                        <li><strong>Include Environmental Factors:</strong> Population density and pH levels provide additional predictive power.</li>
                        <li><strong>Use Ensemble Methods:</strong> Random Forest and Gradient Boosting show robust performance with good feature importance agreement.</li>
                        <li><strong>Feature Engineering:</strong> Consider creating interaction features between risk levels and environmental factors.</li>
                        <li><strong>Model Selection:</strong> Both Random Forest and Gradient Boosting are recommended for deployment based on their strong CV performance.</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download summary
                    st.download_button(
                        "📥 Download Full Summary Report",
                        summary_text,
                        "feature_relevance_summary.txt",
                        "text/plain"
                    )
if __name__ == "__main__":
    main()
