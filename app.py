"""
Microplastic Risk Analysis System
Home Page - Objective 1: Analyze the distribution of risk score
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif, chi2
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .objective-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .code-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    .sub-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #34495e;
        margin: 1rem 0;
    }
    .stButton > button {
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
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


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

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Objectives")
st.sidebar.markdown("""
- ✅ **Obj 1**: Analyze Risk Score Distribution
- ⬜ Obj 2: Preprocessing
- ⬜ Obj 3: Feature Selection
- ⬜ Obj 4: Model Training
- ⬜ Obj 5: Cross Validation
""")


# ==================== DATA FUNCTIONS ====================
def load_data(uploaded_file):
    """Load dataset with multiple encoding support"""
    try:
        if uploaded_file.name.endswith('.csv'):
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    st.success(f"✅ File loaded with {enc} encoding")
                    return df
                except:
                    continue
            # If all fail, try with error handling
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
            st.warning("⚠️ File loaded with some encoding issues")
            return df
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            st.success("✅ Excel file loaded")
            return df
        else:
            st.error("❌ Unsupported format. Please upload CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def generate_sample_data():
    """Generate sample microplastic dataset"""
    np.random.seed(42)
    n = 500
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
        'MP_Count_per_L': np.random.poisson(lam=50, size=n),
        'Particle_Size_um': np.random.normal(100, 30, n),
        'Risk_Score': np.random.normal(70, 15, n),
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n, p=[0.3, 0.35, 0.25, 0.1]),
        'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n),
        'Polymer_Type': np.random.choice(['PE', 'PP', 'PS', 'PET', 'PVC'], n),
        'Water_Source': np.random.choice(['River', 'Lake', 'Ocean', 'Groundwater'], n),
        'pH': np.random.normal(7, 0.5, n),
        'Temperature_C': np.random.normal(20, 5, n),
        'Location': np.random.choice(['Urban', 'Rural', 'Industrial', 'Coastal'], n),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n)
    }
    df = pd.DataFrame(data)
    # Add some outliers for better visualization
    outlier_idx = np.random.choice(n, size=20, replace=False)
    df.loc[outlier_idx, 'Risk_Score'] = np.random.uniform(100, 150, 20)
    return df


# ==================== HOME PAGE ====================
def home_page():
    st.markdown('<h1 class="main-header">🏠 Home - Microplastic Risk Analysis</h1>', unsafe_allow_html=True)
    
    # ============================================
    # File Upload Section
    # ============================================
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Dataset (CSV/Excel)", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload your microplastic dataset"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Generate Sample Data", use_container_width=True):
            st.session_state.data = generate_sample_data()
            st.success("✅ Sample dataset generated!")
            st.rerun()
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.data is not None:
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data",
                data=csv,
                file_name="microplastic_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.data = df
    
    # Check if data exists
    if 'data' not in st.session_state or st.session_state.data is None:
        st.info("👆 Please upload a dataset or generate sample data to begin analysis.")
        return
    
    df = st.session_state.data
    
    # ============================================
    # OBJECTIVE 1: Analyze the distribution of risk score
    # ============================================
    st.markdown("---")
    
    # Objective description
    st.markdown("""
    <div class="objective-box">
        <h2 style="color: #667eea; margin: 0;">🎯 Objective 1: Analyze the Distribution of Risk Score</h2>
        <p style="color: #666; margin-top: 0.5rem;">
            <strong>Reasoning:</strong> Visualize the distribution of the <code>Risk_Score</code> column 
            using a histogram and a box plot as instructed, including titles and labels for clarity.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if Risk_Score column exists
    if 'Risk_Score' not in df.columns:
        st.error("❌ 'Risk_Score' column not found in the dataset!")
        st.write("Available columns:", df.columns.tolist())
        return
    
    # Convert to numeric and handle missing values
    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
    clean_risk = df['Risk_Score'].dropna()
    
    if len(clean_risk) == 0:
        st.error("❌ No valid data in Risk_Score column!")
        return
    
    # ============================================
    # Code Display
    # ============================================
    st.markdown('<h3 class="sub-header">💻 Source Code</h3>', unsafe_allow_html=True)
    
    with st.expander("📝 View Code", expanded=False):
        st.markdown("""
        <div class="code-box">
        <pre><code>
# Create a histogram of the Risk_Score column
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Risk_Score', kde=True, bins=30)
plt.title('Distribution of Risk Score')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.show()

# Create a box plot of the Risk_Score column
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='Risk_Score')
plt.title('Box Plot of Risk Score')
plt.ylabel('Risk Score')
plt.show()
        </code></pre>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # Data Preview Table
    # ============================================
    st.markdown('<h3 class="sub-header">📋 Data Preview - First 5 Rows</h3>', unsafe_allow_html=True)
    
    # Highlight Risk_Score column
    st.dataframe(
        df.head().style.background_gradient(subset=['Risk_Score'], cmap='Blues'),
        use_container_width=True
    )
    
    # ============================================
    # Statistics Table
    # ============================================
    st.markdown('<h3 class="sub-header">📊 Risk Score Statistics Table</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary statistics table
        stats_data = {
            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', '25th Percentile', 
                         '75th Percentile', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f'{len(clean_risk):,}',
                f'{clean_risk.mean():.4f}',
                f'{clean_risk.median():.4f}',
                f'{clean_risk.std():.4f}',
                f'{clean_risk.min():.4f}',
                f'{clean_risk.quantile(0.25):.4f}',
                f'{clean_risk.quantile(0.75):.4f}',
                f'{clean_risk.max():.4f}',
                f'{clean_risk.skew():.4f}',
                f'{clean_risk.kurtosis():.4f}'
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Risk categories table
        st.markdown("#### Risk Score Categories")
        bins = [0, 25, 50, 75, float('inf')]
        labels = ['🟢 Low (0-25)', '🟡 Medium (25-50)', '🟠 High (50-75)', '🔴 Critical (75+)']
        categories = pd.cut(clean_risk, bins=bins, labels=labels)
        cat_counts = categories.value_counts().sort_index()
        
        cat_data = {
            'Category': cat_counts.index.tolist(),
            'Count': cat_counts.values.tolist(),
            'Percentage': [f'{(c/len(clean_risk))*100:.1f}%' for c in cat_counts.values]
        }
        cat_df = pd.DataFrame(cat_data)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
        
        # Progress bars for categories
        st.markdown("#### Distribution")
        for cat, count in zip(cat_counts.index, cat_counts.values):
            pct = (count / len(clean_risk)) * 100
            st.markdown(f"**{cat}**: {count:,} ({pct:.1f}%)")
            st.progress(int(pct))
    
    # ============================================
    # VISUALIZATIONS - Using your exact code
    # ============================================
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📊 Risk Score Distribution Visualizations</h3>', unsafe_allow_html=True)
    
    # Tabs for different views
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Histogram", 
        "📦 Box Plot", 
        "🔍 Combined View"
    ])
    
    with viz_tab1:
        st.markdown("#### Histogram of Risk Score")
        st.markdown("*With KDE curve and statistical indicators*")
        
        # ============================================
        # YOUR CODE: Histogram
        # ============================================
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='Risk_Score', kde=True, bins=30, 
                    color='skyblue', edgecolor='black', alpha=0.7, ax=ax)
        plt.title('Distribution of Risk Score', fontsize=16, fontweight='bold')
        plt.xlabel('Risk Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Add mean and median lines
        ax.axvline(clean_risk.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {clean_risk.mean():.2f}')
        ax.axvline(clean_risk.median(), color='green', linestyle='--', 
                  linewidth=2, label=f'Median: {clean_risk.median():.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    with viz_tab2:
        st.markdown("#### Box Plot of Risk Score")
        st.markdown("*Shows quartiles, outliers, and data spread*")
        
        # ============================================
        # YOUR CODE: Box Plot
        # ============================================
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, y='Risk_Score', color='lightblue', width=0.4, ax=ax)
        plt.title('Box Plot of Risk Score', fontsize=16, fontweight='bold')
        plt.ylabel('Risk Score', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistical annotations
        Q1 = clean_risk.quantile(0.25)
        Q3 = clean_risk.quantile(0.75)
        IQR = Q3 - Q1
        
        stats_text = f"Min: {clean_risk.min():.2f}\n"
        stats_text += f"Q1: {Q1:.2f}\n"
        stats_text += f"Median: {clean_risk.median():.2f}\n"
        stats_text += f"Q3: {Q3:.2f}\n"
        stats_text += f"Max: {clean_risk.max():.2f}\n"
        stats_text += f"IQR: {IQR:.2f}"
        
        ax.text(1.15, clean_risk.median(), stats_text, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10, verticalalignment='center')
        
        st.pyplot(fig)
        plt.close()
    
    with viz_tab3:
        st.markdown("#### Combined View - Histogram & Box Plot")
        st.markdown("*Side-by-side comparison for comprehensive analysis*")
        
        # Combined visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        sns.histplot(data=df, x='Risk_Score', kde=True, bins=30,
                    color='skyblue', edgecolor='black', alpha=0.7, ax=ax1)
        ax1.axvline(clean_risk.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {clean_risk.mean():.2f}')
        ax1.axvline(clean_risk.median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {clean_risk.median():.2f}')
        ax1.set_title('Distribution of Risk Score (Histogram)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Risk Score', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box Plot
        sns.boxplot(data=df, y='Risk_Score', color='lightblue', width=0.4, ax=ax2)
        ax2.set_title('Box Plot of Risk Score', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Risk Score', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Risk Score Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
    
    # ============================================
    # Quick Metrics
    # ============================================
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📈 Quick Metrics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(clean_risk):,}")
    with col2:
        st.metric("Mean Score", f"{clean_risk.mean():.2f}")
    with col3:
        st.metric("Median Score", f"{clean_risk.median():.2f}")
    with col4:
        st.metric("Std Deviation", f"{clean_risk.std():.2f}")
    with col5:
        outliers = clean_risk[(clean_risk < Q1 - 1.5*IQR) | (clean_risk > Q3 + 1.5*IQR)]
        st.metric("Outliers", len(outliers))
    
    # ============================================
    # Insights
    # ============================================
    st.markdown("---")
    st.markdown('<h3 class="sub-header">💡 Key Insights</h3>', unsafe_allow_html=True)
    
    skewness = clean_risk.skew()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if abs(skewness) < 0.5:
            st.success("✅ **Symmetric Distribution**: The risk score is approximately symmetric.")
        elif skewness > 0:
            st.info("📊 **Right Skewed**: More samples have lower risk scores, with some high outliers.")
        else:
            st.info("📊 **Left Skewed**: More samples have higher risk scores.")
    
    with col2:
        outlier_pct = (len(outliers) / len(clean_risk)) * 100
        if outlier_pct > 10:
            st.warning(f"⚠️ **High Outliers**: {outlier_pct:.1f}% of data points are outliers.")
        else:
            st.success(f"✅ **Normal Outliers**: Only {outlier_pct:.1f}% outliers detected.")


# ==================== OTHER PAGES (Placeholders) ====================
def preprocessing_page():
    st.markdown('<h1 class="main-header">⚙️ Preprocessing</h1>', unsafe_allow_html=True)
    st.info("Preprocessing features will be added here.")
    
    if 'data' in st.session_state and st.session_state.data is not None:
        st.dataframe(st.session_state.data.describe(), use_container_width=True)


def feature_selection_page():
    st.markdown('<h1 class="main-header">🎯 Feature Selection</h1>', unsafe_allow_html=True)
    st.info("Feature selection features will be added here.")


def modeling_page():
    st.markdown('<h1 class="main-header">🤖 Modeling</h1>', unsafe_allow_html=True)
    st.info("Model training features will be added here.")


def cross_validation_page():
    st.markdown('<h1 class="main-header">🔄 Cross Validation</h1>', unsafe_allow_html=True)
    st.info("Cross validation features will be added here.")


def visualization_page():
    st.markdown('<h1 class="main-header">📊 Visualization</h1>', unsafe_allow_html=True)
    st.info("Additional visualizations will be added here.")


# ==================== MAIN ====================
def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Page routing
    pages = {
        "🏠 Home": home_page,
        "⚙️ Preprocessing": preprocessing_page,
        "🎯 Feature Selection": feature_selection_page,
        "🤖 Modeling": modeling_page,
        "🔄 Cross Validation": cross_validation_page,
        "📊 Visualization": visualization_page
    }
    
    pages[page]()


if __name__ == "__main__":
    main()
