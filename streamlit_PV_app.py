#!/usr/bin/env python3
"""
Advanced Universal ML Predictor - Multi-Model Comparison
=========================================================
Upload ANY dataset, select features with checkboxes, and see ALL models at once!
Exactly like the video demo you shared.

Features:
- Upload any CSV dataset
- Select features with checkboxes
- Train 9 models simultaneously
- Display all results in a single view
- Real-time model comparison
- Interactive predictions

Author: ML Team
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Model Comparator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .model-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
    }
    .best-model {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .stCheckbox {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_dataset(uploaded_file):
    """Load and process uploaded dataset"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload CSV or Excel file")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(df, selected_features, target_col):
    """Preprocess data with selected features"""
    
    # Get features and target
    X = df[selected_features].copy()
    y = df[target_col].copy()
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.info(f"Converting categorical columns to numeric: {', '.join(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Remove rows with missing values
    missing_before = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    
    if len(X) < missing_before:
        st.warning(f"Removed {missing_before - len(X)} rows with missing values")
    
    return X, y

def train_all_models(X_train, y_train, X_test, y_test):
    """Train all models and return comprehensive results"""
    
    # Scalers
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models_config = {
        'Random Forest': {
            'model': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'scaled': False,
            'color': '#2ecc71',
            'icon': '🌲'
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'scaled': False,
            'color': '#3498db',
            'icon': '🚀'
        },
        'Linear Regression': {
            'model': LinearRegression(),
            'scaled': True,
            'color': '#e74c3c',
            'icon': '📈'
        },
        'Ridge Regression': {
            'model': Ridge(alpha=1.0),
            'scaled': True,
            'color': '#9b59b6',
            'icon': '🔷'
        },
        'Lasso Regression': {
            'model': Lasso(alpha=1.0, max_iter=10000),
            'scaled': True,
            'color': '#f39c12',
            'icon': '🔶'
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(max_depth=10, random_state=42),
            'scaled': False,
            'color': '#1abc9c',
            'icon': '🌳'
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42),
            'scaled': False,
            'color': '#34495e',
            'icon': '⚡'
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsRegressor(n_neighbors=5),
            'scaled': True,
            'color': '#e67e22',
            'icon': '🎯'
        },
        'Support Vector Machine': {
            'model': SVR(kernel='rbf', C=100, gamma=0.1),
            'scaled': True,
            'color': '#95a5a6',
            'icon': '🔘'
        }
    }
    
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, config) in enumerate(models_config.items()):
        status_text.text(f"Training {name}... ({idx+1}/{len(models_config)})")
        
        try:
            model = config['model']
            
            # Train
            if config['scaled']:
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            
            # Metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            mape = np.mean(np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1, y_test))) * 100
            
            # Cross-validation
            try:
                X_cv = X_train_scaled if config['scaled'] else X_train
                cv_scores = cross_val_score(model, X_cv, y_train, cv=min(5, len(X_train)//2), 
                                           scoring='r2', n_jobs=-1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = r2_train
                cv_std = 0
            
            results[name] = {
                'model': model,
                'scaler': scaler if config['scaled'] else None,
                'scaled': config['scaled'],
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'color': config['color'],
                'icon': config['icon']
            }
            
        except Exception as e:
            st.warning(f"Could not train {name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(models_config))
    
    progress_bar.empty()
    status_text.empty()
    
    return results, scaler

def display_all_models_results(results, y_test, target_col):
    """Display all model results simultaneously (like in the video)"""
    
    st.markdown("---")
    st.markdown("## 📊 All Models Performance Comparison")
    
    # Sort by R² score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2_test'], reverse=True)
    
    # Best model highlight
    best_model_name, best_model_data = sorted_results[0]
    
    st.markdown(f"""
    <div class="best-model">
        <h3>🏆 Best Model: {best_model_name}</h3>
        <h2>R² Score: {best_model_data['r2_test']:.4f}</h2>
        <p>RMSE: {best_model_data['rmse']:.2f} | MAE: {best_model_data['mae']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all models in grid
    st.markdown("### 📈 Individual Model Performance")
    
    # Create 3 columns for models
    cols_per_row = 3
    for i in range(0, len(sorted_results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < len(sorted_results):
                model_name, model_data = sorted_results[i + j]
                
                with cols[j]:
                    # Model card
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>{model_data['icon']} {model_name}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics in columns
                    metric_cols = st.columns(2)
                    
                    with metric_cols[0]:
                        st.metric("R² Score", f"{model_data['r2_test']:.4f}")
                        st.metric("RMSE", f"{model_data['rmse']:.2f}")
                    
                    with metric_cols[1]:
                        st.metric("MAE", f"{model_data['mae']:.2f}")
                        st.metric("MAPE", f"{model_data['mape']:.2f}%")
                    
                    # Mini actual vs predicted plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test[:50],
                        y=model_data['y_pred_test'][:50],
                        mode='markers',
                        marker=dict(size=6, color=model_data['color']),
                        name='Predictions'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=1),
                        name='Perfect'
                    ))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=False,
                        xaxis_title=f"Actual {target_col}",
                        yaxis_title="Predicted",
                        font=dict(size=9)
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison table
    st.markdown("---")
    st.markdown("### 📋 Detailed Comparison Table")
    
    comparison_df = pd.DataFrame([
        {
            'Model': f"{data['icon']} {name}",
            'R² Score': f"{data['r2_test']:.4f}",
            'RMSE': f"{data['rmse']:.2f}",
            'MAE': f"{data['mae']:.2f}",
            'MAPE (%)': f"{data['mape']:.2f}",
            'CV R² (mean±std)': f"{data['cv_mean']:.3f}±{data['cv_std']:.3f}"
        }
        for name, data in sorted_results
    ])
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Comparison charts
    st.markdown("---")
    st.markdown("### 📊 Visual Comparison")
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('R² Score Comparison', 'RMSE Comparison', 'MAE Comparison')
    )
    
    models = [name for name, _ in sorted_results]
    r2_scores = [data['r2_test'] for _, data in sorted_results]
    rmse_scores = [data['rmse'] for _, data in sorted_results]
    mae_scores = [data['mae'] for _, data in sorted_results]
    colors = [data['color'] for _, data in sorted_results]
    
    fig.add_trace(
        go.Bar(x=models, y=r2_scores, marker_color=colors, name='R²'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=models, y=rmse_scores, marker_color=colors, name='RMSE'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=models, y=mae_scores, marker_color=colors, name='MAE'),
        row=1, col=3
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

def make_predictions(results, input_data, feature_names):
    """Make predictions with all models"""
    
    predictions = {}
    
    for name, model_data in results.items():
        try:
            # Create input dataframe with correct feature names
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            if model_data['scaled']:
                input_scaled = model_data['scaler'].transform(input_df)
                pred = model_data['model'].predict(input_scaled)[0]
            else:
                pred = model_data['model'].predict(input_df)[0]
            
            predictions[name] = max(0, pred)  # Ensure non-negative
        except Exception as e:
            st.warning(f"Prediction error for {name}: {e}")
            predictions[name] = 0
    
    return predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Universal ML Model Comparator</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Upload Dataset → Select Features → Compare All Models Simultaneously!")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Your Dataset",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with your data"
    )
    
    if not uploaded_file:
        # Welcome screen
        st.info("👈 **Upload your dataset** in the sidebar to get started!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📤 Step 1: Upload
            - CSV or Excel file
            - Any number of columns
            - Numeric data preferred
            """)
        
        with col2:
            st.markdown("""
            ### ✅ Step 2: Select
            - Choose target column
            - Select features with checkboxes
            - Configure train/test split
            """)
        
        with col3:
            st.markdown("""
            ### 🚀 Step 3: Compare
            - Train 9 models at once
            - See all results together
            - Make predictions
            """)
        
        st.markdown("---")
        st.markdown("""
        ## 🎯 Supported Use Cases
        
        - 📊 **Solar/Wind Power Prediction**
        - 🏠 **Energy Consumption Forecasting**
        - 💰 **Sales/Revenue Prediction**
        - 📈 **Stock Price Forecasting**
        - 🌡️ **Temperature/Weather Prediction**
        - And any regression problem!
        """)
        
        st.stop()
    
    # Load data
    df = load_dataset(uploaded_file)
    
    if df is None:
        st.stop()
    
    st.success(f"✅ Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
    
    # Display data preview
    with st.expander("👀 Preview Data"):
        st.dataframe(df.head(20), use_container_width=True)
        st.write(f"**Shape**: {df.shape}")
        st.write(f"**Columns**: {', '.join(df.columns.tolist())}")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found! Please upload a dataset with numeric data.")
        st.stop()
    
    # Sidebar: Target selection
    st.sidebar.markdown("### 🎯 Target Column")
    st.sidebar.markdown("*What do you want to predict?*")
    
    # Auto-detect target
    target_keywords = ['power', 'output', 'generation', 'energy', 'target', 
                       'price', 'sales', 'revenue', 'consumption', 'demand']
    
    default_target = None
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in target_keywords):
            default_target = col
            break
    
    if not default_target:
        default_target = numeric_cols[-1]
    
    target_col = st.sidebar.selectbox(
        "Select target column",
        numeric_cols,
        index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
    )
    
    # Feature selection with checkboxes (like in the video!)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ Feature Selection")
    st.sidebar.markdown("*Select features to use for prediction*")
    
    available_features = [col for col in numeric_cols if col != target_col]
    
    # Select all / Deselect all buttons
    col1, col2 = st.sidebar.columns(2)
    select_all = col1.button("Select All")
    deselect_all = col2.button("Deselect All")
    
    # Initialize session state for checkboxes
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = available_features.copy()
    
    if select_all:
        st.session_state.selected_features = available_features.copy()
    if deselect_all:
        st.session_state.selected_features = []
    
    # Checkboxes for each feature
    selected_features = []
    for feature in available_features:
        is_selected = st.sidebar.checkbox(
            feature,
            value=feature in st.session_state.selected_features,
            key=f"feature_{feature}"
        )
        if is_selected:
            selected_features.append(feature)
    
    # Update session state
    st.session_state.selected_features = selected_features
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature!")
        st.stop()
    
    st.sidebar.success(f"✅ {len(selected_features)} features selected")
    
    # Model settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model Settings")
    
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
    random_state = st.sidebar.number_input("Random Seed", 0, 100, 42)
    
    # Preprocess data
    X, y = preprocess_data(df, selected_features, target_col)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.sidebar.info(f"""
    📊 **Data Split**
    - Training: {len(X_train)} samples
    - Testing: {len(X_test)} samples
    - Features: {len(selected_features)}
    """)
    
    # Train button
    st.sidebar.markdown("---")
    train_button = st.sidebar.button("🚀 Train All Models", type="primary", use_container_width=True)
    
    if train_button:
        with st.spinner("Training 9 ML models... Please wait..."):
            results, scaler = train_all_models(X_train, y_train, X_test, y_test)
            
            st.session_state['results'] = results
            st.session_state['scaler'] = scaler
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = selected_features
            st.session_state['target_col'] = target_col
            st.session_state['df'] = df
        
        st.success("✅ All models trained successfully!")
        st.balloons()
    
    # Check if models are trained
    if 'results' not in st.session_state:
        st.info("👈 Click **'🚀 Train All Models'** to train and compare models!")
        
        # Show data statistics
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", len(df))
        col2.metric("Selected Features", len(selected_features))
        col3.metric("Target Column", target_col)
        col4.metric(f"Avg {target_col}", f"{df[target_col].mean():.2f}")
        
        st.dataframe(df[selected_features + [target_col]].describe(), use_container_width=True)
        
        st.stop()
    
    # Retrieve results
    results = st.session_state['results']
    y_test = st.session_state['y_test']
    feature_names = st.session_state['feature_names']
    target_col = st.session_state['target_col']
    df = st.session_state['df']
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 All Models Results", "🔮 Make Predictions", "📈 Detailed Analysis"])
    
    # Tab 1: All models results (like in the video!)
    with tab1:
        display_all_models_results(results, y_test, target_col)
    
    # Tab 2: Make predictions
    with tab2:
        st.markdown("## 🔮 Make Predictions with All Models")
        
        st.markdown("*Adjust input values to see predictions from all models*")
        
        # Create input fields
        input_values = {}
        
        cols = st.columns(3)
        for idx, feature in enumerate(feature_names):
            col_idx = idx % 3
            
            with cols[col_idx]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                if min_val == max_val:
                    input_values[feature] = st.number_input(
                        f"📊 {feature}",
                        value=mean_val,
                        key=f"input_{feature}"
                    )
                else:
                    input_values[feature] = st.slider(
                        f"📊 {feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
        
        # Make predictions
        st.markdown("---")
        input_data = [input_values[f] for f in feature_names]
        predictions = make_predictions(results, input_data, feature_names)
        
        # Display predictions
        st.markdown("### 🎯 Predictions from All Models")
        
        # Create prediction cards
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        cols = st.columns(3)
        for idx, (model_name, pred_value) in enumerate(sorted_preds):
            col_idx = idx % 3
            
            with cols[col_idx]:
                model_data = results[model_name]
                st.markdown(f"""
                <div class="model-card">
                    <h4>{model_data['icon']} {model_name}</h4>
                    <p class="metric-value" style="color: {model_data['color']}">{pred_value:.2f}</p>
                    <p class="metric-label">{target_col}</p>
                    <small>R² Score: {model_data['r2_test']:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Prediction", f"{np.mean(list(predictions.values())):.2f}")
        with col2:
            st.metric("Maximum", f"{max(predictions.values()):.2f}")
        with col3:
            st.metric("Minimum", f"{min(predictions.values()):.2f}")
        with col4:
            st.metric("Std Dev", f"{np.std(list(predictions.values())):.2f}")
        
        # Prediction comparison chart
        fig = go.Figure()
        
        models = list(predictions.keys())
        values = list(predictions.values())
        colors_list = [results[m]['color'] for m in models]
        
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            marker_color=colors_list,
            text=[f"{v:.2f}" for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="<b>Prediction Comparison Across All Models</b>",
            xaxis_title="Model",
            yaxis_title=f"Predicted {target_col}",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Detailed analysis
    with tab3:
        st.markdown("## 📈 Detailed Model Analysis")
        
        # Select model for detailed view
        model_names = list(results.keys())
        selected_model = st.selectbox(
            "Select model for detailed analysis",
            model_names,
            index=0
        )
        
        model_data = results[selected_model]
        
        # Display detailed metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score (Test)", f"{model_data['r2_test']:.4f}")
            st.metric("R² Score (Train)", f"{model_data['r2_train']:.4f}")
        
        with col2:
            st.metric("RMSE", f"{model_data['rmse']:.2f}")
            st.metric("MAE", f"{model_data['mae']:.2f}")
        
        with col3:
            st.metric("MAPE", f"{model_data['mape']:.2f}%")
            st.metric("CV R² (mean)", f"{model_data['cv_mean']:.4f}")
        
        with col4:
            overfit = model_data['r2_train'] - model_data['r2_test']
            st.metric("Overfitting Gap", f"{overfit:.4f}")
            st.metric("CV Std", f"{model_data['cv_std']:.4f}")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=model_data['y_pred_test'],
                mode='markers',
                marker=dict(color=model_data['color'], size=8, opacity=0.6),
                name='Predictions'
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction'
            ))
            fig.update_layout(
                title=f"<b>Actual vs Predicted - {selected_model}</b>",
                xaxis_title=f"Actual {target_col}",
                yaxis_title=f"Predicted {target_col}",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residual distribution
            residuals = y_test.values - model_data['y_pred_test']
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residuals,
                marker_color=model_data['color'],
                nbinsx=30
            ))
            fig.update_layout(
                title="<b>Residual Distribution</b>",
                xaxis_title="Residual (Actual - Predicted)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(model_data['model'], 'feature_importances_'):
            st.markdown("### 🔍 Feature Importance")
            
            importance = model_data['model'].feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance,
                'Importance (%)': importance * 100
            }).sort_values('Importance', ascending=False)
            
            fig = go.Figure(go.Bar(
                x=feature_imp_df['Importance'],
                y=feature_imp_df['Feature'],
                orientation='h',
                marker=dict(color=model_data['color'])
            ))
            
            fig.update_layout(
                title=f"<b>Feature Importance - {selected_model}</b>",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=max(400, len(feature_names) * 30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(feature_imp_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>🤖 Universal ML Model Comparator | Compare 9 Models Simultaneously</p>
        <p>Built with ❤️ using Streamlit & scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
