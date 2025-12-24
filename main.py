#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Student Grade Prediction Module.

This script processes student data, trains a Linear Regression model to predict
final grades (G3), and generates various Plotly visualizations to analyze the results
and feature importance.
"""
#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.io as pio
import plotly.graph_objects as go

# Set default renderer for Plotly
pio.renderers.default = 'browser'
#%%

def load_and_clean(filename: str) -> pd.DataFrame:
    """
    Loads the student dataset and performs data cleaning and preprocessing.

    Operations performed:
    - Filters rows where G3 (final grade) is greater than 0.
    - Drops intermediate grades 'G1' and 'G2' to prevent data leakage.
    - Encodes binary categorical columns (yes/no) to integers (1/0).
    - Maps specific categorical columns (sex, address, etc.) to binary values.
    - Scales the target variable 'G3' to a percentage (0-100).
    - One-hot encodes remaining categorical variables using get_dummies.

    Args:
        filename (str): Path to the .csv file containing student data.

    Returns:
        pd.DataFrame: A cleaned and preprocessed dataframe ready for analysis.
    """
    df = pd.read_csv(filename, sep=';')
    
    #Filter valid grades and drop intermediate period grades, these choices are explained in README
    df = df.query('G3 > 0')
    df = df.drop(['G1', 'G2'], axis=1)
    
    #Standard binary mapping
    binary_yes_no = [
        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
        'higher', 'internet', 'romantic'
    ]
    
    for col in binary_yes_no:
        df[col] = df[col].replace({'no': 0, 'yes': 1})
    
    # Specific column mapping
    df['sex'] = df['sex'].replace({'F': 0, 'M': 1})        
    df['address'] = df['address'].replace({'U': 0, 'R': 1}) 
    df['famsize'] = df['famsize'].replace({'LE3': 0, 'GT3': 1}) 
    df['Pstatus'] = df['Pstatus'].replace({'T': 0, 'A': 1}) 
    df['school']  = df['school'].replace({'GP': 0, 'MS': 1})
    
    # Convert G3 (0-20 scale) to percentage (0-100)
    df['G3'] = (df['G3'] * 100) / 20
    
    # Separate types for dummy encoding
    catvars = df.select_dtypes(include='object').columns.tolist()
    numvars = df.select_dtypes(exclude='object').columns.tolist()
    
    dummyv = pd.get_dummies(df[catvars], dtype=int)
    
    df_clean = pd.concat([df[numvars], dummyv], axis=1)
    
    return df_clean

#%%

def prepare_data(filename: str, threshold: float):
    """
    Orchestrates the data loading, splitting, feature selection, and scaling.

    Args:
        filename (str): Path to the source CSV file.
        threshold (float): Correlation threshold for feature selection.

    Returns:
        tuple: Contains X_train, X_test, y_train, y_test (all scaled/processed),
               the list of selected features, and the fitted scaler object.
    """
    df_clean = load_and_clean(filename)

    X_all = df_clean.drop('G3', axis=1)
    y_all = df_clean['G3']

    # Split data
    X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Select features based on training data only
    selected_features, _ = get_important_features(X_train_all, y_train, threshold=threshold)
    
    X_train = X_train_all[selected_features]
    X_test = X_test_all[selected_features]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconstruct DataFrames to keep column names and indices
    X_train = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)
    
    return X_train, X_test, y_train, y_test, selected_features, scaler
#%%


def get_important_features(X_train: pd.DataFrame, y_train: pd.Series, threshold: float):
    """
    Selects features based on their Pearson correlation coefficient with the target variable G3.
    
    Uses standard Pearson correlation (r) to measure linear relationships. 
    It calculates the absolute value of r, meaning strong negative correlations 
    are treated as equally important as strong positive ones.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training target values (G3).
        threshold (float): The absolute correlation value required to keep a feature.

    Returns:
        tuple: 
            - selected_features (list): List of column names that meet the threshold.
            - correlations (pd.Series): The correlation series sorted by absolute value.
    """
    
    temp_df = X_train.copy()
    temp_df['G3'] = y_train
    
    correlations = temp_df.corr()['G3'].abs().sort_values(ascending=False)
    selected_features = correlations[correlations > threshold].index.tolist()
    
    if 'G3' in selected_features:
        selected_features.remove('G3')
        
    return selected_features, correlations

#%%


def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Trains a Linear Regression model and evaluates its performance.

    Args:
        X_train (pd.DataFrame): Scaled training features.
        X_test (pd.DataFrame): Scaled testing features.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Testing target values.

    Returns:
        tuple: 
            - lin_reg (sklearn.model): The trained model object.
            - error (float): Mean Absolute Error (MAE).
            - r2 (float): R-squared score.
            - y_test (pd.Series): Actual test values.
            - y_pred (array): Predicted values for the test set.
    """
    lin_reg = LinearRegression().fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return lin_reg, error, r2, y_test, y_pred
#%%

def predict_grade(incoming_data_dict: dict, model, training_columns: list, scaler) -> float:
    """
    Predicts a grade for a single new student entry.

    Ensures the incoming data structure matches the training data by handling
    dummy variables and reindexing missing columns with 0.

    Args:
        incoming_data_dict (dict): Raw data for a single student.
        model (sklearn.model): The trained estimator.
        training_columns (list): List of columns used during training (feature selection).
        scaler (StandardScaler): The scaler fitted on training data.

    Returns:
        float: The predicted grade, clamped between 0 and 100.
    """
    df_new = pd.DataFrame(incoming_data_dict)
    
    # Handle categorical variables
    cat_cols = df_new.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        df_dummies = pd.get_dummies(df_new[cat_cols], dtype=int)
        df_new = df_new.drop(cat_cols, axis=1)
        df_new = pd.concat([df_new, df_dummies], axis=1)

    # Align columns with training data (fill missing dummies with 0)
    df_final = df_new.reindex(columns=training_columns, fill_value=0)
    
    # Scale input
    df_final_scaled = scaler.transform(df_final)
    
    # Predict
    model_guess = model.predict(df_final_scaled)[0]
    
    # Clamp result
    if model_guess < 0: return 0.0
    if model_guess > 100: return 100.0
    
    return model_guess


# %% Visualization Functions

def accuracy_fig(y_test: pd.Series, y_pred) -> go.Figure:
    """
    Creates a scatter plot comparing actual grades vs predicted grades.
    
    Args:
        y_test: Actual target values.
        y_pred: Predicted target values.
        
    Returns:
        plotly.graph_objects.Figure: The scatter plot object.
    """
    fig_model_accuracy = px.scatter(
        x=y_test, 
        y=y_pred,
        labels={'x': "Actual grade", 'y': 'Grade according to model'}, 
        title="Plot over the method's accuracy", 
        color_discrete_sequence=['green']
    )
    # Add a reference line for perfect prediction
    fig_model_accuracy.add_shape(
        type="line", line=dict(dash='dash', color='red'),
        x0=0, y0=0, x1=100, y1=100
    )
    
    return fig_model_accuracy

#%%
def grade_fig(df: pd.DataFrame) -> go.Figure:
    """
    Creates a histogram showing the distribution of final grades (G3).

    Args:
        df (pd.DataFrame): Dataframe containing student data, must include 'G3'.

    Returns:
        go.Figure: A Plotly histogram showing the grade distribution.
    """
    fig_g3_dist = px.histogram(
        df, 
        x='G3', 
        nbins=20, 
        title='Distribution of grades (G3)',
        labels={'G3': 'Grade (%)', 'count': 'Number of students'},
        color_discrete_sequence=['green'] 
        )
    fig_g3_dist.update_layout(bargap=0.1, yaxis_title="Number of students")
    
    return fig_g3_dist

#%%

def feature_fig(X: pd.DataFrame, model: LinearRegression) -> go.Figure:
    """
    Creates a bar chart visualizing the coefficients (impact) of each feature.

    Args:
        X (pd.DataFrame): The feature set (training data) used for the model.
        model (LinearRegression): The fitted Linear Regression model containing .coef_.

    Returns:
        go.Figure: A bar chart sorted by the absolute impact of features.
    """
    coef_df = pd.DataFrame({'Feature': X.columns, 'Impact': model.coef_})
    coef_df = coef_df.sort_values(by='Impact', ascending=False)
    
    fig_feature = px.bar(
        coef_df, 
        x='Impact', 
        y='Feature', 
        title='What variables impact the grade most?',
        labels={'Impact': 'Coefficent value (impact)', 'Feature': 'Variable'},
        text_auto='.2f', 
        color='Impact', 
        height=700, 
        color_continuous_scale=['red', 'white', 'green']
    )
    fig_feature.update_layout(yaxis=dict(autorange="reversed"))
    
    return fig_feature

#%%

def heatmap_fig(threshold: float, selected_features: list, df_clean: pd.DataFrame) -> go.Figure:
    """
    Creates a correlation heatmap for the selected features against G3.
    Correlations below the threshold are masked to 0 for clarity.

    Args:
        threshold (float): The minimum absolute correlation required to display a value.
        selected_features (list): A list of column names (strings) to include in the map.
        df_clean (pd.DataFrame): The complete dataframe containing features and G3.

    Returns:
        go.Figure: A heatmap visualization of the correlation matrix.
    """
    heatmap_cols = selected_features + ['G3'] 
    corr_matrix = df_clean[heatmap_cols].corr().round(2)
    
    # Mask low correlations
    corr_matrix[corr_matrix.abs() < threshold] = 0
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,            
        aspect="auto",              
        title=f"Heatmap over Selected Features vs Grade (Correlations < {threshold} set to 0)",
        color_continuous_scale=['red', 'white', 'green'],
        zmin=-1, zmax=1            
    )
    fig_corr.update_layout(height=800)
    
    return fig_corr
#%%

def box_fig(df: pd.DataFrame, x_col: str = 'sex') -> go.Figure:
    """
    Creates a box plot to compare grade distributions across a category.

    Args:
        df (pd.DataFrame): The source dataframe.
        x_col (str): The column to group by (default: 'sex').
        
    Returns:
        plotly.graph_objects.Figure: The box plot.
    """
    plot_df = df.copy()

    # Map binary 0/1 back to No/Yes for better visualization labels
    if plot_df[x_col].dtype in ['int64', 'float64'] and plot_df[x_col].nunique() == 2:
        plot_df[x_col] = plot_df[x_col].map({0: 'No', 1: 'Yes'})

    fig_boxplot = px.box(
        plot_df, 
        x=x_col, 
        y='G3',
        title=f"How does '{x_col}' affect Grades?",
        color=x_col,
        points="all",  # IMPORTANT: Shows every individual student as a point
        labels={x_col: x_col.capitalize(), 'G3': 'Final Grade (0-100)'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig_boxplot.update_layout(showlegend=False)
    
    return fig_boxplot


# %% Main Execution Block

if __name__ == '__main__':
    # Define file path and parameters
    data_file = 'student-mat.csv'
    correlation_threshold = 0.06
    
    # Run pipeline
    print("Loading and preparing data...")
    df_full = load_and_clean(data_file)
    X_train, X_test, y_train, y_test, selected_features, scaler = prepare_data(data_file, correlation_threshold)
    
    print(f"Training model with {len(selected_features)} selected features...")
    lin_reg, error, r2, y_test, y_pred = train_model(X_train, X_test, y_train, y_test)
    
    # Output results
    print("-" * 30)
    print("Model Performance:")
    print(f"Mean Absolute Error: {error:.2f}")
    print(f"R2 Score: {r2:.5f}")
    print("-" * 30)

    # Example: Generating a plot (uncomment to view)
    # fig = accuracy_fig(y_test, y_pred)
    # fig.show()