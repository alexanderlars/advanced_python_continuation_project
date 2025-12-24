#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Student Performance Analytics Dashboard.

This Dash application serves as an interactive dashboard to visualize student
performance data. It includes an Exploratory Data Analysis (EDA) section,
dataset statistics, model evaluation metrics, and a prediction calculator
that estimates a student's final math grade (G3).

Usage:
    Run this script directly to launch the local web server.
    $ python app.py

Attributes:
    HOST (str): The hostname for the local server (default: 'localhost').
    PORT (int): The port number for the local server (default: 8050).

Dependencies:
    - dash: For building the web application.
    - plotly: For generating interactive figures.
    - pandas: For data manipulation.
    - main: Local module containing ML logic (prepare_data, train_model, etc.).
"""

import webbrowser
from threading import Timer
from typing import List, Dict, Any, Tuple

import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.io as pio

# Local application imports
from main import (
    prepare_data,
    train_model,
    predict_grade,
    load_and_clean,
    accuracy_fig,
    grade_fig,
    feature_fig,
    heatmap_fig
)

# --- Configuration ---
pio.renderers.default = 'browser'
HOST: str = 'localhost'
PORT: int = 8050


def open_browser(host: str = 'localhost', port: int = 8050) -> None:
    """Opens the default web browser to the application URL.

    This function is intended to be called via a threading Timer to ensure
    the server has started before the browser attempts to connect.

    Args:
        host (str): The hostname where the server is running. Defaults to 'localhost'.
        port (int): The port number. Defaults to 8050.
    """
    webbrowser.open_new(f'http://{host}:{port}')


# --- Data Loading and Model Training ---
# Note: These operations run at the module level to initialize the app state.
# In a production environment with multiple workers, caching strategies
# (like Flask-Caching) would be preferred over global variables.

# Load dataset
df = load_and_clean('student-mat.csv')

# Prepare features and target variables
# Split ratio: 0.06 is used for testing in this specific configuration
X_train, X_test, y_train, y_test, selected_features, scaler = prepare_data('student-mat.csv', 0.06)

# Train the Linear Regression model
lin_reg, error, r2, y_test, y_pred = train_model(X_train, X_test, y_train, y_test)

# Generate static figures for the dashboard
fig_model_accuracy = accuracy_fig(y_test, y_pred)
fig_g3_dist = grade_fig(df)
fig_features = feature_fig(X_train, lin_reg)
fig_correlation = heatmap_fig(0.1, selected_features, df)


# --- Constants and Mappings ---

feature_map: Dict[str, str] = {
    'Mjob_teacher': "Mother's Job: Teacher",
    'Mjob_health': "Mother's Job: Health Care",
    'Mjob_services': "Mother's Job: Civil Services",
    'Mjob_at_home': "Mother's Job: At Home",
    'Mjob_other': "Mother's Job: Other",
    'Fjob_teacher': "Father's Job: Teacher",
    'Fjob_health': "Father's Job: Health Care",
    'Fjob_services': "Father's Job: Civil Services",
    'Fjob_at_home': "Father's Job: At Home",
    'Fjob_other': "Father's Job: Other",
    'reason_home': "Reason: Close to Home",
    'reason_reputation': "Reason: School Reputation",
    'reason_course': "Reason: Course Preference",
    'reason_other': "Reason: Other",
    'guardian_mother': "Guardian: Mother",
    'guardian_father': "Guardian: Father",
    'guardian_other': "Guardian: Other",
    'school': 'School Choice',
    'sex': 'Gender',
    'age': 'Age',
    'address': 'Urban/Rural Area',
    'famsize': 'Family Size',
    'Pstatus': 'Parents Living Apart',
    'Medu': "Mother's Education Level",
    'Fedu': "Father's Education Level",
    'traveltime': 'Travel Time to School',
    'studytime': 'Weekly Study Time',
    'failures': 'Past Class Failures',
    'schoolsup': 'Extra School Support',
    'famsup': 'Family Educational Support',
    'paid': 'Extra Paid Classes',
    'activities': 'Extra-curricular Activities',
    'nursery': 'Attended Nursery School',
    'higher': 'Wants Higher Education',
    'internet': 'Internet Access at Home',
    'romantic': 'In a Relationship',
    'famrel': 'Quality of Family Relations',
    'freetime': 'Free Time after School',
    'goout': 'Going Out with Friends',
    'Dalc': 'Workday Alcohol Consumption',
    'Walc': 'Weekend Alcohol Consumption',
    'health': 'Current Health Status',
    'absences': 'Number of Absences',
    'Mjob': "Mother's Job",
    'Fjob': "Father's Job",
    'reason': "Reason for School Choice"
}

# Calculate statistics for display
stats_data: List[Tuple[str, float]] = [
    (feature_map['sex'] + " (Male)", df['sex'].mean() * 100),
    (feature_map['school'] + " (Gabriel Pereira)", (1 - df['school'].mean()) * 100),
    (feature_map['address'] + " (Urban)", (1 - df['address'].mean()) * 100),
    (feature_map['higher'], df['higher'].mean() * 100),
    (feature_map['internet'], df['internet'].mean() * 100),
    (feature_map['romantic'], df['romantic'].mean() * 100),
    (feature_map['activities'], df['activities'].mean() * 100),
    (feature_map['Pstatus'], df['Pstatus'].mean() * 100),
    (feature_map['famsize'] + " Bigger than 3", df['famsize'].mean() * 100),
    (feature_map['famsup'], df['famsup'].mean() * 100),
    (feature_map['paid'], df['paid'].mean() * 100)
]

# Identify excluded features for the "Excluded Features" list
all_columns = [col for col in df.columns if col != 'G3']
excluded_features = [col for col in all_columns if col not in selected_features]
list_items = [
    html.Li(
        feature_map.get(col, col), 
        style={'marginBottom': '10px'}
    ) 
    for col in excluded_features
]


# Variables available for the EDA dropdown
eda_variables = [
    'sex', 'address', 'school', 'failures', 'Medu',
    'Walc', 'Dalc', 'goout', 'studytime', 'higher', 'internet'
]
eda_options = [{'label': feature_map.get(var, var), 'value': var} for var in eda_variables]


# --- Helper Functions ---

def create_input_group(label_text: str, d_id: str, d_options: List[Dict[str, Any]], d_value: Any) -> html.Div:
    """Creates a standardized input group containing a label and a dropdown.

    Args:
        label_text (str): The text to display above the dropdown.
        d_id (str): The unique ID for the Dash Dropdown component.
        d_options (List[Dict[str, Any]]): A list of dictionaries defining the dropdown options
                                          (e.g., [{'label': 'A', 'value': 'a'}]).
        d_value (Any): The default value selected in the dropdown.

    Returns:
        html.Div: A Dash Div component containing the Label and Dropdown.
    """
    return html.Div([
        html.Label(label_text, style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
        dcc.Dropdown(id=d_id, options=d_options, value=d_value)
    ], style={'marginBottom': '20px'})


# --- Dash Application Layout ---

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        'padding': '40px',
        'fontFamily': 'sans-serif',
        'maxWidth': '1000px',
        'margin': '0 auto'
    },
    children=[
        html.H1('Student Performance Analytics',
                style={'textAlign': 'center', 'marginBottom': '40px'}),

        # --- Exploratory Data Analysis Section ---
        html.Div([
            html.H2('Exploratory Data Analysis', style={'borderBottom': '2px solid green', 'paddingBottom': '10px'}),
            html.Label("Select a variable to analyze vs Grade:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='eda-dropdown',
                options=eda_options,
                value='failures',
                clearable=False,
                style={'marginBottom': '20px'}
            ),
            dcc.Graph(id='dynamic-eda-graph'),
            html.H3("Overall Grade Distribution", style={'marginTop': '30px'}),
            dcc.Graph(id='dist-graph', figure=fig_g3_dist),
        ], style={'marginBottom': '50px'}),

        # --- Dataset Statistics Section ---
        html.Div([
            html.H2('Dataset Statistics', style={'borderBottom': '2px solid green', 'paddingBottom': '10px'}),
            html.P("Overview of the data averages:"),
            html.Ul(
                children=[
                    html.Li([
                        html.Strong(f"{label}: "), f"{value:.1f}%"
                    ], style={'marginBottom': '8px', 'fontSize': '16px'})
                    for label, value in stats_data
                ],
            )
        ], style={'marginBottom': '50px'}),

        # --- Model Evaluation Section ---
        html.Div([
            html.H2('Model Evaluation', style={'borderBottom': '2px solid green', 'paddingBottom': '10px'}),
            html.Div([
                html.P([html.Strong("Mean Absolute Error: "), f"{error:.2f} points"]),
                html.P([html.Strong("R² Score: "), f"{r2:.2f}"]),
            ], style={'padding': '15px', 'marginBottom': '20px', 'borderRadius': '5px'}),

            html.H3("Excluded Features with Pearson's r Correlation"),
            html.Ul(
                children=list_items,
                style={'columnCount': 2, 'fontSize': '16px', 'padding': '20px'}
            ),
            dcc.Graph(id='accuracy-graph', figure=fig_model_accuracy),
            dcc.Graph(id='features-graph', figure=fig_features),
            dcc.Graph(id='corr-graph', figure=fig_correlation),
        ], style={'marginBottom': '50px'}),

        # --- Prediction Calculator Section ---
        html.Div([
            html.H2('Grade Calculator', style={'borderBottom': '2px solid green', 'paddingBottom': '10px'}),
            html.P("Enter student details below to predict the final math grade.", style={'marginBottom': '20px'}),

            html.Div([
                # Column 1: Personal & Family
                html.Div([
                    html.H3("Student & Family", style={'fontSize': '20px', 'color': 'green', 'borderBottom': '2px solid green', 'paddingBottom': '10px', 'marginBottom': '20px'}),

                    create_input_group(feature_map['school'], 'school', [{'label': 'Gabriel Pereira', 'value': 0}, {'label': 'Mousinho da Silveira', 'value': 1}], 0),
                    create_input_group(feature_map['sex'], 'sex', [{'label': 'Female', 'value': 0}, {'label': 'Male', 'value': 1}], 0),
                    create_input_group(feature_map['age'], 'age', [{'label': str(i), 'value': i} for i in range(15, 23)], 18),
                    create_input_group(feature_map['address'], 'address', [{'label': 'Urban', 'value': 0}, {'label': 'Rural', 'value': 1}], 0),
                    create_input_group(feature_map['internet'], 'internet', [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], 1),
                    create_input_group(feature_map['romantic'], 'romantic', [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], 0),
                    create_input_group(feature_map['Medu'], 'Medu', [{'label': str(i), 'value': i} for i in range(0, 5)], 0),
                    create_input_group(feature_map['Fedu'], 'Fedu', [{'label': str(i), 'value': i} for i in range(0, 5)], 0),
                    create_input_group(feature_map['Mjob'], 'Mjob', [{'label': 'At Home', 'value': 'at_home'}, {'label': 'Health Care', 'value': 'health'}, {'label': 'Other', 'value': 'other'}, {'label': 'Civil Services', 'value': 'services'}, {'label': 'Teacher', 'value': 'teacher'}], 'other'),
                    create_input_group(feature_map['Fjob'], 'Fjob', [{'label': 'At Home', 'value': 'at_home'}, {'label': 'Health Care', 'value': 'health'}, {'label': 'Other', 'value': 'other'}, {'label': 'Civil Services', 'value': 'services'}, {'label': 'Teacher', 'value': 'teacher'}], 'other'),
                    create_input_group(feature_map['reason'], 'reason', [{'label': 'Close to Home', 'value': 'home'}, {'label': 'Reputation', 'value': 'reputation'}, {'label': 'Course Preference', 'value': 'course'}, {'label': 'Other', 'value': 'other'}], 'course'),

                ], style={'flex': 1, 'padding': '20px', 'minWidth': '300px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),

                # Column 2: Academic & Lifestyle
                html.Div([
                    html.H3("Academic & Lifestyle", style={'fontSize': '20px', 'color': 'green', 'borderBottom': '2px solid green', 'paddingBottom': '10px', 'marginBottom': '20px'}),

                    create_input_group(feature_map['higher'], 'higher', [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], 1),
                    create_input_group(feature_map['studytime'], 'studytime', [{'label': 'Low (<2h)', 'value': 1}, {'label': 'Moderate', 'value': 1}, {'label': 'High', 'value': 3}, {'label': 'Very High (>10h)', 'value': 4}], 1),
                    create_input_group(feature_map['traveltime'], 'traveltime', [{'label': '<15 min', 'value': 1}, {'label': '15-30 min', 'value': 1}, {'label': '30-60 min', 'value': 3}, {'label': '>1 hour', 'value': 4}], 1),
                    create_input_group(feature_map['failures'], 'failures', [{'label': str(i), 'value': i} for i in range(0, 5)], 0),
                    create_input_group(feature_map['schoolsup'], 'schoolsup', [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], 0),
                    create_input_group(feature_map['famsup'], 'famsup', [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], 1),
                    create_input_group(feature_map['activities'], 'activities', [{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], 1),
                    create_input_group(feature_map['goout'], 'goout', [{'label': str(i), 'value': i} for i in range(1, 6)], 1),
                    create_input_group(feature_map['Dalc'], 'Dalc', [{'label': str(i), 'value': i} for i in range(1, 6)], 1),
                    create_input_group(feature_map['Walc'], 'Walc', [{'label': str(i), 'value': i} for i in range(1, 6)], 1),
                    create_input_group(feature_map['absences'], 'absences', [{'label': str(i), 'value': i} for i in range(0, 31)], 0),
                    create_input_group(feature_map['health'], 'health', [{'label': str(i), 'value': i} for i in range(1, 6)], 1),

                ], style={'flex': 1, 'padding': '20px', 'minWidth': '300px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),

            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '40px', 'justifyContent': 'center'}),

            html.Div([
                html.Button(
                    'Calculate Grade',
                    id='submit-btn',
                    n_clicks=0,
                    style={'backgroundColor': 'green', 'color': 'white', 'padding': '15px 30px', 'borderRadius': '5px', 'fontSize': '18px', 'cursor': 'pointer', 'width': '100%', 'border': 'none'}
                ),
                html.Div(id='respons', style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#e8f5e9', 'color': 'green', 'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'borderRadius': '5px'})
            ], style={'maxWidth': '600px', 'margin': '40px auto'})

        ], style={'padding': '40px', 'fontFamily': 'Arial, sans-serif'})
    ]
)


# --- Callbacks ---

@app.callback(
    Output('dynamic-eda-graph', 'figure'),
    Input('eda-dropdown', 'value')
)
def update_graph(selected_variable: str) -> Any:
    """Updates the Exploratory Data Analysis graph based on dropdown selection.

    This callback creates a copy of the main dataframe, maps the raw categorical
    or binary values to human-readable labels (e.g., 0/1 to Female/Male), and
    generates a Box Plot comparing the selected feature to the final grade (G3).

    Args:
        selected_variable (str): The column name selected from the EDA dropdown.

    Returns:
        plotly.graph_objects.Figure: A Plotly Box plot figure visualization.
    """
    plot_df = df.copy()
    nice_name = feature_map.get(selected_variable, selected_variable)

    # Map raw data to readable labels for specific binary/categorical columns
    mappings = {
        'sex': {0: 'Female', 1: 'Male'},
        'address': {0: 'Urban', 1: 'Rural'},
        'school': {0: 'Gabriel Pereira', 1: 'Mousinho da Silveira'},
        'internet': {0: 'No', 1: 'Yes'},
        'higher': {0: 'No', 1: 'Yes'}
    }

    if selected_variable in mappings:
        plot_df[selected_variable] = plot_df[selected_variable].map(mappings[selected_variable])

    # Create the visualization
    fig = px.box(
        plot_df,
        x=selected_variable,
        y='G3',
        color=selected_variable,
        title=f"Impact of '{nice_name}' on Final Grade",
        labels={selected_variable: nice_name, 'G3': 'Final Grade (0-100)'}
    )

    return fig


@app.callback(
    Output('respons', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('school', 'value'),
    State('sex', 'value'),
    State('age', 'value'),
    State('address', 'value'),
    State('internet', 'value'),
    State('higher', 'value'),
    State('Medu', 'value'),
    State('Fedu', 'value'),
    State('Mjob', 'value'),
    State('Fjob', 'value'),
    State('reason', 'value'),
    State('famsup', 'value'),
    State('traveltime', 'value'),
    State('studytime', 'value'),
    State('failures', 'value'),
    State('schoolsup', 'value'),
    State('activities', 'value'),
    State('absences', 'value'),
    State('romantic', 'value'),
    State('goout', 'value'),
    State('Dalc', 'value'),
    State('Walc', 'value'),
    State('health', 'value')
)
def calculate_grade(
    n_clicks: int,
    school: int, sex: int, age: int, address: int, internet: int, higher: int,
    medu: int, fedu: int, mjob: str, fjob: str, reason: str, famsup: int,
    traveltime: int, studytime: int, failures: int, schoolsup: int,
    activities: int, absences: int, romantic: int, goout: int,
    dalc: int, walc: int, health: int
) -> str:
    """Collects inputs from the UI, formats them, and predicts the final grade.

    Triggered by the 'Calculate Grade' button. It aggregates all State values
    from the Dash layout into a dictionary, which is passed to the prediction
    logic in the `main` module.

    Args:
        n_clicks (int): Number of times the submit button has been clicked.
        school (int): School ID (0: GP, 1: MS).
        sex (int): Gender (0: Female, 1: Male).
        age (int): Age of the student.
        address (int): Living area (0: Urban, 1: Rural).
        internet (int): Internet access (0: No, 1: Yes).
        higher (int): Wants higher education (0: No, 1: Yes).
        medu (int): Mother's education level (0-4).
        fedu (int): Father's education level (0-4).
        mjob (str): Mother's job category.
        fjob (str): Father's job category.
        reason (str): Reason for school choice.
        famsup (int): Family educational support (0: No, 1: Yes).
        traveltime (int): Travel time category (1-4).
        studytime (int): Weekly study time category (1-4).
        failures (int): Number of past class failures.
        schoolsup (int): Extra educational support (0: No, 1: Yes).
        activities (int): Extra-curricular activities (0: No, 1: Yes).
        absences (int): Number of school absences.
        romantic (int): In a romantic relationship (0: No, 1: Yes).
        goout (int): Going out frequency (1-5).
        dalc (int): Workday alcohol consumption (1-5).
        walc (int): Weekend alcohol consumption (1-5).
        health (int): Current health status (1-5).

    Returns:
        str: A formatted string displaying the calculated grade (e.g.,
        "Calculated Grade: 75.0%") or a prompt to enter details if the
        button has not been clicked yet.
    """
    if n_clicks > 0:
        # Construct input dictionary matching model expectations
        input_data = {
            'school': [school],
            'sex': [sex],
            'age': [age],
            'address': [address],
            'internet': [internet],
            'higher': [higher],
            'Medu': [medu],
            'Fedu': [fedu],
            'Mjob': [mjob],
            'Fjob': [fjob],
            'reason': [reason],
            'famsup': [famsup],
            'traveltime': [traveltime],
            'studytime': [studytime],
            'failures': [failures],
            'schoolsup': [schoolsup],
            'activities': [activities],
            'absences': [absences],
            'romantic': [romantic],
            'goout': [goout],
            'Dalc': [dalc],
            'Walc': [walc],
            'health': [health]
        }

        # Predict using the imported logic
        grade = predict_grade(input_data, lin_reg, X_train.columns, scaler)

        return f"Calculated Grade: {grade:.1f}%"

    return "Enter your details and click Calculate."


# --- Main Execution Block ---

if __name__ == '__main__':
    # Open browser automatically after a short delay
    Timer(1, open_browser, args=[HOST, PORT]).start()

    # Start the Dash server
    app.run(debug=True, host=HOST, port=PORT)