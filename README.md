# Student Performance Analytics Dashboard
An interactive web application built to analyze student performance data and predict final math grades. This project was developed as part of the **Lund University Finance Society (LINC) Advanced Python Workshop (HT 25)**.

The project uses **Linear Regression** to identify the most important factors for higher grades in maths and provides an model to estimate student grades based on demographic and lifestyle inputs.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Screenshots](#screenshots)

---
## Project Overview
The goal of this project is to analyze the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) and answer the question: **"Can we predict a student's final math grade (G3) based on their background, lifestyle, and study habits, without relying on earlier grades?"**







## Data Source
The dataset used is the **Student Performance Data Set** from the UCI Machine Learning Repository.
* **Source:** [UCI Machine Learning Repository - Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
* **Context:** The data attributes include student grades, demographic, social, and school-related features collected from two secondary schools in Portugal.

<details>
<summary><strong>Click here to view the full Data Dictionary (Variable Descriptions)</strong></summary>
<br>

| Variable | Description | Type/Range |
| :--- | :--- | :--- |
| **school** | Student's school | Binary: 'GP' (Gabriel Pereira) or 'MS' (Mousinho da Silveira) |
| **sex** | Student's sex | Binary: 'F' (Female) or 'M' (Male) |
| **age** | Student's age | Numeric: 15 to 22 |
| **address** | Home address type | Binary: 'U' (Urban) or 'R' (Rural) |
| **famsize** | Family size | Binary: 'LE3' (≤3) or 'GT3' (>3) |
| **Pstatus** | Parent's cohabitation status | Binary: 'T' (Together) or 'A' (Apart) |
| **Medu** | Mother's education | 0 (None) to 4 (Higher Education) |
| **Fedu** | Father's education | 0 (None) to 4 (Higher Education) |
| **Mjob** | Mother's job | Nominal (e.g., teacher, health, services) |
| **Fjob** | Father's job | Nominal (e.g., teacher, health, services) |
| **reason** | Reason to choose this school | Nominal (home, reputation, course, other) |
| **guardian** | Student's guardian | Nominal (mother, father, other) |
| **traveltime** | Home to school travel time | 1 (<15 min) to 4 (>1 hour) |
| **studytime** | Weekly study time | 1 (<2 hours) to 4 (>10 hours) |
| **failures** | Past class failures | Numeric (n if 1≤n<3, else 4) |
| **schoolsup** | Extra educational support | Binary: Yes/No |
| **famsup** | Family educational support | Binary: Yes/No |
| **paid** | Extra paid classes | Binary: Yes/No |
| **activities** | Extra-curricular activities | Binary: Yes/No |
| **nursery** | Attended nursery school | Binary: Yes/No |
| **higher** | Wants to take higher education | Binary: Yes/No |
| **internet** | Internet access at home | Binary: Yes/No |
| **romantic** | With a romantic relationship | Binary: Yes/No |
| **famrel** | Quality of family relationships | 1 (Very Bad) to 5 (Excellent) |
| **freetime** | Free time after school | 1 (Very Low) to 5 (Very High) |
| **goout** | Going out with friends | 1 (Very Low) to 5 (Very High) |
| **Dalc** | Workday alcohol consumption | 1 (Very Low) to 5 (Very High) |
| **Walc** | Weekend alcohol consumption | 1 (Very Low) to 5 (Very High) |
| **health** | Current health status | 1 (Very Bad) to 5 (Very Good) |
| **absences** | Number of school absences | Numeric: 0 to 93 |
| **G1** | First period grade | *Excluded from model (Data Leakage)* |
| **G2** | Second period grade | *Excluded from model (Data Leakage)* |
| **G3** | Final grade (Target) | Numeric: 0 to 20 (Scaled to 0-100%) |

</details>



## Screenshots

### 1. Exploratory Data Analysis (EDA) & Statistics
*Overview of the interactive dashboard, variable exploration, and dataset statistics.*

![EDA Overview](images/exploratory_data_analysis.png)
![EDA Options Dropdown](images/exploratory_data_analysis_options.png)
![Dataset Statistics](images/dataset_statistics.png)
![Grade Distribution](images/grade_distribution.png)

### 2. Model Evaluation & Insights
*Visualizations showing model accuracy, feature importance, and correlation matrices.*

![Model Metrics](images/model_evaluation.png)
![Feature Importance](images/model_feature_importance.png)
![Correlation Matrix](images/correlation_matrix.png)
![Accuracy Scatter Plot](images/scatter_plot_accuracy.png)

### 3. Grade Prediction Calculator
*The interactive tool where users input student data to receive a predicted grade.*

![Calculator Tool](images/grade_calculator.png)
![Calculator with Input Values](images/grade_calculator_with_values.png)
