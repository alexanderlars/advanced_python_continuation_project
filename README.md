# Student Performance Analytics Dashboard
An interactive web application built to analyze student performance data and predict final math grades. This project was developed as part of the **Lund University Finance Society (LINC) Advanced Python Workshop (HT 25)**.

The project uses **Linear Regression** to identify the most important factors for higher grades in maths and provides a model to estimate student grades based on demographic and lifestyle inputs.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Methodology and Model Choices](#methodology-and-model-choices)
- [Screenshots](#screenshots)
- [Installation and Usage](#installation-and-usage)

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


## Methodology and Model Choices

To create a strong model for a meaningful analysis, several defining choices were made for the project.

### 1. Data Cleaning and Preprocessing
Raw data processing is handled in `main.py`. The following choices were made:

* **Handling Data Leakage:** The original dataset included grades for three periods `G1` (1st period) and `G2` (2nd period) and final grade (`G3`). The choice was made to **remove** `G1` and `G2` because of their very high correlation to `G3`, which would make the analysis boring and not give space for the other, more interesting parameters affecting the final grade.
* **Target Variable Scaling:** The target variable `G3` (originally 0-20) was scaled to a percentage (0-100) to be more intuitive for dashboard users.
* **Filtering:** Students with `G3 = 0` were **removed** as outliers. An unusual amount of students had grade with the value `0`, which diverged from the overall trend seen in the **Overall grade Distribution**. In the context of the Portuguese grading system, a score of 0 typically indicates a student who was **absent from the final exam** or dropped out, rather than a student who attempted the exam and demonstrated zero knowledge. Including these values could damage the model and by removing them (34 values) the R2-score was increased by 25.0% and the MAE decreased by 24.9%. 
  
* **Encoding:**
    * Binary variables (e.g., `sex`, `romantic`) were mapped to 0/1.
    * Categorical variables (e.g., `Mjob`, `reason`) were encoded using `pd.get_dummies` to allow the linear model to interpret them correctly.
      
### 2. Feature Selection Strategy
To prevent overfitting and reduce noise, we employed a correlation-based feature selection method:
* **Metric:** Pearson Correlation Coefficient ($r$).
* **Threshold:** Features with an absolute correlation $|r| < 0.06$ with the target variable were excluded. The excluded features are shown in **Model Evaluation** and seem reasonable, e.g `guardian`, `nursery` or `famsize` should not affect the grade in any major way. The threshold was chosen experimentally to maximize the R2-score. Including variables with small correlation introduced noise that worsened model performance on the test set.

### 3. Model Selection
**Linear Regression** was selected as the primary model for this project.
* **Justification:** The primary goal is *interpretability* and inference. We need to understand *how* specific factors (like study time or absences) impact the grade. Linear Regression provides clear coefficients that map directly to "points gained/lost," which is essential for the "Grade Calculator" feature in the dashboard.
  
* **Preprocessing:** Since the dataset contains features with very different ranges (e.g., `absences` ranging from 0 to 93, while `studytime` ranges only from 1 to 4), unscaled linear regression coefficients would be misleading in a comparison. A unit change in "study time" is far more significant than a unit change in "absences". By applying  ranges only fr ($z = \frac{x - \mu}{\sigma}$) the data is normalized to a mean of 0 and a standard deviation of 1. This ensures that the model coefficients represent the impact of a variable relative to its variance, making the coefficients comparable. This allows us to correctly rank features by importance, as seen in the **Model Feature Importance**.

---

## Screenshots

### 1. Exploratory Data Analysis (EDA) and Statistics
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



## Installation and Usage
To run this application follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/alexanderlars/advanced_python_continuation_project.git](https://github.com/alexanderlars/advanced_python_continuation_project.git)
   cd advanced_python_continuation_project
 
2.**Install dependencies:**
Make sure you have the required libraries installed.
```bash
pip install -r requirements.txt
```
3.**Run the application:**

```bash
python app.py

