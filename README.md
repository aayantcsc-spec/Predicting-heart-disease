# ❤️ Heart Disease Analysis & Prediction

### Clinical Prediction Using Machine Learning

An end-to-end machine learning analysis of the **UCI Heart Disease dataset**, covering 920 patient records from four hospital cohorts. The project investigates not only how accurately machine learning can predict heart disease, but also **where that performance breaks down when models are tested across different hospitals**.

> **Portfolio & Teaching Project — Not a Clinical Tool**

---

## 📌 Project Overview

Heart disease prediction is a classic clinical machine learning problem. However, achieving high accuracy on a random train/test split does not necessarily mean that a model will work reliably in a real hospital.

This project focuses on three questions:

1. **Can machine learning predict heart disease from clinical measurements?**
2. **Which clinical variables contribute most to prediction?**
3. **Does a model trained on one hospital population generalize to another?**

The analysis compares six machine learning algorithms and evaluates them using both conventional held-out testing and **cross-site generalization**.

---

## 📊 Key Results

| Metric                 |                  Result |
| ---------------------- | ----------------------: |
| Total Patients         |                 **920** |
| Hospital Cohorts       |                   **4** |
| Disease Prevalence     |               **55.3%** |
| Clinical Features      |                  **13** |
| Best Model             |       **Random Forest** |
| Best Test ROC-AUC      |               **0.912** |
| Best Recall            |               **0.882** |
| Most Missing Variable  | **Major Vessels — 66%** |
| Cross-Site Performance | **Significantly lower** |

### 🏆 Best Model

**Random Forest**

* Cross-validation ROC-AUC: **0.875**
* Test ROC-AUC: **0.912**
* Accuracy: **83.0%**
* Precision: **82.4%**
* Recall: **88.2%**
* F1-score: **85.2%**

---

# 🗂️ Dataset Validation

Before training any model, two candidate datasets were evaluated.

## ❌ Synthetic Dataset

`heart_disease_dataset.csv`

* 500 rows
* 19 columns
* No missing values
* Identified by its README as **synthetic**
* Random Forest ROC-AUC: **0.458**

The synthetic dataset contained little meaningful relationship between the predictors and target. Rather than reporting misleadingly high metrics, it was rejected.

## ✅ UCI Heart Disease Dataset

The final analysis uses the **UCI Heart Disease archive**.

The dataset contains **920 real patient records** collected from four institutions:

* Cleveland Clinic Foundation
* Hungarian Institute of Cardiology, Budapest
* University Hospitals Zurich and Basel
* V.A. Medical Center, Long Beach

The data is incomplete and historically dated, but the relationships between clinical variables and disease status are based on real patient observations.

---

# 🏥 Hospital Cohorts

One of the central findings of this project is that the four cohorts are substantially different.

| Study Site    | Patients | Disease Prevalence |
| ------------- | -------: | -----------------: |
| Cleveland     |      303 |              45.9% |
| Hungary       |      294 |              36.1% |
| Long Beach VA |      200 |              74.5% |
| Switzerland   |      123 |              93.5% |

The prevalence ranges from approximately **36% in Hungary to 94% in Switzerland**.

This matters because a model can appear strong when training and testing data come from the same mixture of hospitals while performing substantially worse on a completely different population.

---

# 🧹 Data Quality & Missing Values

The dataset contains substantial missing information.

The most affected variables are:

| Variable               | Missing Values |
| ---------------------- | -------------: |
| Major Vessels (`ca`)   |        **66%** |
| Thallium Scan (`thal`) |        **53%** |

These variables require additional investigations or procedures and therefore were not recorded consistently across all hospitals.

### Switzerland Data Cleaning

The Switzerland cohort contains cholesterol values recorded as `0` when measurements were unavailable.

These values were converted to **missing values** rather than allowing the model to interpret `0` as a legitimate cholesterol measurement.

---

# 🔍 Exploratory Data Analysis

The exploratory analysis examines relationships between clinical measurements and heart disease.

## Important Findings

### Maximum Heart Rate

Maximum heart rate achieved during exercise is one of the clearest continuous predictors.

Patients with heart disease generally achieve lower maximum heart rates during exercise.

### ST Depression

ST depression also shows a strong relationship with disease status.

### Age

Age provides some separation between disease groups but is considerably weaker than some exercise-related variables.

### Resting Blood Pressure

Resting blood pressure provides relatively weak separation.

This is an important negative finding because intuitive assumptions do not always translate into strong predictive features.

---

# 🫀 Chest Pain Type

One of the most interesting findings is the relationship between **chest pain type and disease prevalence**.

Patients reporting **asymptomatic chest pain** show a surprisingly high disease rate.

This demonstrates why relying only on classic textbook symptoms can miss clinically important cases.

---

# 🔗 Correlation Analysis

The correlation matrix shows that no pair of numeric predictors exceeds approximately:

**|r| = 0.4**

Therefore, severe multicollinearity is not a major concern in this dataset.

This also helps explain why **Logistic Regression remains competitive** with more complex ensemble models.

---

# 🤖 Machine Learning Models

Six classification algorithms were evaluated using a consistent preprocessing and modelling pipeline.

### Models

1. Logistic Regression
2. K-Nearest Neighbours
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. Support Vector Machine (RBF)

All models use the same preprocessing methodology to ensure a fair comparison.

---

# 📈 Model Performance

| Model               |    CV AUC |  Test AUC |  Accuracy | Precision |    Recall |        F1 |
| ------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| **Random Forest**   | **0.875** | **0.912** | **0.830** | **0.824** | **0.882** | **0.852** |
| SVM (RBF)           |     0.875 |     0.905 |     0.826 |     0.818 | **0.882** |     0.849 |
| Gradient Boosting   |     0.864 |     0.899 |     0.804 |     0.793 |     0.874 |     0.832 |
| Logistic Regression |     0.873 |     0.892 |     0.796 |     0.812 |     0.819 |     0.816 |
| KNN                 |     0.869 |     0.888 |     0.813 |     0.800 |     0.882 |     0.839 |
| Decision Tree       |     0.828 |     0.835 |     0.744 |     0.809 |     0.701 |     0.751 |

---

# 🏆 Random Forest

Random Forest achieved the highest held-out ROC-AUC:

> **ROC-AUC = 0.912**

It also achieved a recall of:

> **Recall = 0.882**

For a screening-oriented application, recall is particularly important because failing to identify a patient with disease can be more consequential than generating an additional false positive.

However, the threshold should be selected according to an explicit clinical or operational cost function rather than automatically using `0.5`.

---

# 📉 ROC-AUC

ROC-AUC was used as the primary model comparison metric because it measures the ability of the classifier to distinguish between patients with and without heart disease across different classification thresholds.

An important finding is that the difference between several models is relatively small.

For example:

* Random Forest: **0.912**
* SVM: **0.905**
* Gradient Boosting: **0.899**
* Logistic Regression: **0.892**

This suggests that **data quality and population differences may matter more than simply choosing a more complex algorithm**.

---

# 🧠 Feature Importance

Permutation feature importance was calculated on held-out data.

Unlike impurity-based tree importance, permutation importance measures how much model performance decreases when a feature is randomly shuffled.

### Key Finding

**Chest pain type** was the strongest predictor in the analysis, substantially exceeding the importance of most other variables.

This makes the feature importance analysis easier to interpret because it reflects the effect of disrupting each feature's predictive information on unseen data.

---

# 🌍 Cross-Site Generalization

This is the most important experiment in the project.

A model that performs well on a random split may still fail when deployed at a different hospital.

The project therefore evaluates performance across hospital cohorts.

| Test Cohort   | Patients |   ROC-AUC | Recall @ 0.5 |
| ------------- | -------: | --------: | -----------: |
| Hungary       |      294 | **0.896** |        0.566 |
| Switzerland   |      123 |     0.768 |        0.548 |
| Long Beach VA |      200 |     0.735 |        0.483 |

---

# ⚠️ The Generalization Gap

The random-split evaluation produced a ROC-AUC of:

> **0.912**

But performance drops substantially when the model is evaluated on previously unseen hospital populations.

The model's recall also falls from approximately:

> **0.88 → below 0.55**

for some external cohorts.

This demonstrates that a model can learn characteristics of the training population rather than learning relationships that generalize reliably across hospitals.

---

# 💡 Main Finding

> **A model with 0.91 ROC-AUC on a random split is not automatically a model that will work in a hospital.**

The project therefore moves beyond:

**"Which classifier has the highest accuracy?"**

and instead asks:

**"Where does the model stop being trustworthy?"**

This population-shift problem is one of the most important considerations in clinical machine learning.

---

# 🔬 Methodology

## Data Preparation

* Combined four UCI hospital cohorts
* Converted invalid zero values to missing values where appropriate
* Handled missing numerical variables using median imputation
* Handled categorical variables using mode/appropriate categorical imputation
* Preprocessing performed inside the machine learning pipeline

## Model Training

* Stratified **75/25 train-test split**
* **5-fold stratified cross-validation**
* Six classification algorithms
* Standardized preprocessing
* Held-out test evaluation
* ROC-AUC as the primary comparison metric

## Leakage Prevention

Imputation and scaling were fitted only on the training data/folds.

This prevents information from the validation or test set from leaking into the training process.

---

# 📁 Project Structure

```text
Heart-Disease-Analysis-and-Prediction/
│
├── analysis.py
├── README.md
├── requirements.txt
│
├── data/
│   └── heart_disease_merged.csv
│
├── results/
│   └── model_results.csv
│
└── figs/
    ├── cohort_distribution.png
    ├── missing_values.png
    ├── feature_distributions.png
    ├── categorical_analysis.png
    ├── correlation_matrix.png
    ├── roc_curves.png
    ├── confusion_matrix.png
    ├── threshold_analysis.png
    ├── calibration_curve.png
    └── permutation_importance.png
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Heart-Disease-Analysis-and-Prediction.git
```

Move into the project directory:

```bash
cd Heart-Disease-Analysis-and-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Main Python libraries used:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
ucimlrepo
```

---

# ▶️ Running the Project

The UCI dataset can be downloaded programmatically using:

```python
from ucimlrepo import fetch_ucirepo

heart = fetch_ucirepo(id=45)
```

Then run:

```bash
python analysis.py
```

The analysis generates:

```text
heart_disease_merged.csv
model_results.csv
```

and publication-ready figures inside:

```text
figs/
```

---

# 📊 Outputs

The project produces visualizations covering:

* Hospital cohort distribution
* Disease prevalence
* Missing-value analysis
* Continuous feature distributions
* Categorical feature relationships
* Correlation matrix
* ROC curves
* Confusion matrix
* Classification threshold analysis
* Calibration curve
* Permutation feature importance

---

# 🚀 Future Improvements

## 1. SHAP Explainability

Add SHAP-based explanations for individual predictions.

Instead of:

> Model predicts 81% risk.

the system could provide an interpretable explanation of which patient features contributed most to the prediction.

---

## 2. Leave-One-Site-Out Cross-Validation

Perform systematic cross-site validation:

```text
Train: Cleveland + Hungary + Switzerland
Test:  Long Beach

Train: Cleveland + Hungary + Long Beach
Test:  Switzerland

Train: Cleveland + Switzerland + Long Beach
Test:  Hungary

Train: Hungary + Switzerland + Long Beach
Test:  Cleveland
```

This would provide a more rigorous estimate of hospital-to-hospital generalization.

---

## 3. Cost-Based Threshold Selection

Instead of automatically using a probability threshold of `0.5`, define an explicit cost ratio between:

* False negatives
* False positives

and select the operating threshold accordingly.

---

## 4. Modern External Validation

The UCI Heart Disease dataset was collected in the **1980s**.

Clinical practices, diagnostic criteria, treatments, and population characteristics have changed significantly.

A modern external dataset would therefore be necessary before considering real-world clinical deployment.

---

# ⚠️ Limitations

This project has several important limitations:

* Historical dataset from approximately 1988
* Small overall sample size for modern clinical ML standards
* Significant missing data
* Strong differences between hospital populations
* Selection bias in some cohorts
* No modern external validation
* No prospective clinical validation
* Model performance may not generalize to current patients
* Thresholds were not optimized using a clinical cost function

---

# 🩺 Medical Disclaimer

**This project is for educational, research, and portfolio purposes only.**

It is **not a medical device, diagnostic system, or clinical decision-support tool**.

The predictions generated by this project must **not** be used to diagnose heart disease, make treatment decisions, or inform medical decisions for real patients.

Any clinical application would require extensive external validation, prospective evaluation, regulatory review, clinical oversight, and appropriate safety testing.

---

# 📚 Dataset & References

### UCI Machine Learning Repository — Heart Disease Dataset

The project uses the UCI Heart Disease dataset originally compiled from four institutions:

* Cleveland Clinic Foundation
* Hungarian Institute of Cardiology
* University Hospitals Zurich and Basel
* V.A. Medical Center, Long Beach

Dataset:

**Heart Disease Data Set — UCI Machine Learning Repository**

The original dataset is associated with work by Detrano et al. and collaborators.

---

# 🛠️ Technology Stack

| Technology        | Purpose                   |
| ----------------- | ------------------------- |
| Python            | Programming language      |
| Pandas            | Data manipulation         |
| NumPy             | Numerical computation     |
| Scikit-learn      | Machine learning          |
| Matplotlib        | Visualization             |
| Seaborn           | Statistical visualization |
| UCI ML Repository | Dataset                   |

---

# 👨‍💻 Author

**Aayan Shaikh**

MSc IT Student
Thakur College of Science and Commerce, Mumbai

---

# ⭐ Project Takeaway

The strongest result of this project is not simply the **0.912 ROC-AUC** achieved by Random Forest.

The stronger finding is the **generalization gap**.

A model can look excellent when evaluated using a random split of historical data and still perform considerably worse when exposed to a different hospital population.

This project therefore demonstrates an important principle of clinical machine learning:

> **Good benchmark performance does not automatically mean good real-world performance.**

The goal is not only to build a model that predicts well — but to understand **when, where, and why that model can be trusted.**

---

## 📌 Project Status

**Completed — Portfolio / Educational Research Project**

Future work includes SHAP explanations, systematic leave-one-site-out validation, cost-sensitive threshold optimization, and validation on modern clinical cohorts.
