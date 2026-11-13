# Predictive Enrollment Analytics Using Machine Learning

A data-driven machine learning and interactive analytics platform built with **Python** and **Streamlit** to predict occupied seats, classify branch demand, and generate longitudinal multi-year enrollment forecasts.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Application Modules](#application-modules)
- [Dataset Structure & Feature Engineering](#dataset-structure--feature-engineering)
- [Machine Learning Models & Performance](#machine-learning-models--performance)
- [Project Structure](#project-structure)
- [Installation Instructions](#installation-instructions)
- [Execution Steps](#execution-steps)
- [Authors](#authors)

---

## Project Overview
Institutional admission and intake planning often suffers when applicant, fee, seat capacity, and historical enrollment records reside in separate files. This system integrates **four enrollment-related CSV datasets** into a unified preprocessing and feature-engineering pipeline, supporting:
1. **Occupied Seat Prediction** (`RandomForestRegressor`)
2. **Branch Demand Classification** (`Low Demand`, `Optimal Demand`, `High Demand`)
3. **Multi-Year Seat Occupancy Forecasting**

---

## Application Modules
The interactive Streamlit application (`app.py`) provides **five core modules**:
1. **Historical and Forecast Analytics:** Explores past enrollment trends, seat utilization, and model-predicted occupancy across institutes and branches.
2. **Multi-Year Forecast Generator:** Projects future applicant demand (with annual growth-rate clipping at $\pm 10\%$) and predicts occupied and remaining seats up to a target year.
3. **Intake Planning and Capacity:** Evaluates seat utilization and remaining capacity to assist institutional intake decisions.
4. **Institute Finder:** Searches and filters institutions, branches, fee structures, and demand tiers.
5. **Model Evaluation Zoo:** Runs 5-Fold Stratified Cross-Validation across candidate classifiers alongside a **Soft Voting Ensemble (`XGBoost` + `CatBoost`)**, reporting accuracy, weighted precision, recall, F1-score, and aggregate confusion matrices.

---

## Dataset Structure & Feature Engineering
Located in the `dataset/` directory:
- `admissions.csv`: Historical admissions records
- `enrollment.csv`: Branch-level enrollment data
- `institute_branch.csv`: Institutional metadata and seat intake capacity
- `FINAL_DATASET.csv`: Consolidated master dataset used for modeling

**Engineered Features:**
- `ApplicantSeatRatio` = $\frac{\text{Applicants}}{\text{MaxSeats} + 10^{-5}}$
- `FeePerSeat` = $\frac{\text{TuitionFee}}{\text{MaxSeats} + 10^{-5}}$
- `TotalAnnualCost` = $\text{TuitionFee} + \text{AnnualFee}$
- `HistoricalOccupiedSeatsLag1`: Previous year's occupied seats grouped by Institute Code and Branch Code
- **Demand Tiers:** High Demand ($\ge 96.0\%$ utilization), Optimal Demand ($85.0\% - 95.9\%$), Low Demand ($< 85.0\%$)

---

## Machine Learning Models & Performance

| Task | Model | Metric | Score |
| :--- | :--- | :--- | :--- |
| **Seat Occupancy Regression** | Random Forest Regressor (`n_estimators=100`) | MAE | **2.42 seats** |
| **Seat Occupancy Regression** | Random Forest Regressor (`n_estimators=100`) | RMSE | **3.20 seats** |
| **Seat Occupancy Regression** | Random Forest Regressor (`n_estimators=100`) | $R^2$ | **0.9949** |
| **Branch Demand Classification** | Random Forest Classifier | Test Accuracy | **98.96%** |
| **Ensemble Classification** | Soft Voting (`XGBoost` + `CatBoost`) | Probability Averaging | Enabled |

---

## Project Structure
```text
predictive-enrollment-analytics/
|-- dataset/
|   |-- admissions.csv
|   |-- enrollment.csv
|   |-- FINAL_DATASET.csv
|   `-- institute_branch.csv
|-- app.py                  # Complete Streamlit UI, preprocessing, ML training & forecasting
|-- requirements.txt        # Python dependencies
|-- .gitignore              # Git ignore rules
`-- README.md               # Project documentation
```

---

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ItzAri-1124/predictive-enrollment-analytics.git
   cd predictive-enrollment-analytics
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Steps

1. Ensure all four CSV files are present inside the `dataset/` folder.
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser to interact with the five modules.

---

## Authors
- **Arihan Chintu**
- **Mahesh Premtej**
- **Kodali Renusri**
- **Katika Akash**

*Department of Computer Science and Engineering, Faculty of Science and Technology, The ICFAI Foundation for Higher Education, Hyderabad, Telangana, India.*