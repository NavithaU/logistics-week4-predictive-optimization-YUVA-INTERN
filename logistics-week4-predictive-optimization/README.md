# Week 4: Predictive Modeling and Optimization in Logistics Systems

This project uses a **hypothetical simulated dataset** to predict delivery time and suggest logistics optimization strategies.

## Run in VS Code
1. Extract the ZIP and open the extracted project folder in VS Code.
2. Choose Terminal > New Terminal.
3. Install packages: `python -m pip install -r requirements.txt`
4. Run: `python predict_logistics.py`
5. Review `outputs` and `visualizations`.

## Contents
- `data/logistics_week4_hypothetical.csv`: 500 simulated shipment records
- `predict_logistics.py`: data preparation, model training, cross-validation, evaluation, prediction charts and scenario outputs
- `outputs/`: model metrics, predictions, feature importance, regional summary and optimization scenarios
- `visualizations/`: generated charts
- `Week4_Predictive_Modeling_Optimization_Report.docx`: detailed report

The target variable is `delivery_time_hours`. Models compared are Linear Regression and Random Forest Regression. All records and cost-saving scenarios are hypothetical; results are not real-world guarantees.
