from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

BASE=Path(__file__).resolve().parent
DATA=BASE/"data"/"logistics_week4_hypothetical.csv"
OUT=BASE/"outputs"; CHARTS=BASE/"visualizations"
OUT.mkdir(exist_ok=True); CHARTS.mkdir(exist_ok=True)
df=pd.read_csv(DATA)
features=["distance_km","shipment_volume_units","shipment_weight_kg","traffic_level","weather_condition","vehicle_type","region","weekend_delivery"]
target="delivery_time_hours"; X=df[features]; y=df[target]
num=["distance_km","shipment_volume_units","shipment_weight_kg","weekend_delivery"]
cat=["traffic_level","weather_condition","vehicle_type","region"]
prep=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median"))]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=120,max_depth=10,min_samples_leaf=2,random_state=42)}
pipes={}; rows=[]
for name,model in models.items():
    pipe=Pipeline([("prep",prep),("model",model)]); pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
    cv=cross_val_score(pipe,X,y,cv=KFold(n_splits=5,shuffle=True,random_state=42),scoring="neg_mean_absolute_error")
    rows.append({"Model":name,"MAE_hours":mean_absolute_error(yte,p),"RMSE_hours":mean_squared_error(yte,p)**.5,"R_squared":r2_score(yte,p),"5_fold_CV_MAE_hours":-cv.mean()}); pipes[name]=pipe
metrics=pd.DataFrame(rows).sort_values("MAE_hours"); metrics.to_csv(OUT/"model_evaluation_metrics.csv",index=False)
best=metrics.iloc[0]["Model"]; pred=pipes[best].predict(Xte)
test=Xte.copy(); test["actual_delivery_time_hours"]=yte; test["predicted_delivery_time_hours"]=pred.round(2); test["absolute_error_hours"]=np.abs(yte-pred).round(2); test.to_csv(OUT/"test_set_predictions.csv",index=False)
plt.figure(figsize=(7,4.5)); plt.scatter(yte,pred,alpha=.6); lim=[min(yte.min(),pred.min()),max(yte.max(),pred.max())]; plt.plot(lim,lim,"--"); plt.xlabel("Actual delivery time (hours)"); plt.ylabel("Predicted delivery time (hours)"); plt.title("Actual vs Predicted Delivery Time"); plt.tight_layout(); plt.savefig(CHARTS/"actual_vs_predicted.png",dpi=140); plt.close()
plt.figure(figsize=(7,4.5)); plt.hist(yte-pred,bins=25,edgecolor="black"); plt.xlabel("Actual - predicted (hours)"); plt.ylabel("Frequency"); plt.title("Prediction Error Distribution"); plt.tight_layout(); plt.savefig(CHARTS/"prediction_error_distribution.png",dpi=140); plt.close()
rf=pipes["Random Forest"]; imp=permutation_importance(rf,Xte,yte,n_repeats=5,random_state=42,scoring="neg_mean_absolute_error")
importance=pd.DataFrame({"feature":features,"importance":imp.importances_mean}).sort_values("importance",ascending=False); importance.to_csv(OUT/"feature_importance.csv",index=False)
plt.figure(figsize=(7,4.5)); tmp=importance.sort_values("importance"); plt.barh(tmp.feature,tmp.importance); plt.xlabel("Decrease in score when feature is shuffled"); plt.title("Random Forest Feature Importance"); plt.tight_layout(); plt.savefig(CHARTS/"feature_importance.png",dpi=140); plt.close()
reg=df.groupby("region").agg(shipments=("delivery_time_hours","count"),avg_delivery_time_hours=("delivery_time_hours","mean"),avg_transport_cost_inr=("transport_cost_inr","mean")).reset_index(); reg.to_csv(OUT/"region_performance_summary.csv",index=False)
baseline=df.transport_cost_inr.sum(); scenarios=pd.DataFrame([{"Scenario":"Baseline","Assumed_cost_reduction_pct":0},{"Scenario":"Route planning improvement","Assumed_cost_reduction_pct":5},{"Scenario":"Route planning + load consolidation","Assumed_cost_reduction_pct":8}]); scenarios["Estimated_total_cost_inr"]=baseline*(1-scenarios.Assumed_cost_reduction_pct/100); scenarios["Estimated_savings_inr"]=baseline-scenarios.Estimated_total_cost_inr; scenarios.to_csv(OUT/"optimization_scenarios.csv",index=False)
print("Rows:",len(df)); print(metrics.to_string(index=False)); print("Best model by lowest MAE:",best); print("Charts saved in visualizations; tables saved in outputs.")
