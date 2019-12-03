## Model Testing and Hyper Parameter Tuning Results

**Logistic Regression**

Metric|Mean|St Dev
---|---|---
Average Precision|0.35|0.2
Precision|0.64|0.18
Recall|0.5|0.2
F1|0.54|0.18

Best hyperparameters: `C` $= 1.0$

**Random Forest**

Metric|Mean|St Dev
---|---|---
Average Precision|0.1|0.12
Precision|0.33|0.47
Recall|0.08|0.12
F1|0.13|0.19

Best hyperparameters: `max_depth` $= 5.0$, `min_samples_split` $= 2.0$


**XGBoost**

Metric|Mean|St Dev
---|---|---
Average Precision|0.3|0.25
Precision|0.67|0.37
Recall|0.33|0.24
F1|0.43|0.27

Best hyperparameters: `colsample_bytree` $= 0.75$, `max_depth` $= 4$, `min_child_weight` $=2$

**XGBoost No Impute**

Metric|Mean|St Dev
---|---|---
Average Precision|0.39|0.11
Precision|0.69|0.16
Recall|0.58|0.19
F1|0.59|0.12

Best hyperparameters: `colsample_bytree` $= 1.0$, `max_depth` $= 2$, `min_child_weight` $=2$, `learning_rate` $=0.01$, `subsample` $=1.0$.