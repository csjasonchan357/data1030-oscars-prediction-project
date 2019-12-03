# Predicting the Oscars
Jason Chan  
M.Sc. Data Science Candidate  
Brown University, Data Science Initiative  
[Github Repository](https://github.com/csjasonchan357/data1030-oscars-prediction-project)

### Introduction

* importance of the oscars
* This data set contains a variety of information and fields about a large number of movies, ranging up until 2018. These fields contain information such as duration, genre, gross, user and critic reviews, etc. There are also fields for a variety of other awards and accolades, such BAFTA, the Golden Globes, Critics Choice awards, etc. In total there are 119 fields, and 1235 movie titles. Additionally, for each Oscar award, there is a field for whether the film was nominated for that particular Oscar award, as well as a field for whether it won the award. This will be the target variable - I can choose a specific Oscar award to train my model towards, such as Best Picture. At this early point, "Best Picture" is the most interesting of the Oscar awards that I could train my model for. However, in the future I can certainly extend this model to cover the variety of other Oscar awards. This problem is worth pursuing because the Oscars are the premier movie accolade, and winning an Oscar is considered the best possible accolade to win in the movie business. Therefore, people spend a significant time discussing and debating about the results of the Oscars. Being able to predict the winner of the Oscars would be an impressive feat.
* Removed a variety of fields, As I mentioned before, there are 1235 movie titles, and 119 fields. I will use all 1235 movie titles, but in my exploration of the fields, I may choose to exclude certain fields, especially if I decide to train my model towards one particular Oscar award. The dataset is relatively well documented, and tells us whether the feature values are numerical, categorical, text, list, or date field. Because of the sheer number of fields, I will not describe each field independently, but a quick overview can be found on the link. I will also likely exclude the fields pertaining to date, since film industry awards are typically given on a yearly basis anyway. There is also a synopsis field, which may lend to interesting NLP analysis. However, that is beyond the scope of this class, and thus I will not pursue it. Perhaps a cool extension to this project would also be to include analysis of movie synopses and see if movie award winnings can be predicted based on the content of the film. Unfortunately, it seems there is no clear documentation for this dataset on the website, other than the given title headings, so there may be some columns that I will have to guess the meaning of, such as: “popularity”, “rate”, and “metascore”. The latter two seem like scores out of 10 and out of 100, respectively. However, the popularity column is a numerical column where the numbers range on the order of several thousand, so I am uncertain about the meaning of the column. Despite the fact that this dataset is online, it seems that, according to the website where I obtained the dataset, there are no existing models and scripts posted publicly using this dataset.
* preprocessed the bar data and categorical data first 
* Academy Awards - world’s most prestigious film festival
* Type of Problem: Binary Classification
* Can be scaled to multi-class model 
* Contains Numerical Info and 20 other Film Awards Info
    * E.g. Golden Globes, BAFTA, Directors Guild, etc.  
* Before Preprocessing Shape: (1235, 69)
    * Containing films from 2000-2018
    * Total Oscar Best Picture Winners: 18
* After Preprocessing Shape: (1235, 1017)



### Exploratory Data Analysis
* Difficulty with EDA
    * Largely categorical data, 
    * Very sparse matrix
* How to encapsulate large amounts of categorical 1’s and 0’s within graphs?
    * Must narrow down question
* Need to consult Film Experts
* Plots
    * Best Picture winners vs Total Awards Won 
    * Best Picture winners vs Total Awards Nominated
    * Best Picture winners vs Important Film Nomination Scatter Matrix
    * Best Picture winners vs IMDB/Metacritic Rating, Gross
    * Award Nominations vs Month

### Methods
In this section, please explain the data preprocessing and ML pipeline you developed. Make sure to discuss which (un)supervised ML models you used, what parameters you tune and the values you try. What metric do you use to evaluate your models’ performance and why? Measure uncertainties due to splitting and due to non-deterministic ML methods you use (e.g., random forest). In general, explain what considerations went into each step of the pipeline.
#### Preprocessing
* Missing Data:
    * Replace missing with -999, so that GridSearchCV won't freak out but we can still use iterative imputer
* Numerical Features
    * E.g. Gross, # of User Reviews, IMDB rating
    * Standard Scaling 
* Oscar Categorical Features (Yes/No)
    * Convert to binary 1/0
* Genre, Nomination and Won Categories
    * Data in the form of
        * Action|Adventure|Sci-Fi
        * Best Song|Best Composer|Best Director|Best Picture
    * Split data, one-hot encoded
* Performed OHE categorical features before I split, due to the complex nature of the data
    * possible source of data leakage, but much less likely than standard scaling of numerical features
* numerical features scaling occured within gridsearchcv pipeline
* missing values (all numerical)

column|fraction of values missing
---|---
metascore|0.023482
gross|0.034008
user_reviews|0.011336
critic_reviews|0.008097
popularity|0.109312
TOTAL MISSING|0.12874493927125505

#### ML Pipeline
* Train Test Split, 20% test set, 80% CV set
* StratifiedKFold split on the 80%, 4 fold, 60% used to train and 20% used to CV
* Preprocessor included two column transformers: 
    * standard scaler for numerical values, 
    * multivariate imputer for missing values
* Models Attempted
    * Logistic Regression
        * Hyperparameters = `{C: np.logspace(-2,4,10)}`
    * Random Forest
        * Hyperparameters = `{max_depth: [  2,   5,  10,  50, 100], min_sample_split:[ 2,  7, 12, 17, 22]}`
    * XGBoost
        * Hyperparameters = `{'colsample_bytree': [0.75, 1.0], 'max_depth':[4,6,8] , 'min_child_weight': [2,5,10]}`
* Metrics Used
    * Cannot use accuracy due to highly imbalanced set (99% majority class)
    * Precision (Percentage of Positive Classifications that were correct, $\frac{tp}{tp+fp}$)
    * Recall (Percentnage of Positive points that were correctly classified, $\frac{tp}{tp+fn}$)
    * Average Precision (Area Under PR-Curve, 1 is best classifier, 0 is worst)
    * F1 (weighted harmonic mean of Precision and Recall)
* Uncertainties from splitting and non-deterministic methods measured by performing multiple iterations of each gridsearch pipeline, averaging metrics over iterations and finding the standard deviation of metrics. 
    

### Results
Discuss how your scores compare to a baseline model (in classification: what is the score if your
prediction is the most populous class; in regression: use the R2 score which returns 0 for a
constant model that predicts the mean of y, and 1 for a perfect regressor). Calculate global
and/or local feature importances and discuss your findings. Translate your results and model
interpretations in the context of the problem. That is, how does your machine learning results fits
into a business/human/academic context. 

* In highly imbalanced classification scenario, cannot discuss baseline model. A baseline model predicts the most frequent class to each data point. My data has a 99% balance. That means that only 1% of the points belong to the positive class. In my case, I have 1217 negatives and 18 positives. 
* If my baseline model predicts the most frequent class (negative) to each data point then my `TP = 0, FP = 0, TN = 1217, FN = 18`. 
* Precision = $\frac{TP}{TP+FP}$ = DIVIDE BY ZERO ERROR. 
* Recall = $\frac{TP}{TP+FN} = 0/18 = 0$.  
* F1-Score = $2\frac{(P\cdot R)}{(P+R)}$ = N/A, because Precision cannot be calculated.

**LOGISTIC REGRESSION RESULTS**

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


**XGBoost RESULTS**

Metric|Mean|St Dev
---|---|---
Average Precision|0.3|0.25
Precision|0.67|0.37
Recall|0.33|0.24
F1|0.43|0.27

Best hyperparameters: `colsample_bytree` $= 0.75$, `max_depth` $= 4$, `min_child_weight` $=2$

**XGBoost No Impute RESULTS**

Metric|Mean|St Dev
---|---|---
Average Precision|0.39|0.11
Precision|0.69|0.16
Recall|0.58|0.19
F1|0.59|0.12

Best hyperparameters: `colsample_bytree` $= 1.0$, `max_depth` $= 2$, `min_child_weight` $=2$, `learning_rate` $=0.01$, `subsample` $=1.0$.


**Global and Local Feature Importances**
**Translationn of Results* 
** Real Life Testing on 2019 data**

### Outlook
The outlook is the place to describe what else you could do to improve the model or the
interpretability, and what are the weak spots of your modeling approach. How would you
improve this model? What additional techniques could you have used? What additional data
could you collect to improve model performance? 

* Large number of dimensions, too many features right now, over 1000
    * dimensionality reduction, use PCA, SelectKBest (F-test, mutual information), TSNE to reduce dimensions and make to possibly improve model and make output more interpretable
    * tune l1 or elastic net more to  
* other forms of resampling
    * tried stratified sampling, perhaps undersampling the majority class or oversampling the minority class? In this case, given that we have very few points in the first place, we may want to oversample, such as using SMOTE (synthetic minority oversampling technique). If wanted to undersample, can use Tomek Links.
    * imblearn package
    * perhaps combine all three? 
* rather than brute force gridsearchcv, due to limited resource constraints (time, computational power), use a wider parameter search ranges, can try randomized search cv. Or just upgrade (in industry), use more servers, distributed computing, stronger processing power (GPUs). 
* missing data
    * failed to get MCAR test to work (return NaN). If can debug, then maybe we can increase efficiency of model by simply dropping missing values if MCAR, or simple imputing them, since multivariate imputation takes a significant portion of time per model train. 
    * can also try reduced features model (reduced_feature_xgb)
* additional data to collect?
    * would need to consult film experts and ask for expert opinion on publically available information that 
    * collect full sets of data from 2019 to add another year's worth of data, before deploying model to predict 2020 results. 
    
### References
https://bigml.com/user/academy_awards/gallery/dataset/5c6886e1eba31d73070017f5
https://sebastianraschka.com/faq/docs/large-num-features.html
https://www.kaggle.com/phunter/xgboost-with-gridsearchcv  
https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost  
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/  
https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
https://github.com/scikit-learn/scikit-learn/issues/2774
https://oscar.go.com/nominees/best-picture

https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
https://www.datacamp.com/community/tutorials/xgboost-in-python
https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
