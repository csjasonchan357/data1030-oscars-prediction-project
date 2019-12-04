# data1030-oscars-prediction-project
### By Jason Chan, Fall 2019
#### Under the guidance of Andras Zsom, DATA1030

Every year, millions of viewers tune in to the Academy Awards, popularly known as the Oscars, to watch the world’s most prestigious film awards ceremony. Hundreds of movies fight for the top spot of earning the Oscar for Best Picture award, but only one film wins. In this machine learning project, I used a data set containing a variety of information on movies to train a model with the goal of predicting next year’s Oscar winner for Best Picture.

This data set contains a variety of movie attributes. These fields contain information such as duration, genre, gross, user and critic reviews, etc, as well as a variety of other film awards, such BAFTA, the Golden Globes, Critics Choice awards, etc. The data was preprocessed, imputed, and ran through stratified k-fold grid search cross validation pipelines to compare models and tune hyperparameters.

The results can be found in the final report, with my model predicting that Roma would win the 2019 Oscar for Best Picture, in line with the established prediction of experts. Unfortunately, Roma was upset by Green Book, due to a variety of possible reasons. Look forward to the application of this model to predicting the 2020 Oscars Best Picture winner, and contact Jason Chan if you're interested in setting up a friendly wager!

The python version used is 3.7.3, and the packages used and their corresponding versions are seen below and in the requirements.txt file. 

**Packages**
* pandas==0.25.0
* matplotlib==3.1.1
* seaborn==0.9.0
* numpy==1.17.1
* sklearn==0.21.3
* xgboost==0.90