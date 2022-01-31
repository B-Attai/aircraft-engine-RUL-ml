
# Background
  
Source: CMAPSS Dataset

Data Set: FD001 Train trjectories: 100 Test trajectories: 100 Conditions: ONE (Sea Level) Fault Modes: ONE (HPC Degradation)

Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:

1) unit number 2) time, in cycles 3) operational setting 1 4) operational setting 2 5) operational setting 3 6) sensor measurement 1 7) sensor measurement 2 ... 26) sensor measurement 26


## Machine Learning Features

 - Superviesed Learning Task
 - Multiple Regression
 - Univariate Regression
 - Plain Batch Learning
 
 
# Running this Project 

Libraries required:
 - sklearn
 - pandas
 - numpy
 - matplotlib
 - seaborn
 
Run the jupyter notebook titled - 'project-v1.ipynb'
 
 
# Results

After performing EDA, the features that were negligble, were dropped from the analysis before being placed into the model. Four models were tested, using a metric of negative root mean squared error and 7 fold cross validation. This resulted in a training and validation scores of the following: 

Root Mean Squared Error

SVR(epsilon=0.2)  - Training Score: 42.21
SVR(epsilon=0.2)  - Validation Score: 42.09
LinearRegression()  - Training Score: 44.61
LinearRegression()  - Validation Score: 44.39
RandomForestRegressor()  - Training Score: 15.54
RandomForestRegressor()  - Validation Score: 42.38
GradientBoostingRegressor()  - Training Score: 40.07
GradientBoostingRegressor()  - Validation Score: 41.89

RandomForestRegressor, looked promising albeit underfitting. A grid search was performed with the following parameters:

GridSearchCV(cv=5, estimator=RandomForestRegressor(),
             param_grid=[{'bootstrap': [False, True],
                          'max_features': ['auto', 'sqrt', 'log2', None],
                          'n_estimators': [100, 500, 1000]}],
             return_train_score=True, scoring='neg_mean_squared_error')
             
This resulted in the best estimator being:

RandomForestRegressor(max_features='log2', n_estimators=1000)


For retrained best estimator on test set, the results were as follows:

Mean Squared Error:  32.64
r2_score:  0.383

# Interpretation

Based on these results, the mean squared error of 32.64 meant that the model was prediction for an engine failing was off by approximately 32.64 cycles. The model still performs relatively poor on the data set although, after grid searching the model performed better than the baseline models. 

When considering, the data set, the data was skewed with less results above 250 cycles in the data, which may have had an impact on the model prediction. 

# Reflection

Upon further investigations, it was suggested in other reasearch on this dataset, to implement methods such as capping the dataset due to the non-linearity of the declining RUL over time. Additionally, the performance of the model can possible be improved using more indepth deep learning or machine learning analysis. 

Due to the time contstraints these were not evaluated.