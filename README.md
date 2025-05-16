# ML-Ops | Part-1, Machine Learning

This was the assignment for week 1 of the acardia ML-Ops course, focused on learning more about data science, exploration and training our own model.

# Data exploration

We first start off by exploring the dataset which was given to us:
[Link to Data](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
By looking through the Data and creating plots we delevlop a feel for the data, and make observations on important things such as skewness, taht we have to account for later on in training.

# Model training

In this step we start training our actual model to make predictions on our target variable.
Since we are trying to predict the bike count of a rental service on a certain day and hour based on the features available from the dataset, we are working on a regression problem.

We start by dropping highly correlated features, and doing some feature enginerring based on the insights we got during exploration. Then the data is split into a training and testing set and afterwards being put through our numerical and categorical pipelines.

Once we are finished with preprocessing we then start training our estimators on it using GridSearchCV for hyperparamter-tuning.

During this entire step we are logging everything with mlflow to gain insight on what improves the accuracy of our model and keeping track of the paramters used and differences between different estimators.
