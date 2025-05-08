import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.impute import SimpleImputer

def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def conf_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("part-1 | Bike | final")
    mlflow.sklearn.autolog(disable=True)

def get_param_grid(estimator):
    if isinstance(estimator, RandomForestRegressor):
        return {
            'estimator__regressor__n_estimators': [100, 150, 200, 250],
            'estimator__regressor__max_depth': [75, 100, 125, None],
            'estimator__regressor__min_samples_split': [4, 6 ,8],
            'estimator__regressor__min_samples_leaf': [1]
        }
    if isinstance(estimator, LinearRegression):
        return {
            'estimator__regressor__n_jobs': [2, 4, 6, 8],
            'estimator__regressor__positive': [True, False],
            'estimator__regressor__fit_intercept': [True, False],
            'estimator__regressor__copy_X': [True, False],
        }
    if isinstance(estimator, XGBRegressor):
        return {
            'estimator__regressor__n_estimators': [500, 600],
            'estimator__regressor__learning_rate': [0.01, 0.05, 0.1],
            'estimator__regressor__max_depth': [3, 6, 9],
            'estimator__regressor__min_child_weight': [20, 50],
            'estimator__regressor__subsample': [0.6, 0.8, 1.0],
            'estimator__regressor__colsample_bytree': [0.6, 0.8, 1.0],
            'estimator__regressor__gamma': [0, 0.1, 0.3, None],
            'estimator__regressor__reg_alpha': [0, 0.1, 1],
            'estimator__regressor__reg_lambda': [1, 1.5, 2],

            # 'estimator__regressor__n_estimators': [500, 550, 600],
            # 'estimator__regressor__min_child_weight': [25, 50,None],
            # 'estimator__regressor__gamma': [10, 25, 50, None],
            # 'estimator__regressor__max_depth': [1, 5, 10, None],
            #
            # 'estimator__regressor__learning_rate': [0.01, 0.02, 0.03, None],
            # 'estimator__regressor__subsample': [0.3, 0.4, 0.5, 0.6, 0.7, None],
            # 'estimator__regressor__colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, None],
            # 'estimator__regressor__colsample_bylevel': [0.3, 0.4, 0.5, 0.6, 0.7, None],
            # 'estimator__regressor__colsample_bynode': [0.3, 0.4, 0.5, 0.6, 0.7, None],
            # 'estimator__regressor__lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, None],
            # 'estimator__regressor__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, None],
        }
    else:
        print("no matching estimator")
        exit(1)


def pre_pipeline(estimator):

    conf_mlflow() #setting up mlflow

    data = pd.read_csv('data/hour.csv') # import the database
    include = ['season', 'yr', 'mnth', 'hr', 'weekday', 'workingday', 'holiday','weathersit', 'temp']
    categorial_features = ['hr', 'mnth', 'weekday', 'season', 'weathersit', 'yr']
    numerical_features = ['temp']

    X = data[include]  # Separating features and target variable
    y = data['cnt']
    X = X.astype({col: 'float64' for col in X.select_dtypes(include='int').columns}) #needed?
    # Creating training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_example = X_train.iloc[:1] # input example for mlflow

    # Transforming categorical features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    ct = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorial_features),
        ],
        remainder = 'passthrough'
    )

    param_grid = get_param_grid(estimator) # get grid depending on estimator

    regr = TransformedTargetRegressor( #log to even out the data skewness
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Pipeline setup
    pipe = Pipeline([('transform', ct), ('estimator', regr)])

    rmsle_scorer = make_scorer(rmsle, greater_is_better=False) # custom scorer for rmsle

    model = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=rmsle_scorer, return_train_score=True, cv=5, n_jobs=-1)

    with mlflow.start_run():
        mlflow.log_param('estimator', estimator.__class__.__name__)
        model.fit(X_train, y_train)

        print('Best Parameters:', model.best_params_)
        print('Best CV score:', model.best_score_)
        results = pd.DataFrame(model.cv_results_)
        y_pred = model.best_estimator_.predict(X_test)
        y_pred = pd.Series(y_pred, index=X_test.index)
        print('R2 score:', r2_score(y_test, y_pred))

        mlflow.log_params(model.best_params_)
        mlflow.sklearn.log_model(model.best_estimator_, "best_model", input_example=input_example)

        mlflow.log_metric("test_r2", r2_score(y_test, y_pred))
        mlflow.log_metric("test_mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_root_msle", np.sqrt(mean_squared_log_error(y_test, y_pred)))
        mlflow.log_metric("best_index", model.best_index_)

        residuals = y_test - y_pred
        sns.scatterplot(x=y_pred, y=residuals, alpha=.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Count')
        plt.ylabel('Predicted')
        plt.savefig('prediction.png')
        mlflow.log_artifact('prediction.png')
        plt.show()
        results.to_csv('pred_vs_actual.csv', index=False)
        mlflow.log_artifact('pred_vs_actual.csv')




# pre_pipeline(LinearRegression())

# pre_pipeline(RandomForestRegressor())

pre_pipeline(XGBRegressor())


# X['hr'] = X['hr'].astype(CategoricalDtype(categories=list(range(24))))
# X['mnth'] = X['mnth'].astype(CategoricalDtype(categories=list(range(1, 13))))
# X['weekday'] = X['weekday'].astype(CategoricalDtype(categories=list(range(7))))
# X['season'] = X['season'].astype(CategoricalDtype(categories=[1, 2, 3, 4]))
# X['weathersit'] = X['weathersit'].astype(CategoricalDtype(categories=[1, 2, 3, 4]))
# X['yr'] = X['yr'].astype(CategoricalDtype(categories=[0, 1]))
#
# X = pd.get_dummies(X, columns=categorial_features, drop_first=True)



# results = pd.DataFrame(model.cv_results_)
# for i, row in results.iterrows():
#         # mlflow.log_params(f"Params for {i}", row['params'])
#         mlflow.log_metric(f"mean_train_score {i}", row['mean_train_score'])
#         mlflow.log_metric(f"mean_test_score {i}", row['mean_test_score'])
#         mlflow.log_metric(f"std_train_score {i}", row['std_train_score'])
#         mlflow.log_metric(f"std_test_score {i}", row['std_test_score'])



# Over and undersampling
# Get column indices of categorical features
# print(Counter(y_train))  # Inspect class distribution
#
# # Drop classes with too few samples (e.g., < 2)
# min_class_count = y_train.value_counts()
# valid_classes = min_class_count[min_class_count >= 2].index
#
# X_train = X_train[y_train.isin(valid_classes)]
# y_train = y_train[y_train.isin(valid_classes)]
# cat_indices = [X_train.columns.get_loc(col) for col in categorial_features]
#
# smote_nc = SMOTENC(categorical_features=cat_indices, k_neighbors=1, random_state=42)
# X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
#
# # Wrap it back into a DataFrame
# X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
# y_train = pd.Series(y_resampled)

# columns = X_train.columns
#
# ros = RandomOverSampler(random_state=42)
# X_train, y_train = ros.fit_resample(X_train, y_train)
#
# X_train = pd.DataFrame(X_train, columns=columns)
# y_train = pd.Series(y_train)
#
# X_train[categorial_features] = X_train[categorial_features].astype('category')


