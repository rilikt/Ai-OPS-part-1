# 🚴 ML-Ops | Part 1: Machine Learning – Bike Sharing Prediction

This project was part of **Week 1** of the Arcadia ML-Ops course, focused on understanding the basics of machine learning, data exploration, and building predictive models.

We worked on a real-world regression problem: predicting the number of bike rentals on a given day and hour based on features such as weather, season, time, and more.

---

## 📊 Dataset

We used the [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) from the UCI Machine Learning Repository.

The dataset includes two years of hourly data from a bike rental service, with features such as:
- Date and time
- Weather conditions
- Temperature
- Working day/holiday indicators
- Season and year
- Rental counts

---

## 🔍 Data Exploration

We began by exploring the dataset using **pandas** and **seaborn** to:
- Visualize distributions and trends
- Identify skewness in the data
- Understand feature correlations
- Spot anomalies and data quality issues

This step helped us make informed decisions about feature engineering and preprocessing.

---

## ⚙️ Preprocessing

We performed the following steps:
- Dropped highly correlated or redundant features
- Created new features from existing ones (e.g., extracting hour/day from timestamp)
- Built separate pipelines for numerical and categorical features using **scikit-learn**
- Wrapped the preprocessing pipeline in a reusable function to apply it across estimators

---

## 🤖 Model Training

We trained the following models:
- `LinearRegression`
- `RandomForestRegressor`
- `XGBRegressor`

Each model was tuned using `GridSearchCV` to optimize hyperparameters, and evaluated using multiple regression metrics.

### Tools Used
- **scikit-learn** – preprocessing, pipelines, modeling
- **xgboost** – gradient boosting regressor
- **mlflow** – experiment tracking and logging
- **uv** – dependency management
- **jupyter** – notebook-based development
- **pandas** & **seaborn** – data analysis and visualization

---

## 📈 Experiment Tracking

We used **mlflow** to:
- Log parameters, metrics, and artifacts from each model run
- Compare estimator performance
- Keep track of pipeline versions and tuning results

This made it easier to understand what improved model performance and reproduce our best results.

---

## 🧠 Key Learnings

- Practical differences between regression and classification
- The impact of skewed data on model accuracy
- The value of pipelines for reproducibility
- How MLFlow improves collaboration and tracking

---

## 🚀 Next Steps

In future stages of this ML-Ops course, we’ll build on this foundation by:
- Packaging the training pipeline
- Deploying the model
- Monitoring it in production

---

Let me know if you’d like a matching `requirements.txt`, `.env`, or even a diagram to illustrate your ML pipeline visually!
