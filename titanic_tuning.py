import mlflow
import mlflow.sklearn
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up the experiment name
experiment_name = "Titanic_Survival_Prediction"

# Load the Titanic dataset
def load_data():
    print("Loading Titanic dataset...")
    # If you've downloaded the data locally:
    train_data = pd.read_csv('train.csv')
    
    return train_data

# Preprocess the Titanic dataset
def preprocess_data(data):
    # Feature engineering
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Extract titles from names
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Map rare titles to more common ones
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Miss",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare"
    }
    data['Title'] = data['Title'].map(title_mapping)
    
    # Fill missing values for embarked
    data['Embarked'] = data['Embarked'].fillna('S')
    
    # Define features and target
    X = data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = data['Survived']
    
    # Define numerical and categorical features
    numerical_features = ['Age', 'Fare', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Apply preprocessing
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    print(f"Training set size: {X_train_preprocessed.shape[0]}")
    print(f"Validation set size: {X_val_preprocessed.shape[0]}")
    print(f"Test set size: {X_test_preprocessed.shape[0]}")
    
    return X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val, y_test, preprocessor

# Define the objective function for RandomForest model
def objective_random_forest(trial, X_train, X_val, y_train, y_val):
    with mlflow.start_run(nested=True) as child_run:
        # Suggest hyperparameters for RandomForest
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        # Initialize the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        validation_accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(trial.params)
        mlflow.log_metric('validation_accuracy', validation_accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return validation_accuracy

# Define the objective function for GradientBoosting model
def objective_gradient_boosting(trial, X_train, X_val, y_train, y_val):
    with mlflow.start_run(nested=True) as child_run:
        # Suggest hyperparameters for GradientBoosting
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        
        # Initialize the model
        model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=42
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        validation_accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(trial.params)
        mlflow.log_metric('validation_accuracy', validation_accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "gradient_boosting_model")
        
        return validation_accuracy

def main():
    # Set up MLflow experiment
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        if experiment.lifecycle_stage == 'deleted':
            print(f"Restoring deleted experiment: {experiment_name}")
            client.restore_experiment(experiment.experiment_id)
        print(f"Using existing experiment: {experiment_name}")
    else:
        print(f"Creating new experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Load and preprocess data
    data = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_data(data)
    
    # Start a parent run for RandomForest
    with mlflow.start_run(run_name="Titanic_RandomForest_Tuning") as parent_run:
        mlflow.log_param("model_type", "RandomForest")
        
        # Create a partial function with fixed parameters
        from functools import partial
        objective_rf = partial(
            objective_random_forest,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        # Create an Optuna study for RandomForest
        print("Starting hyperparameter tuning for RandomForest...")
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(objective_rf, n_trials=15)
        
        # Fetch experiment and parent run ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        parent_run_id = mlflow.active_run().info.run_id
        
        # Fetch all child runs
        runs_rf = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{parent_run_id}'"
        )
        
        # Prepare data for visualization
        runs_rf['validation_accuracy'] = runs_rf['metrics.validation_accuracy'].astype(float)
        
        # Identify the best hyperparameter set
        best_run_id_rf = runs_rf.loc[runs_rf['validation_accuracy'].idxmax(), 'run_id']
        best_accuracy_rf = runs_rf['validation_accuracy'].max()
        
        # Log best trial parameters and metrics
        mlflow.log_metric("best_validation_accuracy", best_accuracy_rf)
        mlflow.log_param("best_run_id", best_run_id_rf)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot all hyperparameter sets with validation accuracy
        plt.plot(range(len(runs_rf)), runs_rf['validation_accuracy'], marker='o', linestyle='-', color='blue', label='Validation Accuracy')
        
        # Highlight the best validation score with gold color
        best_index = runs_rf.index[runs_rf['run_id'] == best_run_id_rf][0]
        plt.plot(best_index, best_accuracy_rf, 'o', color='gold', markersize=12, label='Best Validation Score')
        
        plt.xticks(range(len(runs_rf)), rotation=45, ha='right')
        plt.xlabel('Trial Number')
        plt.ylabel('Validation Accuracy')
        plt.title('RandomForest: Validation Accuracy vs. Trial (Best in Gold)')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        
        # Save the plot
        rf_plot_path = "rf_validation_accuracy.png"
        plt.savefig(rf_plot_path)
        mlflow.log_artifact(rf_plot_path, "plots")
        print(f"RandomForest plot saved as {rf_plot_path}")
        
        plt.close()
    
    # Start a parent run for GradientBoosting
    with mlflow.start_run(run_name="Titanic_GradientBoosting_Tuning") as parent_run:
        mlflow.log_param("model_type", "GradientBoosting")
        
        # Create a partial function with fixed parameters
        from functools import partial
        objective_gb = partial(
            objective_gradient_boosting,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        # Create an Optuna study for GradientBoosting
        print("Starting hyperparameter tuning for GradientBoosting...")
        study_gb = optuna.create_study(direction='maximize')
        study_gb.optimize(objective_gb, n_trials=15)
        
        # Fetch experiment and parent run ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        parent_run_id = mlflow.active_run().info.run_id
        
        # Fetch all child runs
        runs_gb = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{parent_run_id}'"
        )
        
        # Prepare data for visualization
        runs_gb['validation_accuracy'] = runs_gb['metrics.validation_accuracy'].astype(float)
        
        # Identify the best hyperparameter set
        best_run_id_gb = runs_gb.loc[runs_gb['validation_accuracy'].idxmax(), 'run_id']
        best_accuracy_gb = runs_gb['validation_accuracy'].max()
        
        # Log best trial parameters and metrics
        mlflow.log_metric("best_validation_accuracy", best_accuracy_gb)
        mlflow.log_param("best_run_id", best_run_id_gb)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot all hyperparameter sets with validation accuracy
        plt.plot(range(len(runs_gb)), runs_gb['validation_accuracy'], marker='o', linestyle='-', color='green', label='Validation Accuracy')
        
        # Highlight the best validation score with gold color
        best_index = runs_gb.index[runs_gb['run_id'] == best_run_id_gb][0]
        plt.plot(best_index, best_accuracy_gb, 'o', color='gold', markersize=12, label='Best Validation Score')
        
        plt.xticks(range(len(runs_gb)), rotation=45, ha='right')
        plt.xlabel('Trial Number')
        plt.ylabel('Validation Accuracy')
        plt.title('GradientBoosting: Validation Accuracy vs. Trial (Best in Gold)')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        
        # Save the plot
        gb_plot_path = "gb_validation_accuracy.png"
        plt.savefig(gb_plot_path)
        mlflow.log_artifact(gb_plot_path, "plots")
        print(f"GradientBoosting plot saved as {gb_plot_path}")
        
        plt.close()
    
    # Evaluate best models on test set
    print("\nEvaluating best models on test set...")
    
    # Load and evaluate best RandomForest model
    best_rf_run = mlflow.get_run(best_run_id_rf)
    best_rf_model_uri = f"runs:/{best_run_id_rf}/random_forest_model"
    best_rf_model = mlflow.sklearn.load_model(best_rf_model_uri)
    
    # Load and evaluate best GradientBoosting model
    best_gb_run = mlflow.get_run(best_run_id_gb)
    best_gb_model_uri = f"runs:/{best_run_id_gb}/gradient_boosting_model"
    best_gb_model = mlflow.sklearn.load_model(best_gb_model_uri)
    
    # Evaluate both models on test set
    y_pred_rf = best_rf_model.predict(X_test)
    y_pred_gb = best_gb_model.predict(X_test)
    
    rf_test_accuracy = accuracy_score(y_test, y_pred_rf)
    gb_test_accuracy = accuracy_score(y_test, y_pred_gb)
    
    print(f"Best RandomForest Test Accuracy: {rf_test_accuracy:.4f}")
    print(f"Best GradientBoosting Test Accuracy: {gb_test_accuracy:.4f}")
    
    # Compare models with a bar chart
    plt.figure(figsize=(10, 6))
    
    models = ['RandomForest', 'GradientBoosting']
    validation_scores = [best_accuracy_rf, best_accuracy_gb]
    test_scores = [rf_test_accuracy, gb_test_accuracy]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, validation_scores, width, label='Validation Accuracy')
    plt.bar(x + width/2, test_scores, width, label='Test Accuracy')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: Validation vs Test Accuracy')
    plt.xticks(x, models)
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save the comparison plot
    comparison_plot_path = "model_comparison.png"
    plt.savefig(comparison_plot_path)
    print(f"Model comparison plot saved as {comparison_plot_path}")
    
    # Log the comparison plot to both runs
    with mlflow.start_run(run_id=best_run_id_rf):
        mlflow.log_artifact(comparison_plot_path, "plots")
    
    with mlflow.start_run(run_id=best_run_id_gb):
        mlflow.log_artifact(comparison_plot_path, "plots")
    
    plt.close()

if __name__ == "__main__":
    main()