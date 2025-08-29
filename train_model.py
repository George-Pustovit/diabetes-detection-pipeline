import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import pickle

def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    features = train.columns.drop("Outcome")
    target = 'Outcome'

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]  

    mlflow.set_experiment('Diabetes_v5')

    # Определяем модели и их параметры
    models = [
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 1000,
                'min_samples_split': 2,
                'min_samples_leaf': 5,
                'random_state': 42  
            }
        },
        {
            'name': 'CatBoost',
            'model': CatBoostClassifier,
            'params': {
                'iterations': 100,
                'depth': 10,
                'learning_rate': 0.2,
                'random_state': 42,
                'verbose': False
            }
        },
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression,
            'params': {
                'penalty': 'l2'
            }
        },
        {
            'name': 'LightGBM',
            'model': LGBMClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1
            }
        }
    ]

    for model_config in models:
        run_name = f"{model_config['name']}"
        
        with mlflow.start_run(run_name=run_name) as run:

            run_id = run.info.run_id

            model = model_config['model'](**model_config['params']) # ** разворачивает словарь  
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)


            # Логируем параметры
            mlflow.log_param("model_type", model_config['name'])

            #Через этот цикл ускоряется развертка параметров
            for param_name, param_value in model_config['params'].items():
                mlflow.log_param(param_name, param_value)
            
           # Логируем метрики
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            
            
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('test_size', len(X_test))
            
            mlflow.sklearn.log_model(model, "model")
            
            model_filename = f"models/{model_config['name'].lower()}_{run_id}_model.pkl" #модели присваевается run_id из запуска ml_flow
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)

            print(f"{model_config['name']} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()