"""
Script de treinamento do modelo de marketing bancário.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

RANDOM_STATE = 42

def load_and_preprocess_data(filepath):
    """Carrega e preprocessa o dataset."""
    print("📂 Carregando dataset...")
    df = pd.read_csv(filepath, sep=";")

    # Remover colunas com vazamento
    cols_drop = ['duration', 'pdays', 'previous', 'contact']
    df = df.drop(columns=cols_drop)

    print(f"✅ Dataset carregado: {df.shape}")
    print(f"   Distribuição target: {df['y'].value_counts().to_dict()}")

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("y_yes", axis=1)
    y = df_encoded["y_yes"]

    feature_columns = X.columns.tolist()

    return X, y, feature_columns

def train_models(X_train, y_train, X_test, y_test):
    """Treina múltiplos modelos e retorna métricas."""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'NaiveBayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=RANDOM_STATE)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n🤖 Treinando {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        report = classification_report(y_test, pred, output_dict=True)
        true_class = report['True']

        results.append({
            'Model': name,
            'Precision': true_class['precision'],
            'Recall': true_class['recall'],
            'F1-Score': true_class['f1-score'],
            'Accuracy': report['accuracy']
        })

        trained_models[name] = model

    return pd.DataFrame(results), trained_models

def main():
    # Carregar dados
    X, y, features = load_and_preprocess_data('data/bank-additional-full.csv')

    # Split
    print("\n🔀 Dividindo dados (70% treino / 30% teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # SMOTE
    print("\n⚖️  Aplicando SMOTE no conjunto de treino...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"   Distribuição após SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

    # Padronização
    print("\n📏 Padronizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Treinar modelos
    print("\n" + "="*60)
    print("🚀 INICIANDO TREINAMENTO DOS MODELOS")
    print("="*60)

    metrics_df, trained_models = train_models(
        X_train_scaled, y_train_res, X_test_scaled, y_test
    )

    # Exibir resultados
    print("\n" + "="*60)
    print("📊 RESULTADOS FINAIS")
    print("="*60)
    print(metrics_df.sort_values('F1-Score', ascending=False).to_string(index=False))

    # Melhor modelo
    best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]

    print(f"\n🏆 Melhor modelo: {best_model_name}")

    # Salvar pipeline
    os.makedirs('models', exist_ok=True)
    pipeline = {
        'model': best_model,
        'scaler': scaler,
        'features': features
    }

    joblib.dump(pipeline, 'models/best_model_pipeline.pkl')
    print(f"\n💾 Pipeline salvo em 'models/best_model_pipeline.pkl'")

if __name__ == "__main__":
    main()
