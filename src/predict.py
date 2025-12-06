"""
Script para fazer previsões com o modelo treinado.
"""

import pandas as pd
import joblib

def predict_single(pipeline, customer_data):
    """Faz previsão para um único cliente."""
    # Criar DataFrame
    X_new = pd.DataFrame([customer_data])

    # One-hot encoding
    X_new_encoded = pd.get_dummies(X_new)

    # Reindexar com features do treino
    X_new_encoded = X_new_encoded.reindex(columns=pipeline['features'], fill_value=0)

    # Escalar
    X_new_scaled = pipeline['scaler'].transform(X_new_encoded)

    # Prever
    prediction = pipeline['model'].predict(X_new_scaled)[0]
    proba = pipeline['model'].predict_proba(X_new_scaled)[0]

    return {
        'prediction': 'SIM' if prediction else 'NÃO',
        'probability_no': f"{proba[0]:.2%}",
        'probability_yes': f"{proba[1]:.2%}"
    }

def main():
    # Carregar pipeline
    print("📂 Carregando modelo...")
    pipeline = joblib.load('models/best_model_pipeline.pkl')

    # Exemplo de cliente
    customer = {
        'age': 35,
        'campaign': 1,
        'emp.var.rate': 1.1,
        'cons.price.idx': 93.994,
        'cons.conf.idx': -36.4,
        'euribor3m': 4.857,
        'nr.employed': 5191.0,
        'job': 'technician',
        'marital': 'married',
        'education': 'university.degree',
        'default': 'no',
        'housing': 'yes',
        'loan': 'no',
        'month': 'may',
        'day_of_week': 'mon',
        'poutcome': 'nonexistent'
    }

    # Fazer previsão
    result = predict_single(pipeline, customer)

    print("\n" + "="*50)
    print("🔮 RESULTADO DA PREVISÃO")
    print("="*50)
    print(f"Cliente irá aderir? {result['prediction']}")
    print(f"Probabilidade NÃO: {result['probability_no']}")
    print(f"Probabilidade SIM: {result['probability_yes']}")
    print("="*50)

if __name__ == "__main__":
    main()
