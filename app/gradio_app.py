"""
Interface Gradio para previsão de adesão à campanha.
"""

import gradio as gr
import pandas as pd
import joblib

# Carregar pipeline
pipeline = joblib.load('../models/best_model_pipeline.pkl')

def predict_campaign(age, job, marital, education, default, housing, loan, 
                     month, day_of_week, campaign, poutcome):
    """Função de previsão para interface Gradio."""

    customer_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'poutcome': poutcome,
        'emp.var.rate': 1.1,  # valores padrão
        'cons.price.idx': 93.994,
        'cons.conf.idx': -36.4,
        'euribor3m': 4.857,
        'nr.employed': 5191.0
    }

    X_new = pd.DataFrame([customer_data])
    X_new_encoded = pd.get_dummies(X_new).reindex(columns=pipeline['features'], fill_value=0)
    X_new_scaled = pipeline['scaler'].transform(X_new_encoded)

    prediction = pipeline['model'].predict(X_new_scaled)[0]
    proba = pipeline['model'].predict_proba(X_new_scaled)[0]

    result = "✅ SIM - Cliente provavelmente irá aderir" if prediction else "❌ NÃO - Cliente provavelmente não irá aderir"
    confidence = f"Confiança: {max(proba):.1%}"

    return f"{result}\n{confidence}"

# Interface
with gr.Blocks(title="Bank Marketing Predictor") as demo:
    gr.Markdown("# 🏦 Previsão de Adesão à Campanha Bancária")
    gr.Markdown("Preencha os dados do cliente para prever se ele irá aderir ao depósito a prazo.")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Idade", value=35)
            job = gr.Dropdown(label="Profissão", choices=['admin.', 'technician', 'services', 'management'], value='admin.')
            marital = gr.Dropdown(label="Estado Civil", choices=['married', 'single', 'divorced'], value='married')
            education = gr.Dropdown(label="Escolaridade", choices=['university.degree', 'high.school', 'basic.9y'], value='university.degree')

        with gr.Column():
            default = gr.Dropdown(label="Inadimplente?", choices=['no', 'yes', 'unknown'], value='no')
            housing = gr.Dropdown(label="Tem Imóvel?", choices=['yes', 'no'], value='yes')
            loan = gr.Dropdown(label="Tem Empréstimo?", choices=['no', 'yes'], value='no')
            campaign = gr.Number(label="Contatos na Campanha", value=1)

    with gr.Row():
        month = gr.Dropdown(label="Mês", choices=['may', 'jun', 'jul', 'aug'], value='may')
        day_of_week = gr.Dropdown(label="Dia da Semana", choices=['mon', 'tue', 'wed', 'thu', 'fri'], value='mon')
        poutcome = gr.Dropdown(label="Resultado Campanha Anterior", choices=['nonexistent', 'success', 'failure'], value='nonexistent')

    output = gr.Textbox(label="Resultado", lines=2)
    btn = gr.Button("🔮 Fazer Previsão")

    btn.click(
        predict_campaign,
        inputs=[age, job, marital, education, default, housing, loan, month, day_of_week, campaign, poutcome],
        outputs=output
    )

demo.launch()
