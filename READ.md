# 🏦 Bank Marketing Campaign - Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Sobre o Projeto

Pipeline completo de Machine Learning para prever se um cliente irá aderir a uma campanha de marketing bancário.

**Dataset**: Bank Marketing (UCI)  
**Melhor Modelo**: SVM com F1-Score de 0.439  
**Técnicas**: SMOTE, StandardScaler, One-Hot Encoding

---

## 🎯 Objetivo

Prever se um cliente irá subscrever um depósito a prazo (`yes` ou `no`).

---

## 📊 Dataset

- **Registros**: 41.188
- **Features**: 17 (após limpeza)
- **Target**: `y` (yes/no)
- **Desbalanceamento**: 11% positivos

### Tratamento
✅ Remoção de features com vazamento (`duration`, `pdays`, `previous`, `contact`)  
✅ One-hot encoding  
✅ Balanceamento com SMOTE  
✅ Padronização com StandardScaler

---

## 🤖 Resultados dos Modelos

| Modelo                | Precision | Recall | F1-Score | Accuracy |
|-----------------------|-----------|--------|----------|----------|
| **SVM** ⭐            | 0.473     | 0.409  | **0.439**| 0.882    |
| XGBoost               | 0.495     | 0.389  | 0.435    | 0.886    |
| KNN                   | 0.317     | 0.430  | 0.365    | 0.831    |
| Naive Bayes           | 0.192     | 0.782  | 0.308    | 0.604    |

---

## 🚀 Como Usar

### 1. Clonar o Repositório
