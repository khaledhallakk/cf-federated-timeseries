# 🧠 Catastrophic Forgetting in Federated Time Series Forecasting

This repository provides a **reproducible benchmark framework** for evaluating *catastrophic forgetting mitigation methods* in **federated time-series forecasting**.  
All experiments are performed using the **Beijing Multi-Site Air Quality Dataset**, where each monitoring station acts as an independent **federated client**.

📄 If you use this work in your research, please cite:
Hallak, K., Kem, O., Benchmarking Catastrophic Forgetting Mitigation Methods in Federated Time Series Forecasting, DOI: 10.48550/arXiv.2510.21491

---

## 📘 Overview

Catastrophic forgetting (CF) occurs when models trained sequentially on non-stationary data lose performance on past tasks.  
In time-series forecasting under federated learning (FL), this challenge is amplified by:

- **Non-IID client data distributions**
- **Temporal concept drift**
- **Limited communication and privacy constraints**

This framework evaluates key continual learning (CL) strategies for mitigating CF within FL systems using an **LSTM-based predictor**.

---

## 🧩 Implemented Methods

The framework benchmarks **five core CF-mitigation algorithms** under a unified setting:

| Method | Description |
|:-------|:-------------|
| 🧱 **Naive FL** | Standard sequential training without forgetting mitigation. |
| 🔁 **Replay** | Memory-based rehearsal using KMeans-selected representative samples. |
| 🧮 **EWC** | Elastic Weight Consolidation (Fisher-based parameter regularization). |
| 🔗 **Online EWC** | Streaming version with moving Fisher matrix and reference weights. |
| 🧑‍🏫 **LwF (KD)** | Knowledge distillation between consecutive task models. |
| 🧠 **SI** | Synaptic Intelligence (importance-weighted plasticity control). |

---

## 📊 Dataset & Task Definition

- **Dataset:** Beijing Multi-Site Air Quality (12 stations = 12 clients)  
- **Input Features:** meteorological + pollutant variables + cyclical time encodings  
- **Target Variables:** `PM2.5`, `TEMP`, or `WSPM` (modifiable in notebook)  
- **Temporal Tasks:** sequential 3-month windows (2014–2017), forming 11 continual tasks  
- **Prediction:** multi-step forecasting (past 12 → next 6 hours)  
- **Normalization:** robust [1st, 99th] percentile scaling across all clients  

---

## ⚙️ Architecture & Framework Design

- **Model:** Bi-layer LSTM with fully-connected output head  
- **FL Aggregation:** FedAvg (equal client weights)  
- **Evaluation Metrics:**
  - Average Performance (AP)  
  - Average Forgetting (AF)  
  - Average Plasticity  
  - CPU Time per mode

---

## 🧪 Reproducibility & Usage

### 🧩 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
