# 🏥 Early Watch: Sepsis AI
### Real-time ICU Sepsis Monitoring with 30-Minute Early Prediction

> Built by **Team CLASH OF CODE** at Ideathon

---

## 🔗 Live Demos

| | Link |
|---|---|
| 🔴 Early Sepsis Alert System | https://early-watch-icu-sepsis-ai-nwuo.onrender.com/
| 

---

## 🧠 What It Does

Sepsis is a life-threatening emergency — and the biggest killer is late detection. **Early Watch** monitors ICU patients in real time and raises an alert **30 minutes before** vitals deteriorate, giving clinicians a crucial window to act.

Two systems, one goal — catch sepsis before it's too late.

---

## 🔥 Features

### 🔴 Early Sepsis Prediction
- LSTM deep learning model trained on ICU vitals
- Predicts sepsis risk **30 minutes into the future**
- Rule-based fallback if no ML model is loaded
- Risk levels: `LOW` → `MODERATE` → `HIGH` → `CRITICAL`
- Trend tracking: rising / stable / falling probability

### 🟢 Multi-Bed ICU Monitor
- Watch **all ICU patients on one screen** simultaneously
- Live vitals per patient: HR, BP, SpO2, Temperature, Lactate, RR, WBC, ECG
- Real-time ECG-style waveforms
- Minute-by-minute history charts
- Full event log with alert timestamps

### ⚙️ Backend
- FastAPI server with embedded dummy patient data generator
- No separate data process needed — runs as a single deployment
- Polls and ingests live vitals every second
- REST API with full CRUD for patient management

---

## 🗂️ Project Structure

```
├── fastapi_server.py       # FastAPI backend + embedded data generator
├── index.html              # Frontend dashboard (vanilla JS)
├── dummy_patients.py       # Standalone data generator (local dev)
├── patients_config.json    # Patient profiles (auto-created)
├── live_data/              # Per-patient JSON data files (auto-created)
├── requirements.txt        # Python dependencies
└── render.yaml             # Render deployment config
```
