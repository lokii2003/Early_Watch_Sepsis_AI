"""
fastapi_server.py
=================
FastAPI server replacing Flask.
Serves actual vitals AND 30-minute early prediction data separately.

Install:
    pip install fastapi uvicorn numpy pandas joblib scikit-learn openpyxl

Run:
    uvicorn fastapi_server:app --host 0.0.0.0 --port 5000 --reload

Endpoints:
    GET  /                        serve index.html
    GET  /api/patients            all patients: actual + early prediction
    GET  /api/patient/{pid}       single patient full detail
    GET  /api/alerts              alert log
    GET  /api/status              server health
    POST /api/add_patient         add new patient
    DELETE /api/remove/{pid}      remove patient
    POST /api/restart             clear all data + buffers
    GET  /health                  Render/Railway health check
"""

import os, json, glob, time, threading
from datetime import datetime
from collections import deque
from typing import Optional

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── CONFIG ─────────────────────────────────────────────────────────
DATA_DIR      = "live_data"
CONFIG_FILE   = "patients_config.json"
PORT          = int(os.environ.get("PORT", 5000))
POLL_SEC      = 10
SEQ_LEN       = 30
THRESHOLD     = 0.50
EARLY_WARN_T  = 0.20
HORIZON       = 30      # matches dummy_patients.py

FEATURE_COLS = [
    "heart_rate_bpm","systolic_bp_mmhg","diastolic_bp_mmhg",
    "resp_rate_br_min","temp_c","spo2_pct","wbc_103_ul",
    "lactate_mmol_l","ecg_mv",
]
FEAT_MIN = np.array([35, 60, 30,  6, 35.0,  70.0,1.0,0.3,-0.5],dtype=np.float32)
FEAT_MAX = np.array([180,200,130,45, 41.5, 100.0,35.0,15.0,3.5], dtype=np.float32)
CLAMPS   = {
    "heart_rate_bpm":(35,180),"systolic_bp_mmhg":(60,200),
    "diastolic_bp_mmhg":(30,130),"resp_rate_br_min":(6,45),
    "temp_c":(35.0,41.5),"spo2_pct":(70.0,100.0),
    "wbc_103_ul":(1.0,35.0),"lactate_mmol_l":(0.3,15.0),"ecg_mv":(-0.5,3.5),
}

# ── MODEL ──────────────────────────────────────────────────────────
_model=None; _scaler=None; model_ok=False; scaler_ok=False

def load_artifacts():
    global _model,_scaler,model_ok,scaler_ok
    for pat in ["sepsis_lstm_model.h5","early_sepsis_lstm_model.h5","*.h5"]:
        hits=glob.glob(pat)
        if hits:
            try:
                import tensorflow as tf; tf.get_logger().setLevel("ERROR")
                _model=tf.keras.models.load_model(hits[0]); model_ok=True
                print("[server] Model: {}".format(hits[0]))
            except Exception as e: print("[server] Model error:",e)
            break
    for pat in ["early_sepsis_scaler.pkl","sepsis_scaler.pkl","*.pkl"]:
        hits=glob.glob(pat)
        if hits:
            try:
                _scaler=joblib.load(hits[0]); scaler_ok=True
                print("[server] Scaler: {}".format(hits[0]))
            except Exception as e: print("[server] Scaler error:",e)
            break
    if not model_ok: print("[server] No model — rule-based fallback")

# Mapping from current column names → original scaler training column names
_SCALER_COL_MAP = {
    "resp_rate_br_min": "resp._rate_br_min",
    "spo2_pct":         "spo2_%",
    "wbc_103_ul":       "wbc_103_µl",
}

def normalise(raw):
    if scaler_ok and _scaler is not None:
        import pandas as pd, warnings
        df = pd.DataFrame([raw], columns=FEATURE_COLS)
        # Rename columns to match what scaler was fitted on
        df = df.rename(columns=_SCALER_COL_MAP)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _scaler.transform(df)[0].astype(np.float32)
    return ((raw-FEAT_MIN)/(FEAT_MAX-FEAT_MIN+1e-8)).astype(np.float32)

def rule_score(window):
    last=window[-1]; raw=last*(FEAT_MAX-FEAT_MIN)+FEAT_MIN
    hr,sbp,dbp,rr,temp,spo2,wbc,lac,ecg=raw; s=0.0
    if hr>100:               s+=min((hr-100)/80,1.0)*0.22
    if sbp<90:               s+=min((90-sbp)/30,1.0)*0.22
    if rr>20:                s+=min((rr-20)/20,1.0)*0.14
    if temp>38.3 or temp<36: s+=0.10
    if spo2<95:              s+=min((95-spo2)/10,1.0)*0.14
    if wbc>12 or wbc<4:      s+=0.05
    if lac>2.0:              s+=min((lac-2.0)/8,1.0)*0.18
    return float(np.clip(s+np.random.normal(0,0.02),0,1))

# ── PATIENT STATE ──────────────────────────────────────────────────
class PatientState(object):
    def __init__(self, pid):
        self.pid                = pid
        self.buffer             = deque(maxlen=SEQ_LEN)
        self.early_buffer       = deque(maxlen=SEQ_LEN)
        self.history            = []
        self.early_history_pred = []   # predicted probability history for early data
        self.last_minute        = -1
        self.last_early_minute  = -1
        self.cur_vitals         = {}
        self.cur_minute_vitals  = {}
        self.cur_early_vitals   = {}
        self.cur_result         = {}
        self.cur_early_result   = {}
        self.minute_history     = []
        self.early_minute_history=[]
        self.alert_log          = []

    def _run_lstm(self, buf):
        """Run LSTM or fallback on a filled buffer."""
        bsz=len(buf)
        if bsz<SEQ_LEN:
            return dict(probability=0.05,prediction=0,
                        risk_level="BUFFERING",buffer_size=bsz,
                        model_used="waiting({}/{})".format(bsz,SEQ_LEN))
        window=np.array(list(buf),dtype=np.float32)
        X=window.reshape(1,SEQ_LEN,len(FEATURE_COLS))
        if model_ok and _model is not None:
            try: prob=float(_model.predict(X,verbose=0)[0][0]); mused="LSTM"
            except: prob=rule_score(window); mused="fallback"
        else: prob=rule_score(window); mused="rule-based"
        if   prob<0.25:       risk="LOW"
        elif prob<EARLY_WARN_T: risk="LOW"
        elif prob<THRESHOLD:  risk="MODERATE"
        elif prob<0.75:       risk="HIGH"
        else:                 risk="CRITICAL"
        early=(EARLY_WARN_T<=prob<THRESHOLD)
        probs=[h.get("probability",0) for h in self.history[-5:]]
        trend="rising" if len(probs)>=3 and probs[-1]-probs[0]>0.05 else (
               "falling" if len(probs)>=3 and probs[0]-probs[-1]>0.05 else "stable")
        eta=None
        if trend=="rising" and prob<THRESHOLD and len(probs)>=2:
            rate=(probs[-1]-probs[0])/max(len(probs),1)
            if rate>0: eta=min(int((THRESHOLD-prob)/rate),60)
        return dict(probability=round(prob,4),prediction=int(prob>=THRESHOLD),
                    risk_level=risk,early_warn=early,eta_minutes=eta,
                    trend=trend,buffer_size=bsz,model_used=mused)

    def ingest_actual(self, vitals):
        """Ingest one actual minute reading."""
        minute=vitals.get("min",-1)
        if minute==self.last_minute: return None
        self.last_minute=minute; self.cur_vitals=vitals
        raw=np.zeros(len(FEATURE_COLS),dtype=np.float32)
        for i,col in enumerate(FEATURE_COLS):
            val=vitals.get(col,None)
            if val is None: lo,hi=CLAMPS[col]; val=(lo+hi)/2.0
            raw[i]=float(np.clip(val,CLAMPS[col][0],CLAMPS[col][1]))
        self.buffer.append(normalise(raw))
        result=self._run_lstm(self.buffer)
        result["minute"]=minute; result["vitals"]=vitals
        self.history.append({"minute":minute,"probability":result["probability"],
                             "risk_level":result["risk_level"]})
        self.history=self.history[-80:]
        if result["risk_level"] in("HIGH","CRITICAL"):
            self.alert_log.append({"time":datetime.now().strftime("%H:%M:%S"),
                "pid":self.pid,"minute":minute,"prob":result["probability"],
                "risk":result["risk_level"],"source":"actual"})
            self.alert_log=self.alert_log[-30:]
        self.cur_result=result; return result

    def ingest_early(self, early_vitals):
        """Ingest one early-prediction minute reading."""
        minute=early_vitals.get("min",-1)
        if minute==self.last_early_minute: return None
        self.last_early_minute=minute; self.cur_early_vitals=early_vitals
        raw=np.zeros(len(FEATURE_COLS),dtype=np.float32)
        for i,col in enumerate(FEATURE_COLS):
            val=early_vitals.get(col,None)
            if val is None: lo,hi=CLAMPS[col]; val=(lo+hi)/2.0
            raw[i]=float(np.clip(val,CLAMPS[col][0],CLAMPS[col][1]))
        self.early_buffer.append(normalise(raw))
        result=self._run_lstm(self.early_buffer)
        result["minute"]=minute; result["vitals"]=early_vitals
        result["is_early"]=True
        result["predicted_for_minute"]=early_vitals.get("predicted_for_minute")
        self.early_history_pred.append({"minute":minute,"probability":result["probability"],
                                        "risk_level":result["risk_level"]})
        self.early_history_pred=self.early_history_pred[-80:]
        if result["risk_level"] in("HIGH","CRITICAL"):
            self.alert_log.append({"time":datetime.now().strftime("%H:%M:%S"),
                "pid":self.pid,"minute":minute,"prob":result["probability"],
                "risk":result["risk_level"],"source":"early_prediction"})
            self.alert_log=self.alert_log[-30:]
        self.cur_early_result=result; return result

# ── GLOBAL STATE ───────────────────────────────────────────────────
patients_lock=threading.Lock()
patients={}; all_alerts=[]; last_poll={"time":"—"}

def get_all_pids():
    if not os.path.exists(CONFIG_FILE): return []
    try:
        with open(CONFIG_FILE,"r",encoding="utf-8") as f: d=json.load(f)
        return [p["pid"] for p in d.get("patients",[])]
    except: return []

def ensure_patient(pid):
    with patients_lock:
        if pid not in patients: patients[pid]=PatientState(pid)

def read_file(path):
    if not os.path.exists(path): return None
    try:
        with open(path,"r",encoding="utf-8") as f: return json.load(f)
    except: return None

def read_raw(pid):
    d=read_file(os.path.join(DATA_DIR,"patient_{}_raw.json".format(pid)))
    return d.get("latest") if d else None

def read_minute(pid):
    d=read_file(os.path.join(DATA_DIR,"patient_{}_minute.json".format(pid)))
    if not d: return None,[]
    return d.get("latest_minute"),d.get("minute_history",[])

def read_early(pid):
    d=read_file(os.path.join(DATA_DIR,"patient_{}_early.json".format(pid)))
    if not d: return None,[]
    return d.get("latest_early"),d.get("early_history",[])

def read_waveform(pid):
    d=read_file(os.path.join(DATA_DIR,"patient_{}_raw.json".format(pid)))
    if not d: return []
    return [r.get("heart_rate_bpm",0) for r in d.get("raw",[])]

def poll_loop():
    global all_alerts
    while True:
        last_poll["time"]=datetime.now().strftime("%H:%M:%S")
        for pid in get_all_pids():
            ensure_patient(pid)

            # Read raw per-second data for LSTM buffer
            raw=read_raw(pid)
            if raw:
                patients[pid].ingest_actual(raw)

            # ALWAYS re-read minute history from file (dummy_patients writes every 10s)
            mv,mh=read_minute(pid)
            if mv:
                patients[pid].cur_minute_vitals=mv
            # Always update history even if empty list
            patients[pid].minute_history=mh if mh else patients[pid].minute_history

            # ALWAYS re-read early history from file
            ev,eh=read_early(pid)
            if ev:
                patients[pid].cur_early_vitals=ev
                patients[pid].ingest_early(ev)
            # Always update early history even if empty list
            patients[pid].early_minute_history=eh if eh else patients[pid].early_minute_history

            # Alert log
            r=patients[pid].cur_result
            if r and r.get("risk_level") in("HIGH","CRITICAL"):
                a=dict(r); a["vitals"]=mv; all_alerts.append(a)

        all_alerts=all_alerts[-50:]
        time.sleep(POLL_SEC)

# ── FASTAPI APP ────────────────────────────────────────────────────
app=FastAPI(title="ICU Sepsis Monitor",version="2.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

# ── SERVE FRONTEND ─────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p=os.path.join(os.path.dirname(os.path.abspath(__file__)),"index.html")
    if os.path.exists(p):
        with open(p,"r",encoding="utf-8") as f: return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>index.html not found</h2>",status_code=404)

@app.get("/health")
async def health():
    return {"status":"ok","version":"2.0"}

# ── PATIENTS ENDPOINT ───────────────────────────────────────────────
@app.get("/api/patients")
async def api_patients():
    pids=get_all_pids(); out=[]
    for pid in pids:
        ensure_patient(pid); p=patients[pid]
        mv=p.cur_minute_vitals if p.cur_minute_vitals else p.cur_vitals
        ev=p.cur_early_vitals
        r =p.cur_result; re=p.cur_early_result
        wf=read_waveform(pid)
        out.append({
            "pid":pid,
            "patient_name":mv.get("patient_name",pid),
            "age":mv.get("age","—"),"sex":mv.get("sex","—"),
            "minute":mv.get("min","—"),"timestamp":mv.get("timestamp","—"),
            # actual prediction
            "probability":r.get("probability",0.0),
            "risk_level":r.get("risk_level","BUFFERING"),
            "early_warn":r.get("early_warn",False),
            "eta_minutes":r.get("eta_minutes",None),
            "trend":r.get("trend","stable"),
            "model_used":r.get("model_used","—"),
            # actual vitals (for labels + actual table)
            "vitals":{k:mv.get(k,"—") for k in [
                "heart_rate_bpm","systolic_bp_mmhg","diastolic_bp_mmhg",
                "resp_rate_br_min","temp_c","spo2_pct","wbc_103_ul",
                "lactate_mmol_l","ecg_mv","sepsis_label"]},
            "minute_history":p.minute_history,
            # early prediction
            "early_probability":re.get("probability",0.0),
            "early_risk_level":re.get("risk_level","BUFFERING"),
            "early_trend":re.get("trend","stable"),
            "early_predicted_for_minute":ev.get("predicted_for_minute") if ev else None,
            # early vitals (for early table)
            "early_vitals":{k:ev.get(k,"—") for k in [
                "heart_rate_bpm","systolic_bp_mmhg","diastolic_bp_mmhg",
                "resp_rate_br_min","temp_c","spo2_pct","wbc_103_ul",
                "lactate_mmol_l","ecg_mv","sepsis_label"]} if ev else {},
            "early_minute_history":p.early_minute_history,
            # waveform
            "hr_waveform":wf,
        })
    return {"patients":out,"polled_at":last_poll["time"]}

@app.get("/api/patient/{pid}")
async def api_patient(pid:str):
    pid=pid.upper(); ensure_patient(pid)
    p=patients[pid]; _,mh=read_minute(pid); _,eh=read_early(pid)
    return {"pid":pid,"cur_result":p.cur_result,"cur_early_result":p.cur_early_result,
            "cur_vitals":p.cur_vitals,"cur_early_vitals":p.cur_early_vitals,
            "history":p.history,"alert_log":p.alert_log,
            "minute_history":mh,"early_minute_history":eh}

@app.get("/api/alerts")
async def api_alerts():
    combined=[]
    for pid in get_all_pids():
        if pid in patients: combined.extend(patients[pid].alert_log)
    combined.sort(key=lambda x:x.get("minute",0),reverse=True)
    return {"alerts":combined[:30]}

@app.get("/api/status")
async def api_status():
    files=os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []
    return {"server":"running","model_ok":model_ok,"scaler_ok":scaler_ok,
            "model_type":"LSTM" if model_ok else "rule-based",
            "data_files":len([f for f in files if f.endswith(".json")]),
            "poll_sec":POLL_SEC,"seq_len":SEQ_LEN,"threshold":THRESHOLD,
            "horizon":HORIZON,"last_poll":last_poll["time"],
            "patient_count":len(get_all_pids()),
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

class AddPatientBody(BaseModel):
    name:  str = "New Patient"
    age:   int = 50
    sex:   str = "M"
    onset: int = -1
    sev:   str = "moderate"

@app.post("/api/add_patient")
async def api_add_patient(body: AddPatientBody):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE,"r",encoding="utf-8") as f: cfg=json.load(f)
    else: cfg={"patients":[]}
    existing=[p["pid"] for p in cfg["patients"]]
    nums=[int(p[1:]) for p in existing if p.startswith("P") and p[1:].isdigit()]
    new_pid="P{:03d}".format((max(nums)+1) if nums else 1)
    sev=body.sev if body.sev in("mild","moderate","severe","none") else "moderate"
    prof={"pid":new_pid,"name":body.name.strip(),"age":body.age,"sex":body.sex.upper(),
          "onset":body.onset,"sev":sev if body.onset>=0 else "none"}
    cfg["patients"].append(prof)
    with open(CONFIG_FILE,"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2)
    ensure_patient(new_pid)
    print("[server] Patient added: {} {}".format(new_pid,body.name))
    return {"status":"ok","pid":new_pid,"profile":prof}

@app.delete("/api/remove/{pid}")
async def api_remove(pid:str):
    pid=pid.upper()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE,"r",encoding="utf-8") as f: cfg=json.load(f)
        cfg["patients"]=[p for p in cfg["patients"] if p["pid"]!=pid]
        with open(CONFIG_FILE,"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2)
    with patients_lock:
        if pid in patients: del patients[pid]
    for suf in ["_raw","_minute","_early",""]:
        fp=os.path.join(DATA_DIR,"patient_{}{}.json".format(pid,suf))
        if os.path.exists(fp): os.remove(fp)
    return {"status":"ok","removed":pid}

@app.post("/api/restart")
async def api_restart():
    global all_alerts
    deleted=[]
    with patients_lock:
        for pid in list(patients.keys()):
            p=patients[pid]; p.buffer.clear(); p.early_buffer.clear()
            p.history=[]; p.alert_log=[]; p.cur_vitals={}
            p.cur_minute_vitals={}; p.cur_early_vitals={}
            p.cur_result={}; p.cur_early_result={}
            p.last_minute=-1; p.last_early_minute=-1
            p.minute_history=[]; p.early_minute_history=[]
            p.early_history_pred=[]
    all_alerts=[]
    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if fname.startswith("patient_") and fname.endswith(".json"):
                try: os.remove(os.path.join(DATA_DIR,fname)); deleted.append(fname)
                except: pass
    msg="Reset OK — {} files deleted".format(len(deleted))
    print("[restart]",msg)
    return {"status":"ok","message":msg,"deleted":deleted,
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# ── STARTUP ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("="*65)
    print("  ICU SEPSIS MONITOR — FastAPI Server")
    print("  http://localhost:{}".format(PORT))
    print("  Actual data    + Early prediction ({}min ahead)".format(HORIZON))
    print("  /api/patients  /api/alerts  /api/status  /api/restart")
    print("="*65)
    load_artifacts()
    t=threading.Thread(target=poll_loop,daemon=True); t.start()
    print("[server] Poll thread started (every {}s)".format(POLL_SEC))

if __name__=="__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app",host="0.0.0.0",port=PORT,reload=False)