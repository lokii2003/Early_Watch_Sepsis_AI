"""
dummy_patients.py
=================
Generates TWO data streams per patient every second:

  live_data/patient_P001_raw.json    -- per-second vitals (ECG waveform)
  live_data/patient_P001_minute.json -- 60s averages (ACTUAL tables)
  live_data/patient_P001_early.json  -- 60s averages 30 min ahead (EARLY PREDICTION tables)

Run:
    python dummy_patients.py              start
    python dummy_patients.py restart      clear + start fresh
    python dummy_patients.py clear        delete all data
"""

import json, os, time, math, sys
import numpy as np
from datetime import datetime
from collections import deque

OUTPUT_DIR  = "live_data"
CONFIG_FILE = "patients_config.json"
MAX_MINUTES = 120
HORIZON     = 30      # minutes ahead for early prediction

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEV_SCALE = {"mild": 1.20, "moderate": 1.80, "severe": 2.50}
CLAMP = dict(
    hr=(35,180), sbp=(60,200), dbp=(30,130), rr=(6,45),
    temp=(35.0,41.5), spo2=(70.0,100.0), wbc=(1.0,35.0),
    lac=(0.3,15.0), ecg=(-0.5,3.5)
)

def cl(v, k):
    lo, hi = CLAMP[k]
    return max(lo, min(hi, float(v)))

def make_baseline(profile):
    seed = sum(ord(c) for c in profile["pid"])
    rng  = np.random.default_rng(seed)
    age  = int(profile.get("age", 55))
    adj  = (age - 45) * 0.12
    return dict(
        hr   = float(np.clip(75  + adj + rng.normal(0,4),    55, 95)),
        sbp  = float(np.clip(120 + adj + rng.normal(0,6),   100,145)),
        dbp  = float(np.clip(78       + rng.normal(0,5),     55, 95)),
        rr   = float(np.clip(15       + rng.normal(0,1.5),   11, 20)),
        temp = float(np.clip(37.0     + rng.normal(0,0.2), 36.2,37.4)),
        spo2 = float(np.clip(97.8-(age-45)*0.04+rng.normal(0,0.4), 95,99.5)),
        wbc  = float(np.clip(7.5      + rng.normal(0,1.0),  4.5,11.0)),
        lac  = float(np.clip(0.95     + rng.normal(0,0.1),  0.5, 1.8)),
        ecg  = float(np.clip(0.86     + rng.normal(0,0.03),0.72, 1.0)),
    )

DEFAULT_PATIENTS = [
    {"pid":"P001","name":"Kumar S.", "age":58,"sex":"M","onset":5, "sev":"moderate"},
    {"pid":"P002","name":"Chen L.",  "age":72,"sex":"F","onset":-1,"sev":"none"},
    {"pid":"P003","name":"Patel R.", "age":45,"sex":"M","onset":8, "sev":"severe"},
    {"pid":"P004","name":"Smith J.", "age":63,"sex":"F","onset":-1,"sev":"none"},
    {"pid":"P005","name":"Desai M.", "age":51,"sex":"M","onset":3, "sev":"mild"},
]

def create_default_config():
    """Auto-create patients_config.json if missing."""
    with open(CONFIG_FILE,"w",encoding="utf-8") as f:
        json.dump({"patients":DEFAULT_PATIENTS},f,indent=2,ensure_ascii=False)
    print("  [auto] Created {} with {} default patients".format(
        CONFIG_FILE, len(DEFAULT_PATIENTS)))

def load_config():
    # Auto-create if missing
    if not os.path.exists(CONFIG_FILE):
        create_default_config()
    try:
        # utf-8-sig handles BOM automatically (PowerShell Out-File issue)
        with open(CONFIG_FILE,"r",encoding="utf-8-sig") as f:
            return json.load(f).get("patients",[])
    except Exception as e:
        print("[config] error:",e)
        print("[config] falling back to default patients")
        return DEFAULT_PATIENTS

def compute_vitals(profile, base, minute, second, noise_offset=0):
    """Compute vitals at a specific (minute, second) point in the scenario."""
    onset = profile.get("onset",-1)
    sev   = profile.get("sev","none")
    t_sec = minute*60+second
    circ  = math.sin(2.0*math.pi*t_sec/28800.0)*0.5
    np.random.seed((abs(hash(profile["pid"]))+t_sec+noise_offset)%(2**31))

    if onset is not None and onset>=0 and minute>=onset:
        elapsed = minute-onset+second/60.0
        prog = 1.0/(1.0+math.exp(-0.15*(elapsed-10)))
        s    = prog*SEV_SCALE.get(sev,0.85)
        hr   = base["hr"]  +s*45 +np.random.normal(0,4.0)+circ
        sbp  = base["sbp"] -s*42 +np.random.normal(0,5.5)
        dbp  = base["dbp"] -s*22 +np.random.normal(0,3.5)
        rr   = base["rr"]  +s*16 +np.random.normal(0,1.8)
        temp = base["temp"]+s*2.0*math.sin(math.pi*min(elapsed/90.0,1))+np.random.normal(0,0.12)
        spo2 = base["spo2"]-s*14 +np.random.normal(0,0.8)
        wbc  = base["wbc"] +s*15 +np.random.normal(0,1.2)
        lac  = base["lac"] +s*6.5+np.random.normal(0,0.25)
        ecg  = base["ecg"] +s*0.6*math.sin(2.0*math.pi*t_sec/60.0)+np.random.normal(0,0.05)
        label=1
    else:
        hr   = base["hr"]  +3*math.sin(2*math.pi*t_sec/90.0)+np.random.normal(0,2.5)+circ
        sbp  = base["sbp"] +np.random.normal(0,3.5)+circ*2
        dbp  = base["dbp"] +np.random.normal(0,2.5)+circ
        rr   = base["rr"]  +np.random.normal(0,1.0)
        temp = base["temp"]+0.2*math.sin(2*math.pi*t_sec/28800.0)+np.random.normal(0,0.08)
        spo2 = base["spo2"]+np.random.normal(0,0.4)
        wbc  = base["wbc"] +np.random.normal(0,0.6)
        lac  = base["lac"] +np.random.normal(0,0.08)
        ecg  = base["ecg"] +0.08*math.sin(2*math.pi*t_sec/60.0)+np.random.normal(0,0.025)
        label=0

    return dict(
        patient_id=profile["pid"], patient_name=profile.get("name",profile["pid"]),
        age=profile.get("age","—"), sex=profile.get("sex","—"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        min=minute, second=second, t_sec=t_sec,
        heart_rate_bpm    =round(cl(hr,  "hr"),  1),
        systolic_bp_mmhg  =round(cl(sbp, "sbp"), 1),
        diastolic_bp_mmhg =round(cl(dbp, "dbp"), 1),
        resp_rate_br_min  =round(cl(rr,  "rr"),  1),
        temp_c            =round(cl(temp,"temp"),1),
        spo2_pct          =round(cl(spo2,"spo2"),1),
        wbc_103_ul        =round(cl(wbc, "wbc"), 2),
        lactate_mmol_l    =round(cl(lac, "lac"), 2),
        ecg_mv            =round(cl(ecg, "ecg"), 3),
        sepsis_label=label,
        sepsis_onset_minute=onset if (onset is not None and onset>=0) else -1,
        severity=sev,
    )

def avg_rows(rows, profile, minute, is_early=False, future_minute=None):
    if not rows: return None
    keys=["heart_rate_bpm","systolic_bp_mmhg","diastolic_bp_mmhg","resp_rate_br_min",
          "temp_c","spo2_pct","wbc_103_ul","lactate_mmol_l","ecg_mv"]
    avg={}
    for k in keys:
        vals=[r[k] for r in rows if isinstance(r.get(k),(int,float))]
        avg[k]=round(sum(vals)/len(vals),2) if vals else None
    last=rows[-1]
    avg.update({
        "patient_id":profile["pid"],"patient_name":profile.get("name",profile["pid"]),
        "age":profile.get("age","—"),"sex":profile.get("sex","—"),
        "timestamp":last["timestamp"],"min":minute,"second":59,"t_sec":last["t_sec"],
        "sepsis_label":last["sepsis_label"],"sepsis_onset_minute":last["sepsis_onset_minute"],
        "severity":last["severity"],"is_average":True,"avg_of_seconds":len(rows),
        "is_early_prediction":is_early,
        "predicted_for_minute":future_minute if is_early else None,
        "horizon_minutes":HORIZON if is_early else 0,
    })
    return avg

def save_raw(pid, buf):
    path=os.path.join(OUTPUT_DIR,"patient_{}_raw.json".format(pid))
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"pid":pid,"raw":list(buf),"latest":buf[-1] if buf else {},
                   "updated":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},f,ensure_ascii=False)

def save_minute(pid, avg, history):
    path=os.path.join(OUTPUT_DIR,"patient_{}_minute.json".format(pid))
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"pid":pid,"latest_minute":avg,"minute_history":list(history),
                   "updated":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},f,ensure_ascii=False)

def save_early(pid, early_avg, early_history):
    path=os.path.join(OUTPUT_DIR,"patient_{}_early.json".format(pid))
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"pid":pid,"latest_early":early_avg,"early_history":list(early_history),
                   "horizon_minutes":HORIZON,
                   "updated":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},f,ensure_ascii=False)

def clear_data():
    if not os.path.isdir(OUTPUT_DIR): print("  live_data/ not found."); return
    files=[f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
    for fname in files: os.remove(os.path.join(OUTPUT_DIR,fname))
    print("  Deleted {} file(s).".format(len(files)))

def main():
    args=sys.argv[1:]
    if "clear" in args:
        print("="*60); print("  CLEARING"); print("="*60)
        clear_data(); return
    if "restart" in args:
        print("="*60); print("  RESTART"); print("="*60)
        clear_data()

    print("="*60)
    print("  DUMMY PATIENTS  |  ACTUAL + EARLY PREDICTION ({}min ahead)".format(HORIZON))
    print("  Ctrl+C to stop")
    print("="*60)

    raw_bufs={};sec_acc={};early_acc={};min_hist={};early_hist={};baselines={};global_sec=0

    while True:
        minute=(global_sec//60)%MAX_MINUTES
        second=global_sec%60

        if global_sec%5==0:
            profiles=load_config()
            for prof in profiles:
                pid=prof["pid"]
                if pid not in raw_bufs:
                    raw_bufs[pid]=deque(maxlen=60); sec_acc[pid]=[]
                    early_acc[pid]=[]; min_hist[pid]=deque(maxlen=120)
                    early_hist[pid]=deque(maxlen=120); baselines[pid]=make_baseline(prof)
                    print("  [+] {} {}".format(pid, prof.get("name","")))
        else:
            profiles=load_config()

        for prof in profiles:
            pid=prof["pid"]
            if pid not in raw_bufs: continue

            # Actual vitals now
            row=compute_vitals(prof,baselines[pid],minute,second,noise_offset=0)
            raw_bufs[pid].append(row); sec_acc[pid].append(row)
            save_raw(pid,raw_bufs[pid])

            # Early prediction vitals (30 min into future)
            future_min=(minute+HORIZON)%MAX_MINUTES
            early_row=compute_vitals(prof,baselines[pid],future_min,second,noise_offset=7777)
            early_acc[pid].append(early_row)

            if second==59:
                # Save actual minute avg
                avg=avg_rows(sec_acc[pid],prof,minute,is_early=False)
                if avg: min_hist[pid].append(avg); save_minute(pid,avg,min_hist[pid])
                sec_acc[pid]=[]

                # Save early prediction minute avg
                early_avg=avg_rows(early_acc[pid],prof,minute,is_early=True,future_minute=future_min)
                if early_avg: early_hist[pid].append(early_avg); save_early(pid,early_avg,early_hist[pid])
                early_acc[pid]=[]

        if second==0 and profiles:
            ts=datetime.now().strftime("%H:%M:%S")
            print("\n[{}] min={:3d} | {} patients".format(ts,minute,len(profiles)))
            for prof in profiles:
                pid=prof["pid"]; buf=raw_bufs.get(pid)
                if buf:
                    r=buf[-1]; tag="SEPSIS" if r.get("sepsis_label") else "ok"
                    print("  {} HR={:5.1f} Lac={:.2f} [{}] early→min{}".format(
                        pid,r["heart_rate_bpm"],r["lactate_mmol_l"],tag,future_min))

        global_sec+=1
        time.sleep(1)

if __name__=="__main__":
    main()