import os
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from scipy.integrate import cumulative_trapezoid
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class SetUpdate(BaseModel):
    exercise: str | None = None
    weight: float | None = None

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Credenziali Supabase mancanti!")

# Inizializzazione Client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Connessione a Supabase inizializzata!")
except Exception as e:
    print(f"Errore durante la creazione del client: {e}")

async def getTotalVectorMagnitude(file : UploadFile):
    contents = await file.read()

    DataFrame = pd.read_csv(
        io.BytesIO(contents),
        sep=None,
        engine='python',
        decimal=',',
        comment='#',
        skip_blank_lines=True,
        on_bad_lines='skip'
    )

    def clean_val(val):
        if pd.isna(val) or val == "":
            return 0.0
        # prende solamente la parte prima dell'eventuale ; presente nel formato dei dati esportati tramite physics toolbox
        s = str(val).split(';')[0]
        try:
            return float(s.replace(',', '.'))
        except:
            return 0.0
    
    time = DataFrame.iloc[:,0].apply(clean_val).values

    cols = DataFrame.columns
    a_tot_col = [c for c in cols if 'aT' in c or 'Total' in c]

    if a_tot_col:
        a_tot = DataFrame[a_tot_col[0]].apply(clean_val).values
    else:
        #calcolo manuale della magnitudo vettoriale totale
        ax = DataFrame.iloc[:, 1].apply(clean_val).values
        ay = DataFrame.iloc[:, 2].apply(clean_val).values
        az = DataFrame.iloc[:, 3].apply(clean_val).values
        a_tot = np.sqrt(ax**2 + ay**2 + az**2)  

    return time, a_tot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Velocity Tracker API Online - Supabase Connected"}

@app.post("/upload-csv")
async def upload_csv(exercise: str, weight: float, pause_type: str, file: UploadFile = File(...)):
    try:
        time, a_tot = await getTotalVectorMagnitude(file)

        if len(a_tot) < 50:
            raise HTTPException(status_code=400, detail="Dati insufficienti nel file .csv")
        
        bias = np.mean(a_tot[:20])
        acc_final = a_tot - bias

        acc_smoothed = pd.Series(acc_final).rolling(window=5, center=True).mean().fillna(0).values.copy()
        acc_smoothed[np.abs(acc_smoothed) < 0.10] = 0

        dt_list = np.diff(time, prepend=time[0])
        sample_rate = 1.0 / np.mean(dt_list)

        silence_seconds = 1.5 if pause_type == "paused" else 1.5
        MIN_V = 0.12
        MAX_SILENCE = int(silence_seconds * sample_rate)
        MIN_REP_FRAMES = int(0.4 * sample_rate)
        if "squat" in exercise.lower() or "deadlift" in exercise.lower():
                MIN_REP_DURATION = 1.5
        else:           
                MIN_REP_DURATION = 0.4
        MAX_V_PHYSICAL = 2.2

        reps_results = []
        temp_rep_vel = []
        silence_frames = 0
        v_zupt = 0.0
        
        for v in range(len(acc_smoothed)):
            if acc_smoothed[v] == 0:
                v_zupt = 0.0
            else:
                v_zupt += acc_smoothed[v] * dt_list[v]

            # hard clamp: physically impossible velocity = reset entirely
            if v_zupt > MAX_V_PHYSICAL:
                v_zupt = 0.0
                temp_rep_vel = []
                silence_frames = 0
                continue

            v_zupt = max(0, v_zupt)

            if v_zupt > MIN_V:
                temp_rep_vel.append(v_zupt)
                silence_frames = 0
            else:
                if len(temp_rep_vel) > MIN_REP_FRAMES:
                    silence_frames += 1
                    if silence_frames == 1:
                        v_zupt = 0.0
                    if silence_frames > MAX_SILENCE:
                        duration = len(temp_rep_vel) * np.mean(dt_list)
                        max_v_found = np.max(temp_rep_vel)
                        if duration > MIN_REP_DURATION and max_v_found < MAX_V_PHYSICAL:
                            avg_vel = float(np.mean(temp_rep_vel))
                            peak_vel = float(np.max(temp_rep_vel))
                            if peak_vel / avg_vel < 2.0:
                                reps_results.append({
                                "avg_vel": avg_vel,
                                "peak_vel": peak_vel,
                                "duration": round(len(temp_rep_vel) * float(np.mean(dt_list)), 2)
                                })
                        temp_rep_vel = []
                        silence_frames = 0
                        v_zupt = 0.0

        # recupero ultima rep
        if len(temp_rep_vel) > MIN_REP_FRAMES:
            duration = len(temp_rep_vel) * np.mean(dt_list)
            max_v_found = np.max(temp_rep_vel)
            if duration > MIN_REP_DURATION and max_v_found < MAX_V_PHYSICAL:
                avg_vel = float(np.mean(temp_rep_vel))
                peak_vel = float(np.max(temp_rep_vel))
                if peak_vel / avg_vel < 2.0:
                    reps_results.append({
                       "avg_vel": avg_vel,
                        "peak_vel": peak_vel,
                        "duration": round(len(temp_rep_vel) * float(np.mean(dt_list)), 2)
                    })

        # calcolo velocity loss
        final_data_reps = []
        total_set_loss = 0.0
        if reps_results:
            serie_max_vel = max([r["avg_vel"] for r in reps_results])
            for i, rep in enumerate(reps_results):
                total_loss = ((serie_max_vel - rep["avg_vel"]) / serie_max_vel) * 100
                final_data_reps.append({
                    "rep_number": i + 1,
                    "avg_vel": round(rep["avg_vel"], 3),
                    "peak_vel": round(rep["peak_vel"], 3),
                    "loss_respect_to_fastest": round(max(0, total_loss), 2),
                    "duration": rep.get("duration", 0)
                })
            total_set_loss = final_data_reps[-1]["loss_respect_to_fastest"] if len(final_data_reps) > 1 else 0

        data = {
            "exercise": exercise,
            "weight": weight,
            "reps": final_data_reps,
            "final_loss": total_set_loss,
            "count": len(final_data_reps),
            "filename": file.filename
        }

        response = supabase.table("sets").insert(data).execute()

        return {
            "message": "Dati salvati con successo",
            "count": len(final_data_reps),
            "reps_details": final_data_reps,
            "db_response": response.data
        }
    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sets")
def get_sets():
    try:
        response = supabase.table("sets").select("*").order("created_at", desc=True).execute()
        return{"sets": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sets/{id}")
def get_set(id: int):
    try:
        response = supabase.table("sets").select("*").eq("id", id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Set non trovato")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sets/{id}")
def delete_set(id: int):
    try:
        response = supabase.table("sets").delete().eq("id", id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Set non trovato")
        return {"message": "Set eliminato con successo", "id": id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/sets/{id}")
def update_set(id: int, data: SetUpdate):
    try:
        updates = {k : v for k, v in data.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(status_code=400, detail="Nessun campo da aggiornare")
        response = supabase.table("sets").update(updates).eq("id", id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Set non trovato")
        return {"message": "Set aggiornato con successo", "updated_fields": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
