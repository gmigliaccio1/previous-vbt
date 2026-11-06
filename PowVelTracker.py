from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Velocity Tracker API Online"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    num_rows = len(df)

    return{
        "filename": file.filename,
        "row": num_rows,
        "message": "File ricevuto correttamente"
    }