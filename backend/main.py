# this is our main api file
from fastapi import FastAPI

app = FastAPI()

# simple health check route
@app.get("/")
def read_root():
    return {"status": "ok"}
