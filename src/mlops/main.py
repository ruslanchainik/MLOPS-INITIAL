from fastapi import FastAPI

from mlops.api.healthz import router as healthz_router
from mlops.api.v1.router import router as v1_router

app = FastAPI()
app.include_router(healthz_router)
app.include_router(v1_router)
