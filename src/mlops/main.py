from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker

from mlops.api.healthz import router as healthz_router
from mlops.api.v1.router import router as v1_router
from mlops.db.session import create_db_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_db_engine()
    app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.include_router(healthz_router)
app.include_router(v1_router)
