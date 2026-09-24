from fastapi import APIRouter

from mlops.api.v1 import health, version

router = APIRouter(prefix="/api/v1")
router.include_router(version.router, tags=["meta"])
router.include_router(health.router, tags=["meta"])
