from fastapi import APIRouter
from importlib.metadata import PackageNotFoundError, version

router = APIRouter()

@router.get("/version")
async def get_version():
    try:
        app_version = version("mlops")
    except PackageNotFoundError:
        app_version = "unknown"
    return {"version": app_version}