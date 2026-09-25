from fastapi import APIRouter

from mlops import __version__

router = APIRouter()


@router.get("/version")
async def get_version():
    return {"version": __version__}
