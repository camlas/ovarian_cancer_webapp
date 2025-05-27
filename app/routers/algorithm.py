from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def algorithm_page(request: Request):
    config_data = settings.get_full_config()
    algorithm_info = config_data['algorithm']

    return templates.TemplateResponse("algorithm.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "config_data": config_data,
        "algorithm": algorithm_info,
        "performance": config_data['performance'],
    })


@router.get("/algorithm", response_class=HTMLResponse)
async def algorithm_redirect(request: Request):
    config_data = settings.get_full_config()
    algorithm_info = config_data['algorithm']

    return templates.TemplateResponse("algorithm.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "config_data": config_data,
        "algorithm": algorithm_info,
        "performance": config_data['performance'],
    })