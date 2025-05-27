from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def resources_root(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("resources.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })


@router.get("/resources", response_class=HTMLResponse)
async def resources_page(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("resources.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })


@router.get("/dataset", response_class=HTMLResponse)
async def dataset_redirect(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("resources.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "highlight_dataset": True  # Optional: to highlight dataset section
    })