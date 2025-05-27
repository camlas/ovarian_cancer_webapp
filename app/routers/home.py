from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("home.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "max_file_size": settings.MAX_FILE_SIZE,
        "allowed_extensions": settings.ALLOWED_EXTENSIONS,
        "performance": config_data['performance'],
        "config_data": config_data
    })


@router.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("home.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "max_file_size": settings.MAX_FILE_SIZE,
        "allowed_extensions": settings.ALLOWED_EXTENSIONS,
        "performance": config_data['performance'],
        "config_data": config_data
    })