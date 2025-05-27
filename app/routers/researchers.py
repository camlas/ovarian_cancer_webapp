from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def researchers_root(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("researchers.html", {
        "request": request,
        "config_data": config_data,
        "researchers": config_data['researchers'],
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })


@router.get("/researchers", response_class=HTMLResponse)
async def researchers_page(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("researchers.html", {
        "request": request,
        "config_data": config_data,
        "researchers": config_data['researchers'],
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })

@router.get("/team", response_class=HTMLResponse)
async def researchers_page(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("researchers.html", {
        "request": request,
        "config_data": config_data,
        "researchers": config_data['researchers'],
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })