from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def privacy_policy_root(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("privacy.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })


@router.get("/privacy", response_class=HTMLResponse)
async def privacy_policy_page(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("privacy.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })


@router.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy_alt(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("privacy.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })