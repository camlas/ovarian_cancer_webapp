from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def about_root(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("about.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })


@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    config_data = settings.get_full_config()

    return templates.TemplateResponse("about.html", {
        "request": request,
        "config_data": config_data,
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    })


@router.get("/mission", response_class=HTMLResponse)
async def mission_redirect(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/about#mission")


@router.get("/vision", response_class=HTMLResponse)
async def vision_redirect(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/about#mission")


@router.get("/technology", response_class=HTMLResponse)
async def technology_redirect(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/about#technology")


@router.get("/impact", response_class=HTMLResponse)
async def impact_redirect(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/about#impact")

