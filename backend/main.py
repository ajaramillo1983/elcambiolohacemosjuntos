"""Backend independiente de El cambio IA. No importa ni modifica Centinela."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import hmac
import io
import os
import re
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

try:
    from .face_restore import prepare_head, locate_head, compose_original_head
    from .source_preserve import prepare_source, create_guide, finish_scene, verify_pipeline, apply_office_pose
except ImportError:  # tests/start command from backend directory
    from face_restore import prepare_head, locate_head, compose_original_head
    from source_preserve import prepare_source, create_guide, finish_scene, verify_pipeline, apply_office_pose

MAX_BYTES = 8 * 1024 * 1024
MAX_BODY = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000
STYLES = {
    'Empresarias': 'smart casual business clothing, soft neutral office light',
    'Deportistas': 'modest sportswear and sneakers, neutral gym light',
    'Graduadas': 'a black graduation gown and mortarboard, soft daylight',
    'De_gala': 'elegant formal evening clothing, warm indoor light',
    'Viajeras': 'casual beach travel clothing, warm sunset light',
}

@dataclass(frozen=True)
class Settings:
    api_key: str
    access_code: str
    enabled: bool = False
    model: str = 'gemini-3.1-flash-image'
    origins: tuple[str, ...] = ('https://ajaramillo1983.github.io',)
    quota_db: str = '/tmp/elcambio/quota.sqlite3'
    daily_limit: int = 20
    person_daily_limit: int = 3

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv('EL_CAMBIO_GEMINI_API_KEY', '').strip(),
            access_code=os.getenv('EL_CAMBIO_ACCESS_CODE', '').strip(),
            enabled=os.getenv('EL_CAMBIO_ENABLED', 'false').lower() == 'true',
            model=os.getenv('EL_CAMBIO_IMAGE_MODEL', 'gemini-3.1-flash-image'),
            origins=tuple(x.strip() for x in os.getenv('CORS_ORIGINS', 'https://ajaramillo1983.github.io').split(',') if x.strip()),
            quota_db=os.getenv('EL_CAMBIO_QUOTA_DB', '/tmp/elcambio/quota.sqlite3'),
            daily_limit=int(os.getenv('EL_CAMBIO_DAILY_LIMIT', '20')),
            person_daily_limit=int(os.getenv('EL_CAMBIO_PERSON_DAILY_LIMIT', '3')),
        )

class GenerateInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    image_base64: str = Field(min_length=1, max_length=11_184_812)
    mime_type: Literal['image/jpeg', 'image/png', 'image/webp']
    style: Literal['Empresarias', 'Deportistas', 'Graduadas', 'De_gala', 'Viajeras']
    consent: Literal[True]


def normalize_image(encoded: str) -> str:
    """Valida y normaliza en memoria; elimina metadatos antes de enviar a Google."""
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, 'La imagen supera los 8 MB.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                    raise ValueError('unsupported image')
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGBA')
                image.thumbnail((1600, 1600))
                clean = Image.new('RGBA', image.size, (0, 0, 0, 0))
                clean.paste(image)
                out = io.BytesIO()
                clean.save(out, format='PNG')
        return base64.b64encode(out.getvalue()).decode('ascii')
    except HTTPException:
        raise
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, 'La imagen no es válida. Usa una foto JPG, PNG o WebP.') from None


_cutout_session = None

def remove_background_server(encoded: str) -> str:
    """Remove the background on the service, independent of browser/WASM support."""
    try:
        from rembg import new_session, remove
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGBA')
            global _cutout_session
            if _cutout_session is None:
                _cutout_session = new_session('isnet-general-use', providers=['CPUExecutionProvider'])
            cutout = remove(image, session=_cutout_session, alpha_matting=False)
            output = io.BytesIO()
            cutout.convert('RGBA').save(output, format='PNG', optimize=True)
        return base64.b64encode(output.getvalue()).decode('ascii')
    except Exception as exc:
        raise HTTPException(503, 'No se pudo separar la persona de la foto. El recorte no se completó y no se consumió una generación.') from exc


def reserve_attempt(settings: Settings, client_ip: str):
    """Reserva atómica antes de gastar; los fallos también cuentan, sin reintentos automáticos."""
    day = datetime.now(timezone.utc).date().isoformat()
    subject = hmac.new(settings.access_code.encode(), (day + client_ip).encode(), hashlib.sha256).hexdigest()
    path = Path(settings.quota_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=5) as db:
        db.execute('CREATE TABLE IF NOT EXISTS attempts (day TEXT NOT NULL, subject TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        total = db.execute('SELECT COUNT(*) FROM attempts WHERE day=?', (day,)).fetchone()[0]
        individual = db.execute('SELECT COUNT(*) FROM attempts WHERE day=? AND subject=?', (day, subject)).fetchone()[0]
        if total >= settings.daily_limit or individual >= settings.person_daily_limit:
            raise HTTPException(429, 'Se alcanzó el límite diario del piloto. Intenta mañana.')
        db.execute('DELETE FROM attempts WHERE day < ?', (day,))
        db.execute('INSERT INTO attempts VALUES (?, ?)', (day, subject))


async def generate_person(settings: Settings, encoded: str, style: str, guide: str | None = None) -> dict:
    # Plantilla confiable del repositorio; ninguna URL ni ruta proviene del cliente.
    template_path = Path(__file__).resolve().parent.parent / 'imagenes' / 'ampliadas' / f'{style}.png'
    if style not in STYLES or not template_path.is_file():
        raise HTTPException(503, 'La plantilla de este estilo no está disponible.')
    template = base64.b64encode(template_path.read_bytes()).decode('ascii')
    prompt = (
        'Create ONE natural group photograph with exactly three people. '
        'IMAGE 1 is the scene template. IMAGE 2 is the ORIGINAL PERSON with transparent background. '
        'IMAGE 3, when present, is the LOCKED COMPOSITION GUIDE that already contains that original person. '
        'Use IMAGE 3 as the canvas to complete, not as an approximate reference. '
        'Keep the original person in exactly the guide position, scale, head orientation and pose. '
        'The original face, hair, ears, neck, exposed upper chest, shoulders, visible skin, arms, torso '
        'and clothing are ONE CONTINUOUS protected region. Do not replace the neck or paste a new face. '
        'Do not redraw, beautify, slim, reshape or change any already visible part. '
        'If the photo shows a full body, preserve that entire body and clothing; only integrate light and scene. '
        'If partial, preserve ALL visible body and clothing and generate ONLY missing lower parts. '
        'If a close selfie, retain the original head AND visible neck/shoulders; construct missing anatomy '
        'continuously from their existing shape and proportions. Never construct a different neck under the face. '
        'Use the original neck and visible skin as the reference for skin tone, undertone, texture and light '
        'of every newly generated exposed body part; do not copy skin color from the other women. '
        'Continue the existing silhouette and garments across the crop boundary with natural anatomy. '
        f'For parts genuinely missing, style: {STYLES[style]}. '
        'Do not change existing clothing to fit this style. '
        'Keep the other two women, their faces and the entire slogan/logo unchanged. '
        'Place all three people side by side on the same depth plane, close enough for their shoulders '
        'to touch naturally. The added person stands shoulder to shoulder beside the nearest woman. '
        'Do not place the added person behind, in front of, on top of, or overlapping another person. '
        'Do not drape an arm or hand over anyone. Keep every body distinct and every face unobstructed. '
        'Match contact shadows, ground plane and perspective. No floating parts, cut lines, halos, '
        'detached heads, duplicated faces, discolored neck, extra limbs or sticker appearance. '
        'Return the complete portrait scene, not a separate body or cutout. Vertical 2:3, little unused left margin.'
    )
    body = {
        'contents': [{'parts': [
            {'text': prompt},
            {'text': 'IMAGE 1: scene to preserve and edit'},
            {'inlineData': {'mimeType': 'image/png', 'data': template}},
            {'text': 'IMAGE 2: identity reference of the additional person'},
            {'inlineData': {'mimeType': 'image/png', 'data': encoded}},
        ]}],
        'generationConfig': {'responseModalities': ['TEXT', 'IMAGE'], 'imageConfig': {'aspectRatio': '2:3'}},
    }
    if guide:
        body['contents'][0]['parts'].extend([
            {'text': 'IMAGE 3: locked guide; preserve original person pixels, generate only missing parts'},
            {'inlineData': {'mimeType': 'image/png', 'data': guide}},
        ])
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(150, connect=15)) as client:
            # Header, no query string: la clave nunca forma parte de una URL.
            response = await client.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{settings.model}:generateContent',
                headers={'x-goog-api-key': settings.api_key}, json=body,
            )
    except httpx.TimeoutException:
        raise HTTPException(504, 'La IA tardó demasiado. No se reintentó para evitar cobros duplicados.') from None
    except httpx.HTTPError:
        raise HTTPException(502, 'No se pudo contactar al servicio de imágenes.') from None
    if response.status_code == 429:
        raise HTTPException(429, 'Google alcanzó su cuota o está ocupado. Intenta más tarde.')
    if response.status_code in (401, 403):
        raise HTTPException(503, 'La credencial de imágenes necesita revisar acceso o facturación en Google.')
    if response.status_code == 404:
        raise HTTPException(503, 'El modelo de imágenes configurado no está disponible para este proyecto.')
    if response.is_error:
        raise HTTPException(502, 'El proveedor rechazó la solicitud de imágenes.')
    try:
        data = response.json()
        for candidate in data.get('candidates', []):
            for part in candidate.get('content', {}).get('parts', []):
                if part.get('thought'):
                    continue
                inline = part.get('inlineData') or part.get('inline_data') or {}
                mime = inline.get('mimeType') or inline.get('mime_type')
                encoded_output = inline.get('data')
                if mime in {'image/png', 'image/jpeg', 'image/webp'} and encoded_output:
                    raw = base64.b64decode(encoded_output, validate=True)
                    if not raw or len(raw) > 12 * 1024 * 1024:
                        continue
                    with Image.open(io.BytesIO(raw)) as image:
                        image.verify()
                    return {'image_base64': encoded_output, 'mime_type': mime, 'generated': True, 'composition': 'scene', 'style': style}
    except (ValueError, TypeError, KeyError, OSError, Image.DecompressionBombError):
        raise HTTPException(502, 'El proveedor devolvió una imagen no válida.') from None
    raise HTTPException(422, 'La IA no devolvió una imagen. Prueba otra foto clara de una sola persona.')


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    verify_pipeline()
    if not re.fullmatch(r'[a-zA-Z0-9._-]{1,100}', settings.model):
        raise ValueError('Modelo de imágenes inválido')
    app = FastAPI(title='El cambio IA', docs_url=None, redoc_url=None, openapi_url=None)
    site_root = Path(__file__).resolve().parent.parent
    app.mount('/imagenes', StaticFiles(directory=site_root / 'imagenes'), name='plantillas')

    @app.get('/')
    async def frontend():
        return FileResponse(site_root / 'index.html', media_type='text/html')

    @app.get('/backend-config.js')
    async def frontend_config():
        return Response('window.EL_CAMBIO_API_BASE = window.location.origin;', media_type='application/javascript')
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.origins), allow_methods=['GET', 'POST'],
                       allow_headers=['Content-Type', 'X-El-Cambio-Code'], allow_credentials=False)
    busy = asyncio.Lock()

    @app.middleware('http')
    async def body_limit(request: Request, call_next):
        if request.method == 'POST':
            size = 0
            chunks = []
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BODY:
                    return JSONResponse({'detail': 'La solicitud supera el tamaño permitido.'}, status_code=413)
                chunks.append(chunk)
            request._body = b''.join(chunks)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/health')
    async def health():
        return {'status': 'ok', 'service': 'elcambio-ia-backend',
                'generation_ready': bool(settings.enabled and settings.api_key and bool(settings.access_code)),
                'access_code_required': True, 'composition': 'scene', 'face_restore': 'source-region-v4'}

    @app.post('/api/persona/generar')
    async def generate(payload: GenerateInput, request: Request):
        if not settings.enabled or not settings.api_key or not settings.access_code:
            raise HTTPException(503, 'La generación de imágenes todavía no está habilitada.')
        supplied = request.headers.get('x-el-cambio-code', '')
        if not hmac.compare_digest(supplied.encode(), settings.access_code.encode()):
            raise HTTPException(401, 'El código de acceso del piloto no es correcto.')
        if busy.locked():
            raise HTTPException(429, 'Hay otra imagen en proceso. Espera un momento.')
        async with busy:
            encoded = await run_in_threadpool(normalize_image, payload.image_base64)
            cutout = await run_in_threadpool(remove_background_server, encoded)
            source = await run_in_threadpool(prepare_source, cutout)
            source = await run_in_threadpool(apply_office_pose, source, payload.style)
            template_path = Path(__file__).resolve().parent.parent / 'imagenes' / 'ampliadas' / f'{payload.style}.png'
            guide, placement = await run_in_threadpool(create_guide, template_path, source)
            # Se usa la IP de la conexión, nunca un X-Forwarded-For arbitrario.
            await run_in_threadpool(reserve_attempt, settings, request.client.host if request.client else 'unknown')
            result = await generate_person(settings, cutout, payload.style, guide=guide)
            result['image_base64'] = await run_in_threadpool(finish_scene, result['image_base64'], source, placement)
            result['mime_type'] = 'image/png'
            result['original_head'] = source
            result['head_placement'] = placement
            result['face_restore'] = 'source-region-v4'
            return result
    return app

app = create_app()
