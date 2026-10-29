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
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

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
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((1600, 1600))
                clean = Image.new('RGB', image.size, 'white')
                clean.paste(image)
                out = io.BytesIO()
                clean.save(out, format='JPEG', quality=92)
        return base64.b64encode(out.getvalue()).decode('ascii')
    except HTTPException:
        raise
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, 'La imagen no es válida. Usa una foto JPG, PNG o WebP.') from None


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


async def generate_person(settings: Settings, encoded: str, style: str) -> dict:
    # No se envían las candidatas ni el eslogan al generador: permanecen originales en el montaje.
    prompt = (
        'Create a photorealistic, isolated full-length portrait of the ONE person in the reference photo. '
        'Preserve their facial identity, apparent age, hair, skin tone and expression as faithfully as possible. '
        'Do not beautify, rejuvenate or substitute another face. '
        'Keep visible body proportions. If the source is a selfie or cropped, complete the missing body '
        'with plausible natural anatomy; do not claim to recover actual measurements. '
        'Standing naturally, head to shoes fully in frame, feet visible, arms relaxed and entirely visible, '
        'camera at eye level, normal perspective, person looking at camera, ample margin around silhouette. '
        f'Clothing and lighting: {STYLES[style]}. '
        'Use a plain uniform light gray background with no objects, no other people, no words, no logos, '
        'no frame, no decorative elements. Return a single image suitable for cutting out and compositing.'
    )
    body = {
        'contents': [{'parts': [{'text': prompt}, {'inlineData': {'mimeType': 'image/jpeg', 'data': encoded}}]}],
        'generationConfig': {'responseModalities': ['TEXT', 'IMAGE'], 'imageConfig': {'aspectRatio': '2:3'}},
    }
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
                    return {'image_base64': encoded_output, 'mime_type': mime, 'generated': True}
    except (ValueError, TypeError, KeyError, OSError, Image.DecompressionBombError):
        raise HTTPException(502, 'El proveedor devolvió una imagen no válida.') from None
    raise HTTPException(422, 'La IA no devolvió una imagen. Prueba otra foto clara de una sola persona.')


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if not re.fullmatch(r'[a-zA-Z0-9._-]{1,100}', settings.model):
        raise ValueError('Modelo de imágenes inválido')
    app = FastAPI(title='El cambio IA', docs_url=None, redoc_url=None, openapi_url=None)
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
                'generation_ready': bool(settings.enabled and settings.api_key and len(settings.access_code) >= 12),
                'access_code_required': True}

    @app.post('/api/persona/generar')
    async def generate(payload: GenerateInput, request: Request):
        if not settings.enabled or not settings.api_key or len(settings.access_code) < 12:
            raise HTTPException(503, 'La generación de imágenes todavía no está habilitada.')
        supplied = request.headers.get('x-el-cambio-code', '')
        if not hmac.compare_digest(supplied.encode(), settings.access_code.encode()):
            raise HTTPException(401, 'El código de acceso del piloto no es correcto.')
        if busy.locked():
            raise HTTPException(429, 'Hay otra imagen en proceso. Espera un momento.')
        async with busy:
            encoded = await run_in_threadpool(normalize_image, payload.image_base64)
            # Se usa la IP de la conexión, nunca un X-Forwarded-For arbitrario.
            await run_in_threadpool(reserve_attempt, settings, request.client.host if request.client else 'unknown')
            result = await generate_person(settings, encoded, payload.style)
            return result
    return app

app = create_app()
