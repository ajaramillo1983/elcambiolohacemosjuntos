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
    # Plantilla confiable del repositorio; ninguna URL ni ruta proviene del cliente.
    template_path = Path(__file__).resolve().parent.parent / 'imagenes' / 'ampliadas' / f'{style}.png'
    if style not in STYLES or not template_path.is_file():
        raise HTTPException(503, 'La plantilla de este estilo no está disponible.')
    template = base64.b64encode(template_path.read_bytes()).decode('ascii')
    prompt = (
        'Make ONE finished professional photorealistic group photograph using both references. '
        'IMAGE 1 is the scene template with TWO existing women and a campaign slogan. '
        'IMAGE 2 is the identity reference of the ONE additional person to integrate. '
        'Keep the scene framing, aspect ratio, two existing women, their faces, clothes, '
        'poses and the exact slogan and logo of IMAGE 1 unchanged. Do not crop the slogan. '
        'Add the person from IMAGE 2 immediately to the LEFT of the two women on the SAME depth plane, '
        'without covering their faces or the slogan. Exactly THREE people in the final scene. '
        'Highest priority: faithfully retain the added person\'s facial identity, exact facial features, '
        'expression, apparent age, skin tone, hairline and hairstyle from IMAGE 2. '
        'Do not beautify, slim the face, rejuvenate, average faces or borrow the other women\'s features. '
        'Remove the source background completely. Dress the added person in '
        f'{STYLES[style]}. '
        'If cropped or a selfie, complete the missing body with plausible age-appropriate anatomy '
        'and natural proportions consistent with visible shoulders and build; do not stretch the selfie. '
        'Show a complete natural body and shoes within the frame. Relaxed hands, correct fingers. '
        'Match the scene camera height, perspective, ground plane and the apparent head sizes of '
        'the existing STANDING woman. The added person must not look oversized or miniature. '
        'GROUP ALIGNMENT IS ESSENTIAL: position the added person so their eye line and top of head '
        'must NOT be higher than those of the nearest existing standing woman. Aim for the added '
        'person\'s crown 1 to 2 percent of image height BELOW that reference crown, with a similar '
        'natural head size. If the nearest woman is seated, use the other standing woman for height. '
        'Do not put the added person closer to the camera, on a step or higher on the floor. '
        'Fit the entire added body proportionally to that reference scale; never stretch or squash '
        'the face, head, torso or legs independently to achieve alignment. '
        'CLOSE GROUP POSE WITH NATURAL OCCLUSION: move the added person inward toward the center '
        'until the nearest existing woman\'s outward elbow is visibly IN FRONT OF the added '
        'person\'s inner upper arm or side of torso. The added person stands immediately beside '
        'and slightly BEHIND that elbow, not in a separate column on the far left. '
        'Their silhouettes should overlap naturally by about one upper-arm width at the elbow; '
        'do not leave visible background between that elbow and the added person\'s torso. '
        'Keep the existing elbow and forearm unchanged and correctly in front; do not erase them, '
        'bend them around the added person or fuse skin and clothing. Keep all three faces clear. '
        'Where the nearest woman has no outward elbow, use slight shoulder/upper-arm occlusion '
        'with the existing woman in front to achieve the same compact shared pose. '
        'Use a subtle inward turn or lean and relaxed arms so all three look like friends posing '
        'for ONE shared photo. Preserve distinct, anatomically correct limbs despite the natural occlusion. '
        'The existing two women keep their positions; move and scale ONLY the added person. '
        'Match directional light, white balance, exposure, color temperature, grain, sharpness and '
        'depth of field. Add realistic ground contact shadows and ambient occlusion. '
        'For the beach use warm sunset rim light and feet resting naturally on the sand; for other '
        'scenes match their floor and lighting. No floating feet, cut edges, halos, sticker look, '
        'distorted limbs, duplicated parts or isolated studio portrait. '
        'Return the COMPLETE integrated group scene, not a cutout or a separate person. '
        'Output one portrait 4:5 image with the entire original template composition visible.'
    )
    body = {
        'contents': [{'parts': [
            {'text': prompt},
            {'text': 'IMAGE 1: scene to preserve and edit'},
            {'inlineData': {'mimeType': 'image/png', 'data': template}},
            {'text': 'IMAGE 2: identity reference of the additional person'},
            {'inlineData': {'mimeType': 'image/jpeg', 'data': encoded}},
        ]}],
        'generationConfig': {'responseModalities': ['TEXT', 'IMAGE'], 'imageConfig': {'aspectRatio': '4:5'}},
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
                    return {'image_base64': encoded_output, 'mime_type': mime, 'generated': True, 'composition': 'scene', 'style': style}
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
                'access_code_required': True, 'composition': 'scene'}

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
