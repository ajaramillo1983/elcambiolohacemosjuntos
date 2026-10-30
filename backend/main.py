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
    prompt = "Crea una única fotografía grupal fotorrealista utilizando las dos imágenes de referencia.\n\nREFERENCIAS\nLa imagen 1 contiene la escena, las dos mujeres y el eslogan. La imagen 2 contiene a la persona que debes incorporar.\n\nPRIORIDAD MÁXIMA: CONSERVAR EL ROSTRO DE LA IMAGEN 2\nLa persona incorporada debe conservar su identidad facial. Trata su rostro como una región protegida: modifica la vestimenta, completa el cuerpo y adapta la composición alrededor de él, sin rediseñarlo.\nConserva la forma y las proporciones del rostro; la distancia, forma y tamaño de los ojos; las cejas; la nariz; los labios; la sonrisa y los dientes visibles; los pómulos; la mandíbula; el mentón; la frente; las orejas visibles; la línea del cabello y las asimetrías naturales.\nMantén la edad aparente, la expresión, la dirección de la mirada, la textura de la piel y los rasgos particulares visibles en la fotografía original. No rejuvenezcas, embellezcas, estilices, afines, suavices ni simetrices el rostro. No cambies la sonrisa ni inventes dientes. No combines sus facciones con las de las otras mujeres ni sustituyas su cara por una versión aproximada.\nConserva la orientación original de la cabeza. Adapta la postura del cuerpo a esa orientación, en lugar de girar o reconstruir el rostro. Si necesitas ajustar el tamaño, escala proporcionalmente la cabeza completa: nunca modifiques sus proporciones internas.\nEn el rostro, limita los ajustes a exposición, balance de blancos y color para armonizarlo con la escena, sin alterar sus rasgos ni borrar su textura. Si existe un conflicto entre una integración estética y la fidelidad facial, prioriza la fidelidad facial.\n\nCUERPO Y VESTIMENTA\nViste a la persona de acuerdo con el estilo seleccionado: ropa empresarial para Empresarias; vestimenta deportiva para Deportistas; toga y birrete para Graduadas; traje o vestido formal para De Gala; ropa apropiada para la playa para Viajeras.\nConserva la complexión visible. Si faltan partes del cuerpo, complétalas con anatomía y proporciones naturales, coherentes con lo que muestra la fotografía. No adelgaces ni exageres el cuerpo. Genera manos y extremidades anatómicamente correctas.\n\nPOSICIÓN Y CERCANÍA\nColoca a la persona a la izquierda, muy próxima al grupo y ligeramente detrás del codo de la mujer más cercana. Ese codo debe quedar naturalmente por delante de parte de su brazo o torso, sin fusionar las extremidades ni cubrir ningún rostro.\nSu cabeza no debe sobresalir por encima de la mujer de pie más cercana: sitúala ligeramente más abajo, manteniendo un tamaño de cabeza y una escala corporal coherentes. Si la mujer más cercana está sentada, utiliza como referencia de altura a la otra mujer de pie.\nLas tres deben parecer estar posando juntas para la misma fotografía. Conserva la cercanía del grupo, con una postura relajada y una ligera orientación corporal hacia el centro.\n\nINTEGRACIÓN Y CONSERVACIÓN DE LA ESCENA\nAdapta la iluminación del cuerpo y la ropa, las sombras de contacto, la perspectiva, el color, la nitidez y el apoyo de los pies al entorno. Evita bordes de recorte, halos, extremidades deformadas y apariencia de figura pegada.\nConserva las posiciones, los rostros, la ropa y las poses de las dos mujeres de la imagen 1. Conserva el eslogan completo, sus letras y el logotipo.\nDevuelve una sola imagen con la escena grupal completa. La persona añadida debe ser reconocible por el mismo rostro de la imagen 2, sin reinterpretación estética." + '\nEstilo seleccionado: ' + style.replace('_', ' ')
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
                'generation_ready': bool(settings.enabled and settings.api_key and bool(settings.access_code)),
                'access_code_required': True, 'composition': 'scene'}

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
            # Se usa la IP de la conexión, nunca un X-Forwarded-For arbitrario.
            await run_in_threadpool(reserve_attempt, settings, request.client.host if request.client else 'unknown')
            result = await generate_person(settings, encoded, payload.style)
            return result
    return app

app = create_app()
