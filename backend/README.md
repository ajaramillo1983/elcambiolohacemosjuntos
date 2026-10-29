# El cambio IA — backend independiente

Este servicio no importa Centinela, no requiere sus credenciales ni accede a su base de datos.
Genera a la persona completa a partir de una foto y devuelve la imagen en memoria.
La web quita el fondo y coloca a la persona junto a las candidatas, conservando la plantilla original.

## Despliegue en el nuevo servicio de Render

- Repository: `ajaramillo1983/elcambiolohacemosjuntos`
- Branch: `main`
- Root Directory: `backend`
- Runtime: Python
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
- Health Check Path: `/api/health`

Render Environment:

- `EL_CAMBIO_GEMINI_API_KEY`: clave del proyecto Google **elcambioia**, no la de Centinela.
- `EL_CAMBIO_ACCESS_CODE`: código privado del piloto de al menos 12 caracteres. No publicar en código.
- `EL_CAMBIO_ENABLED`: `true` al iniciar las pruebas; `false` desactiva los gastos de generación.
- `EL_CAMBIO_IMAGE_MODEL`: `gemini-3.1-flash-image`, modificable si el proyecto no tiene acceso.
- `CORS_ORIGINS`: `https://ajaramillo1983.github.io`.

No se guardan fotos ni respuestas de Google en disco ni en logs. La foto se envía a Google
solo al pedir generación y consentir ese envío. Las claves se pasan en un header y no en URLs.
La clave permanece en Render; el navegador solo conoce el código de acceso del piloto.

## Límites del piloto

Una generación simultánea, 20 intentos por día UTC en total y 3 por IP de conexión.
Los intentos fallidos cuentan y no se reintentan automáticamente. La IP puede ser la del proxy
según el despliegue, por lo que el límite individual puede compartirse. CORS no es autenticación:
el código privado protege el endpoint del uso público casual.

Las cuotas viven en SQLite. Con `/tmp` sobreviven al reinicio del proceso, pero pueden perderse
al reiniciar/reemplazar la instancia o desplegar. Para un piloto público usar un disco persistente
y configurar `EL_CAMBIO_QUOTA_DB` a su ruta, o un contador externo duradero antes de retirar
el código de acceso. No usar varias instancias con este contador local. No es un límite monetario
de facturación y no limita el uso de la misma clave fuera de este servicio.

Una selfie no contiene las proporciones reales de un cuerpo completo: el resultado es sintético
y debe revisarse. El parecido y la anatomía necesitan validarse con fotos reales del piloto.

## API

`GET /api/health` informa si la generación está configurada sin revelar secretos.

`POST /api/persona/generar`, header `X-El-Cambio-Code`, JSON:

```json
{"image_base64":"BASE64_DE_LA_FOTO","mime_type":"image/jpeg","style":"Empresarias","consent":true}
```

Respuesta: `image_base64`, `mime_type` y `generated: true`. Admite los cinco estilos de la web.
No acepta URLs ni prompts arbitrarios. Máximo 8 MB y 24 megapíxeles en la imagen de entrada.

## Verificación local

```sh
pip install -r requirements.txt pytest
pytest -q
```

Las pruebas usan respuestas simuladas. No generan imágenes, no usan claves reales ni consumen crédito.
Documentación del proveedor: https://ai.google.dev/gemini-api/docs/generate-content/image-generation


## Montaje integrado (octubre 2026)
La generación recibe dos referencias: la plantilla local elegida y la foto normalizada del usuario. Devuelve una escena completa (composition: scene); el navegador muestra esa imagen sin recortarla ni pegarla sobre otro fondo. Cambiar estilo o foto invalida el resultado y exige una nueva generación explícita. La descarga permanece deshabilitada hasta tener un montaje de IA. El prompt solicita preservar identidad, candidatas y texto, pero no garantiza coincidencia facial exacta ni preservación píxel a píxel. Revisar cada resultado; no prometer rostro intacto. No hay reintentos de pago automáticos.
