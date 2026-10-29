import base64
import io
from dataclasses import replace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import main


def picture():
    out = io.BytesIO()
    Image.new('RGB', (40, 80), '#bd9176').save(out, format='PNG')
    return base64.b64encode(out.getvalue()).decode()


@pytest.fixture
def settings(tmp_path):
    return main.Settings(api_key='TEST_ONLY_NOT_A_REAL_KEY', access_code='test-pilot-code-12345',
                         enabled=True, quota_db=str(tmp_path / 'quota.sqlite3'))


def payload(**overrides):
    return dict(image_base64=picture(), mime_type='image/png', style='Empresarias', consent=True) | overrides


def test_disabled_is_safe_and_health_has_no_secret(settings):
    app = main.create_app(replace(settings, enabled=False))
    with TestClient(app) as c:
        r = c.get('/api/health')
        assert r.json()['generation_ready'] is False
        assert settings.api_key not in r.text
        assert c.post('/api/persona/generar', json=payload()).status_code == 503


def test_requires_pilot_code(settings):
    with TestClient(main.create_app(settings)) as c:
        r = c.post('/api/persona/generar', json=payload())
        assert r.status_code == 401
        assert settings.access_code not in r.text


def test_validates_image_consent_and_style(settings):
    with TestClient(main.create_app(settings)) as c:
        for bad in [{'consent': False}, {'style': 'unknown'}, {'image_base64': 'not an image'}]:
            r = c.post('/api/persona/generar', json=payload(**bad), headers={'X-El-Cambio-Code': settings.access_code})
            assert r.status_code == 422
        assert c.post('/api/persona/generar', json=payload(prompt='arbitrary'), headers={'X-El-Cambio-Code': settings.access_code}).status_code == 422


def test_success_and_per_person_limit(settings, monkeypatch):
    calls = []
    async def fake(settings, encoded, style):
        calls.append(style)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:
            assert im.format == 'JPEG'
        return {'image_base64': picture(), 'mime_type': 'image/png', 'generated': True}
    monkeypatch.setattr(main, 'generate_person', fake)
    with TestClient(main.create_app(settings)) as c:
        for _ in range(3):
            r = c.post('/api/persona/generar', json=payload(), headers={'X-El-Cambio-Code': settings.access_code})
            assert r.status_code == 200
            assert r.json()['generated'] is True
            assert r.headers['Cache-Control'] == 'no-store'
        r = c.post('/api/persona/generar', json=payload(), headers={'X-El-Cambio-Code': settings.access_code})
        assert r.status_code == 429
        assert len(calls) == 3


def test_global_quota_is_shared_and_persisted(settings):
    settings = replace(settings, daily_limit=2)
    main.reserve_attempt(settings, 'one')
    main.reserve_attempt(settings, 'two')
    with pytest.raises(HTTPException) as e:
        main.reserve_attempt(settings, 'three')
    assert e.value.status_code == 429


def test_cors_allowed_origin(settings):
    with TestClient(main.create_app(settings)) as c:
        r = c.options('/api/persona/generar', headers={'Origin': 'https://ajaramillo1983.github.io', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'x-el-cambio-code,content-type'})
        assert r.status_code == 200
        assert r.headers['Access-Control-Allow-Origin'] == 'https://ajaramillo1983.github.io'
        r = c.get('/api/health', headers={'Origin': 'https://untrusted.invalid'})
        assert 'Access-Control-Allow-Origin' not in r.headers


def test_stream_body_limit(settings, monkeypatch):
    monkeypatch.setattr(main, 'MAX_BODY', 10)
    with TestClient(main.create_app(settings)) as c:
        assert c.post('/api/persona/generar', content=b'x' * 11).status_code == 413


@pytest.mark.parametrize('code,expected', [(401,503),(403,503),(404,503),(429,429),(500,502)])
def test_google_errors_are_sanitized(settings, monkeypatch, code, expected):
    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            assert settings.api_key not in url
            assert kwargs['headers']['x-goog-api-key'] == settings.api_key
            return httpx.Response(code, json={'error': {'message': settings.api_key}})
    monkeypatch.setattr(main.httpx, 'AsyncClient', FakeClient)
    import asyncio
    with pytest.raises(HTTPException) as e:
        asyncio.run(main.generate_person(settings, picture(), 'Viajeras'))
    assert e.value.status_code == expected
    assert settings.api_key not in e.value.detail


def test_google_image_response(settings, monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            assert kwargs['json']['generationConfig']['responseModalities'] == ['TEXT', 'IMAGE']
            return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'inlineData':{'mimeType':'image/png','data':picture()}}]}}]})
    monkeypatch.setattr(main.httpx, 'AsyncClient', FakeClient)
    import asyncio
    result=asyncio.run(main.generate_person(settings,picture(),'Graduadas'))
    assert result['generated'] is True
