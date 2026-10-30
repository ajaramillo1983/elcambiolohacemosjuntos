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
            assert im.format == 'PNG'
        return {'image_base64': picture(), 'mime_type': 'image/png', 'generated': True}
    monkeypatch.setattr(main, 'generate_person', fake)
    monkeypatch.setattr(main, 'prepare_head', lambda encoded: {'image_base64': picture()})
    monkeypatch.setattr(main, 'locate_head', lambda encoded, head: {'x': 0, 'y': 0})
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
            body = kwargs['json']
            assert body['generationConfig']['responseModalities'] == ['TEXT', 'IMAGE']
            assert body['generationConfig']['imageConfig']['aspectRatio'] == '4:5'
            images = [p['inlineData'] for p in body['contents'][0]['parts'] if 'inlineData' in p]
            assert len(images) == 2
            assert images[0]['mimeType'] == 'image/png'
            assert base64.b64decode(images[0]['data']).startswith(b'\x89PNG')
            assert images[1]['data'] == picture()
            return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'inlineData':{'mimeType':'image/png','data':picture()}}]}}]})
    monkeypatch.setattr(main.httpx, 'AsyncClient', FakeClient)
    import asyncio
    result=asyncio.run(main.generate_person(settings,picture(),'Graduadas'))
    assert result['generated'] is True

    assert result['composition'] == 'scene'
    assert result['style'] == 'Graduadas'


def test_original_head_pixels_are_not_generated(monkeypatch):
    import numpy as np
    import face_restore
    # Color pattern makes any replacement/blending of the facial interior detectable.
    rgb = np.zeros((160, 120, 3), dtype=np.uint8)
    rgb[:] = (180, 210, 230)
    rgb[35:95, 35:85] = (110, 70, 50)
    rgb[50:80, 45:75] = (203, 151, 109)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format='PNG')
    monkeypatch.setattr(face_restore, 'faces', lambda im: [(35, 35, 50, 60)])
    head = face_restore.prepare_head(base64.b64encode(out.getvalue()).decode())
    patch = np.array(Image.open(io.BytesIO(base64.b64decode(head['image_base64']))))
    fx, fy, w, h = head['face_box']
    assert tuple(patch[fy+30, fx+25, :3]) == tuple(rgb[65,60])
    assert patch[fy+30, fx+25, 3] == 255
    assert patch[-1,:,3].max() == 0


def test_head_placement_uses_uniform_scale_and_left_subject(monkeypatch):
    import face_restore
    monkeypatch.setattr(face_restore, 'faces', lambda im: [(20,10,8,8),(60,10,8,8)])
    h = {'face_box':[4,6,16,16], 'width':28, 'height':36}
    p = face_restore.locate_head(picture(),h)
    # source picture is 40 wide: detection at x20 is outside left half; manual fallback.
    assert p['manual_adjustment'] is True
    monkeypatch.setattr(face_restore, 'faces', lambda im: [(5,10,8,8),(25,10,8,8)])
    p = face_restore.locate_head(picture(),h)
    assert p['manual_adjustment'] is False
    assert p['width']/h['width'] == p['height']/h['height'] == .5


def test_no_source_face_rejected_before_paid_generation(settings,monkeypatch):
    from fastapi import HTTPException
    def reject(encoded):
        raise HTTPException(422, 'No face')
    monkeypatch.setattr(main,'prepare_head',reject)
    async def should_not_call(*args):
        raise AssertionError('Provider must not be called')
    monkeypatch.setattr(main,'generate_person',should_not_call)
    with TestClient(main.create_app(settings)) as c:
        r=c.post('/api/persona/generar',json=payload(),headers={'X-El-Cambio-Code':settings.access_code})
        assert r.status_code==422
    assert not __import__('pathlib').Path(settings.quota_db).exists()
