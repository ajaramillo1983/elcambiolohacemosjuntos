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
    async def fake(settings, encoded, style, **kwargs):
        calls.append(style)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:
            assert im.format == 'PNG'
        return {'image_base64': picture(), 'mime_type': 'image/png', 'generated': True}
    monkeypatch.setattr(main, 'generate_person', fake)
    monkeypatch.setattr(main, 'prepare_source', lambda encoded: {'image_base64': picture()})
    monkeypatch.setattr(main, 'apply_office_pose', lambda source, style: source)
    monkeypatch.setattr(main, 'create_guide', lambda path, source: ('TEST_GUIDE', {'automatic':True}))
    monkeypatch.setattr(main, 'locate_head', lambda encoded, head: {'automatic': True})
    monkeypatch.setattr(main, 'finish_scene', lambda encoded, source, placement: encoded)
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
            assert body['generationConfig']['imageConfig']['aspectRatio'] == '2:3'
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
    monkeypatch.setattr(face_restore, 'detections', lambda im: [{'box':[35,35,50,60],'eyes':[[47,55],[73,55]],'score':.99}])
    head = face_restore.prepare_head(base64.b64encode(out.getvalue()).decode())
    patch = np.array(Image.open(io.BytesIO(base64.b64decode(head['image_base64']))))
    fx, fy, w, h = head['face_box']
    assert tuple(patch[fy+30, fx+25, :3]) == tuple(rgb[65,60])
    assert patch[fy+30, fx+25, 3] == 255
    assert patch[-1,:,3].max() == 0


def test_head_alignment_excludes_floating_face_and_matches_eyes(monkeypatch):
    import face_restore
    import numpy as np
    out=io.BytesIO(); Image.new('RGB',(160,240),'white').save(out,format='PNG')
    scene=base64.b64encode(out.getvalue()).decode()
    def detection(x,y):
        return {'box':[x,y,24,32],'eyes':[[x+6,y+12],[x+18,y+12]],'score':.99}
    detected=[detection(40,35),detection(75,35),detection(110,35),detection(5,190)]
    monkeypatch.setattr(face_restore,'detections',lambda im:detected)
    head={'eyes':[[10,20],[34,20]]}
    p=face_restore.locate_head(scene,head)
    assert p['automatic'] is True
    assert p['target_box'][1]==35
    matrix=np.array(p['matrix'])
    mapped=(matrix[:,:2]@np.array(head['eyes']).T).T+matrix[:,2]
    assert np.allclose(mapped,detected[0]['eyes'])
    assert np.allclose(matrix[:,:2].T@matrix[:,:2],np.eye(2)*.25)
    monkeypatch.setattr(face_restore,'detections',lambda im:detected[:2])
    with pytest.raises(HTTPException) as e:
        face_restore.locate_head(scene,head)
    assert e.value.status_code==422  # no guessed default or floating head


def test_server_composition_keeps_source_pixels():
    import face_restore
    out=io.BytesIO();Image.new('RGBA',(10,10),(10,20,30,255)).save(out,format='PNG')
    head={'image_base64':base64.b64encode(out.getvalue()).decode()}
    result=face_restore.compose_original_head(picture(),head,{'matrix':[[1,0,4],[0,1,6]]})
    im=Image.open(io.BytesIO(base64.b64decode(result)))
    assert im.getpixel((8,10))==(10,20,30)
    assert im.getpixel((0,0))==(189,145,118)


def test_no_source_face_rejected_before_paid_generation(settings,monkeypatch):
    from fastapi import HTTPException
    def reject(encoded):
        raise HTTPException(422, 'No face')
    monkeypatch.setattr(main,'prepare_source',reject)
    async def should_not_call(*args):
        raise AssertionError('Provider must not be called')
    monkeypatch.setattr(main,'generate_person',should_not_call)
    with TestClient(main.create_app(settings)) as c:
        r=c.post('/api/persona/generar',json=payload(),headers={'X-El-Cambio-Code':settings.access_code})
        assert r.status_code==422
    assert not __import__('pathlib').Path(settings.quota_db).exists()


def test_tight_crop_preserves_left_edge_of_colored_slogan():
    import face_restore
    import numpy as np
    rgb=np.zeros((900,600,3),np.uint8)
    rgb[20:110,40:560]=(200,0,0)
    cropped=face_restore.tighten_left_margin(rgb,[300,300,60,80])
    assert cropped.shape==(900,572,3)
    assert np.array_equal(cropped[20:110,12:532],rgb[20:110,40:560])
    blank=np.zeros_like(rgb)
    assert face_restore.tighten_left_margin(blank,[300,300,60,80]).shape==blank.shape


def test_render_frontend_serves_automatic_ui_and_no_credentials(settings):
    with TestClient(main.create_app(settings)) as c:
        r=c.get('/')
        assert r.status_code==200
        assert 'source-region-v3' in r.text
        assert 'id="headX"' not in r.text
        assert settings.api_key not in r.text
        cfg=c.get('/backend-config.js')
        assert cfg.status_code==200
        assert 'window.location.origin' in cfg.text
        assert settings.access_code not in cfg.text


def test_source_region_startup_gate_and_original_neck_pixels():
    import source_preserve
    source_preserve.verify_pipeline()

def test_normalization_keeps_png_alpha():
    out=io.BytesIO()
    im=Image.new('RGBA',(50,100),(0,0,0,0))
    for y in range(10,90):
        for x in range(10,40):
            im.putpixel((x,y),(112,78,51,255))
    im.save(out,format='PNG')
    normalized=main.normalize_image(base64.b64encode(out.getvalue()).decode())
    result=Image.open(io.BytesIO(base64.b64decode(normalized)))
    assert result.mode=='RGBA'
    assert result.getpixel((0,0))[3]==0
    assert result.getpixel((20,60))==(112,78,51,255)



def test_office_pose_releases_arm_but_protects_face_neck_and_other_styles():
    import source_preserve
    import numpy as np
    rgba=Image.new('RGBA',(200,800),(121,82,53,255))
    out=io.BytesIO();rgba.save(out,format='PNG')
    source={'image_base64':base64.b64encode(out.getvalue()).decode(),'face_box':[60,40,70,90]}
    edited=source_preserve.apply_office_pose(source,'Empresarias')
    patch=np.array(Image.open(io.BytesIO(base64.b64decode(edited['image_base64']))))
    assert np.array_equal(patch[:160],np.array(rgba)[:160])  # face and neck stay intact
    assert patch[300,190,3]==0  # inward arm is available for generation
    assert patch[300,50,3]==255  # other arm/body remain protected
    assert patch[650,190,3]==255  # legs remain protected
    assert np.array_equal(patch[:,:,:3],np.array(rgba)[:,:,:3])
    assert source_preserve.apply_office_pose(source,'Viajeras') is source
    assert source_preserve.apply_office_pose(source,'Deportistas') is source
