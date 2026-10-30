"""Local face detection and original-pixel head matte. No generative face swap."""
import base64
import io
import threading

import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException

cv2.setNumThreads(1)
_lock = threading.Lock()
_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def decode(encoded):
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
        im.load()
        return np.array(im.convert('RGB'))


def faces(rgb):
    # Detection only: no identity recognition, embeddings or face storage.
    factor = min(1.0, 1200 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=factor, fy=factor) if factor < 1 else rgb
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    with _lock:
        found = _cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(24, 24))
    return [tuple(int(round(v / factor)) for v in box) for box in found]


def prepare_head(encoded):
    rgb = decode(encoded)
    found = faces(rgb)
    if len(found) != 1:
        raise HTTPException(422, 'Para conservar tu rostro, usa una foto nítida de una sola persona, mirando de frente o casi de frente. No se ha solicitado ninguna generación.')
    x, y, w, h = found[0]
    if w < 32 or h < 32:
        raise HTTPException(422, 'El rostro es demasiado pequeño. Sube una foto más cercana y nítida. No se ha solicitado ninguna generación.')
    # Include hair, ears and a short neck transition around the detected face.
    left, top = max(0, x - int(w * .4)), max(0, y - int(h * .65))
    right, bottom = min(rgb.shape[1], x + int(w * 1.4)), min(rgb.shape[0], y + int(h * 1.3))
    crop = rgb[top:bottom, left:right].copy()
    ch, cw = crop.shape[:2]
    fx, fy = x - left, y - top
    mask = np.full((ch, cw), cv2.GC_PR_BGD, dtype=np.uint8)
    cv2.ellipse(mask, (fx + w // 2, fy + int(h * .28)),
                (int(w * .77), int(h * .91)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    # Protect the face interior from background segmentation.
    core = np.zeros((ch, cw), np.uint8)
    cv2.ellipse(core, (fx + w // 2, fy + h // 2),
                (max(1, int(w * .43)), max(1, int(h * .50))), 0, 0, 360, 255, -1)
    mask[core > 0] = cv2.GC_FGD
    mask[:2, :] = mask[-2:, :] = mask[:, :2] = mask[:, -2:] = cv2.GC_BGD
    try:
        # Bound segmentation memory on the free server. Original RGB pixels stay full resolution.
        ratio = min(1.0, 480 / max(ch, cw))
        work = cv2.resize(crop, (max(1, round(cw * ratio)), max(1, round(ch * ratio)))) if ratio < 1 else crop
        work_mask = cv2.resize(mask, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST) if ratio < 1 else mask
        cv2.grabCut(work, work_mask, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((work_mask == cv2.GC_FGD) | (work_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if ratio < 1:
            alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
    except cv2.error:
        raise HTTPException(422, 'No se pudo separar la cabeza del fondo. Prueba una foto con el cabello y el rostro bien visibles. No se ha solicitado ninguna generación.') from None
    # Feather only the silhouette boundary. Never blend the facial interior with AI pixels.
    alpha = cv2.GaussianBlur(alpha, (3, 3), .7)
    alpha[core > 0] = 255
    # Discard disconnected/background-colored pieces caught beside cheeks or neck.
    envelope = np.zeros((ch, cw), np.uint8)
    cv2.ellipse(envelope, (fx + w // 2, fy + int(h * .43)),
                (max(1, int(w * .55)), max(1, int(h * .63))), 0, 0, 360, 255, -1)
    cv2.ellipse(envelope, (fx + w // 2, fy - int(h * .08)),
                (max(1, int(w * .72)), max(1, int(h * .55))), 0, 0, 360, 255, -1)
    envelope = cv2.GaussianBlur(envelope, (3, 3), .7)
    alpha = np.minimum(alpha, envelope)
    alpha[core > 0] = 255
    # Exclude the source shirt/collar; transition just beneath the detected chin.
    fade_start = fy + h
    fade_end = min(ch, fade_start + max(2, int(h * .07)))
    if fade_start < ch:
        for row in range(fade_start, fade_end):
            alpha[row] = (alpha[row].astype(np.float32) * (fade_end - row) / (fade_end - fade_start)).astype(np.uint8)
        alpha[fade_end:] = 0
    rgba = np.dstack((crop, alpha))
    out = io.BytesIO()
    Image.fromarray(rgba).save(out, format='PNG')
    return {'image_base64': base64.b64encode(out.getvalue()).decode(),
            'mime_type': 'image/png', 'width': cw, 'height': ch,
            'face_box': [fx, fy, w, h]}


def locate_head(scene_encoded, head):
    scene = decode(scene_encoded)
    height, width = scene.shape[:2]
    found = sorted((b for b in faces(scene) if (b[0] + b[2] / 2) < width * .5), key=lambda b: b[0])
    # If detection is uncertain, return a manual placement; never request another paid image.
    manual = not found
    target = found[0] if found else (int(width * .15), int(height * .22), int(width * .12), int(width * .12))
    tx, ty, tw, th = target
    sx, sy, sw, sh = head['face_box']
    scale = tw / sw  # uniform scale; no facial reshaping
    return {'x': tx + tw / 2 - (sx + sw / 2) * scale,
            'y': ty + th / 2 - (sy + sh / 2) * scale,
            'width': head['width'] * scale, 'height': head['height'] * scale,
            'scene_width': width, 'scene_height': height, 'manual_adjustment': manual}
