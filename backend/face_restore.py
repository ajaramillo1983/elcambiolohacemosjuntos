"""Local face detection and original-pixel head matte. No generative face swap."""
import base64
import io
import threading
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException

cv2.setNumThreads(1)
_lock = threading.Lock()
_model = str(Path(__file__).resolve().parent / 'models' / 'face_detection_yunet_2023mar.onnx')
_detector = cv2.FaceDetectorYN.create(_model, '', (320, 320), .85, .3, 5000)


def decode(encoded):
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
        im.load()
        return np.array(im.convert('RGB'))


def detections(rgb):
    # Five landmarks, not identity recognition. Model is packaged, no runtime download.
    factor = min(1.0, 960 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=factor, fy=factor) if factor < 1 else rgb
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    with _lock:
        _detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, found = _detector.detect(bgr)
    if found is None:
        return []
    return [{'box': [float(v / factor) for v in row[:4]],
             'eyes': sorted([[float(row[i] / factor), float(row[i+1] / factor)] for i in (4, 6)]),
             'score': float(row[14])} for row in found]


def faces(rgb):
    return [tuple(round(v) for v in d['box']) for d in detections(rgb)]


def prepare_head(encoded):
    rgb = decode(encoded)
    detected = detections(rgb)
    found = [tuple(round(v) for v in d['box']) for d in detected]
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
            'face_box': [fx, fy, w, h],
            'eyes': [[ex-left, ey-top] for ex,ey in detected[0]['eyes']]}


def locate_head(scene_encoded, head):
    scene = decode(scene_encoded)
    height, width = scene.shape[:2]
    # Only the foreground group in the upper half. This excludes background guests
    # and the low false positives that previously created a floating head.
    people = [d for d in detections(scene)
              if .04 * width <= d['box'][2] <= .25 * width
              and .01 * height <= d['box'][1] <= .55 * height]
    people = sorted(people, key=lambda d: d['box'][2] * d['box'][3], reverse=True)[:3]
    people.sort(key=lambda d: d['box'][0] + d['box'][2] / 2)
    if len(people) != 3:
        raise HTTPException(422, 'No se pudo encajar automáticamente el rostro con seguridad. Usa una foto frontal y nítida. No se ha solicitado otra generación automática.')
    target = people[0]
    centers = [d['box'][1] + d['box'][3]/2 for d in people]
    if max(centers)-min(centers) > .18*height:
        raise HTTPException(422, 'La escena no permite un encaje seguro del rostro. No se ha solicitado otra generación automática.')
    source_eyes = np.array(head['eyes'], dtype=np.float64)
    target_eyes = np.array(target['eyes'], dtype=np.float64)
    sv, tv = source_eyes[1]-source_eyes[0], target_eyes[1]-target_eyes[0]
    source_distance, target_distance = float(np.linalg.norm(sv)), float(np.linalg.norm(tv))
    if source_distance < 8 or target_distance < 8:
        raise HTTPException(422, 'No hay suficiente detalle en los ojos para conservar el rostro con precisión.')
    scale = target_distance/source_distance
    theta = math.atan2(tv[1],tv[0])-math.atan2(sv[1],sv[0])
    if abs(theta) > math.radians(25):
        raise HTTPException(422, 'La orientación de la cabeza generada difiere demasiado de la foto original.')
    rotation = np.array([[math.cos(theta),-math.sin(theta)], [math.sin(theta),math.cos(theta)]]) * scale
    shift = target_eyes.mean(axis=0) - rotation @ source_eyes.mean(axis=0)
    matrix = np.column_stack((rotation,shift))
    return {'matrix': matrix.tolist(), 'scene_width': width, 'scene_height': height,
            'automatic': True, 'target_box': target['box']}


def compose_original_head(scene_encoded, head, placement):
    # Final composition occurs here, so preview, saved file and share use the same image.
    scene = Image.fromarray(decode(scene_encoded)).convert('RGBA')
    patch = np.array(Image.open(io.BytesIO(base64.b64decode(head['image_base64']))).convert('RGBA'))
    matrix = np.array(placement['matrix'], dtype=np.float64)
    # Premultiplied alpha avoids colored halos from transparent source pixels.
    alpha = patch[:,:,3:4].astype(np.float32)/255
    premultiplied = np.concatenate((patch[:,:,:3].astype(np.float32)*alpha, alpha),axis=2)
    warped = cv2.warpAffine(premultiplied,matrix,(scene.width,scene.height),flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    a = warped[:,:,3:4]
    rgb = warped[:,:,:3] + np.array(scene)[:,:,:3].astype(np.float32)*(1-a)
    out=io.BytesIO()
    Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).save(out,format='PNG')
    return base64.b64encode(out.getvalue()).decode('ascii')
