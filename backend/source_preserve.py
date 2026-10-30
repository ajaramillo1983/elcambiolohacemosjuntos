"""Preserve the complete visible person. No isolated head transplant."""
import base64
import io
import math
import numpy as np
import cv2
from PIL import Image, ImageOps
from fastapi import HTTPException
try:
    from .face_restore import detections, decode, tighten_left_margin
except ImportError:
    from face_restore import detections, decode, tighten_left_margin

GUIDE_W, GUIDE_H = 1024, 1536
_verified = False

def png64(image):
    out = io.BytesIO()
    image.save(out, format='PNG')
    return base64.b64encode(out.getvalue()).decode('ascii')

def prepare_source(encoded):
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
        source = ImageOps.exif_transpose(im).convert('RGBA')
    rgba = np.array(source)
    alpha = rgba[:,:,3]
    if (alpha < 20).mean() < .02 or (alpha > 200).mean() < .005:
        raise HTTPException(422, 'Primero debe quitarse el fondo de la foto. No se ha solicitado ninguna generación.')
    found = detections(rgba[:,:,:3])
    found = [d for d in found if alpha[min(source.height-1,max(0,int(d['box'][1]+d['box'][3]/2))), min(source.width-1,max(0,int(d['box'][0]+d['box'][2]/2)))] > 100]
    if len(found) != 1:
        raise HTTPException(422, 'Usa una foto nítida de una sola persona con el rostro visible. No se ha solicitado ninguna generación.')
    ys, xs = np.where(alpha > 20)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    patch = source.crop((left,top,right,bottom))
    eye_points = [[x-left,y-top] for x,y in found[0]['eyes']]
    distance = float(np.linalg.norm(np.array(eye_points[1])-eye_points[0]))
    if distance < 8:
        raise HTTPException(422, 'El rostro es demasiado pequeño. Usa una foto más cercana. No se ha solicitado ninguna generación.')
    face = found[0]['box']
    ratio = patch.height / max(1,face[3])
    coverage = 'full_or_long' if ratio > 6 else 'partial' if ratio > 3 else 'portrait'
    return {'image_base64':png64(patch), 'width':patch.width,'height':patch.height,
            'eyes':eye_points,'coverage':coverage,
            'face_box':[face[0]-left,face[1]-top,face[2],face[3]]}

def apply_office_pose(source, style):
    """Release only the inward shoulder/arm for the requested office gesture.

    The same matte is used in the guide and final composition, so the preserved
    old arm cannot be pasted over the newly generated hand. Face and neck stay opaque.
    """
    if style != 'Empresarias':
        return source
    patch = np.array(Image.open(io.BytesIO(base64.b64decode(source['image_base64']))).convert('RGBA'))
    fx, fy, fw, fh = source['face_box']
    yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
    def smooth(t):
        t=np.clip(t,0,1)
        return t*t*(3-2*t)
    inward = smooth((xx-(fx+fw*.80))/max(1,fw*.35))
    below_neck = smooth((yy-(fy+fh*1.35))/max(1,fh*.30))
    above_legs = 1-smooth((yy-(fy+fh*5.0))/max(1,fh*.65))
    released=inward*below_neck*above_legs
    patch[:,:,3]=np.rint(patch[:,:,3]*(1-released)).astype(np.uint8)
    edited=dict(source)
    edited['image_base64']=png64(Image.fromarray(patch))
    edited['pose_edit']='office_hand_on_cream_shoulder'
    return edited


def create_guide(template_path, source):
    with Image.open(template_path) as im:
        template = im.convert('RGB')
    # Anchor the existing women to the right and trim only surplus left background.
    k = max(GUIDE_W/template.width, GUIDE_H/template.height)
    template = template.resize((round(template.width*k),round(template.height*k)),Image.Resampling.LANCZOS)
    template = template.crop((template.width-GUIDE_W,0,template.width,GUIDE_H)).convert('RGBA')
    faces = sorted(detections(np.array(template)[:,:,:3]),key=lambda d:d['box'][2]*d['box'][3],reverse=True)[:2]
    if len(faces) != 2:
        raise HTTPException(503,'No se pudo preparar la plantilla. No se ha solicitado ninguna generación.')
    nearest = min(faces,key=lambda d:d['box'][0])
    standing = min(faces,key=lambda d:sum(e[1] for e in d['eyes'])/2)
    ref = np.array(standing['eyes'],dtype=float)
    src = np.array(source['eyes'],dtype=float)
    span = float(np.linalg.norm(ref[1]-ref[0]))*.97
    scale = span/float(np.linalg.norm(src[1]-src[0]))
    target = np.array([nearest['box'][0] + nearest['box'][2]/2 - nearest['box'][2]*1.45,
                       ref[:,1].mean()+GUIDE_H*.012])
    # Keep the source orientation intact; adapt synthetic anatomy to it.
    if source['coverage']=='full_or_long':
        scale = min(scale,(GUIDE_H-35-target[1])/max(1,source['height']-src[:,1].mean()))
    center = src.mean(axis=0)
    shift = target - center*scale
    # Include the whole original person; do not clip hands/arms at the left frame edge.
    shift[0] = max(12,shift[0])
    matrix = [[scale,0,float(shift[0])],[0,scale,float(shift[1])]]
    placement={'matrix':matrix,'scene_width':GUIDE_W,'scene_height':GUIDE_H,'automatic':True,
               'target_eyes':(src*scale+shift).tolist(),'coverage':source['coverage'],
               'target_box':[source['face_box'][0]*scale+shift[0],source['face_box'][1]*scale+shift[1],
                             source['face_box'][2]*scale,source['face_box'][3]*scale]}
    guide = composite(template,source,matrix)
    return png64(guide), placement

def composite(background,source,matrix):
    patch = np.array(Image.open(io.BytesIO(base64.b64decode(source['image_base64']))).convert('RGBA'))
    a = patch[:,:,3:4].astype(np.float32)/255
    pm = np.concatenate((patch[:,:,:3].astype(np.float32)*a,a),axis=2)
    warped=cv2.warpAffine(pm,np.array(matrix,dtype=float),(background.width,background.height),
                         flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    rgb=warped[:,:,:3]+np.array(background.convert('RGB')).astype(np.float32)*(1-warped[:,:,3:4])
    return Image.fromarray(np.clip(rgb,0,255).astype(np.uint8))

def finish_scene(scene_encoded,source,placement):
    scene=Image.fromarray(decode(scene_encoded))
    k=min(scene.width/GUIDE_W,scene.height/GUIDE_H)
    offset=np.array([(scene.width-GUIDE_W*k)/2,(scene.height-GUIDE_H*k)/2])
    expected=np.array(placement['target_eyes'])*k+offset
    detected=detections(np.array(scene))
    if not detected:
        raise HTTPException(422,'La IA no respetó la guía de composición. No se ha solicitado otro intento automático.')
    candidate=min(detected,key=lambda d:float(np.linalg.norm(np.array(d['eyes']).mean(axis=0)-expected.mean(axis=0))))
    delta=float(np.linalg.norm(np.array(candidate['eyes']).mean(axis=0)-expected.mean(axis=0)))
    if delta>max(12,placement['target_box'][2]*k*.45):
        raise HTTPException(422,'La IA desplazó a la persona fuera de la guía. No se ha solicitado otra generación automática.')
    matrix=np.array(placement['matrix'],dtype=float)*k
    matrix[:,2]+=offset
    result=composite(scene,source,matrix)
    box=np.array(placement['target_box'])*k
    box[:2]+=offset
    final=tighten_left_margin(np.array(result),box.tolist())
    return png64(Image.fromarray(final))

def verify_pipeline():
    global _verified
    if _verified:
        return
    # Startup gate: checks transparency, continuity and original pixels; no external API calls.
    patch=Image.new('RGBA',(20,40),(121,82,53,255))
    source={'image_base64':png64(patch)}
    result=np.array(composite(Image.new('RGB',(60,80),'white'),source,[[1,0,10],[0,1,12]]))
    if tuple(result[20,15])!=(121,82,53) or tuple(result[45,15])!=(121,82,53) or tuple(result[0,0])!=(255,255,255):
        raise RuntimeError('Source preservation self-check failed')

    from pathlib import Path
    root=Path(__file__).resolve().parent.parent/'imagenes'/'ampliadas'
    source.update({'width':20,'height':40,'eyes':[[5,8],[15,8]],'face_box':[3,2,14,16],'coverage':'portrait'})
    for style in ('Empresarias','Deportistas','Graduadas','De_gala','Viajeras'):
        guide,placement=create_guide(root/(style+'.png'),source)
        if not guide or not placement['automatic']:
            raise RuntimeError('Template preparation self-check failed')
    _verified=True
    print('Source-region-v3: original pixels and five templates verified.',flush=True)
