"""Download the meme templates in memes_rough.csv from Imgflip, crop multi-panel ones to a single
face, keep only images where MediaPipe finds a face, and write the cleaned list to memes.csv."""
import subprocess, cv2, numpy as np, pandas as pd, mediapipe as mp
from pathlib import Path
from mediapipe.tasks.python import vision, BaseOptions

# id -> (Imgflip file, optional crop x0,y0,x1,y1 isolating one face/panel)
SOURCES = {
    'hide_the_pain_harold': ('gk5el.jpg', (0, 300, 480, 601)),
    'disaster_girl': ('23ls.jpg', None), 'success_kid': ('7bk2.jpg', None),
    'leo_cheers': ('39t1o.jpg', None), 'laughing_leo': ('4acd7j.png', None),
    'overly_attached_gf': ('25w8.jpg', None), 'condescending_wonka': ('1bim.jpg', None),
    'bad_luck_brian': ('1bip.jpg', None), 'dafoe_scientist': ('27qxmb.jpg', None),
    'roll_safe': ('1h7in3.jpg', None), 'oprah_car': ('gtj5t.jpg', None),
    'drake_approve': ('30b1gx.jpg', (0, 600, 600, 1200)),
    'drake_disapprove': ('30b1gx.jpg', (0, 0, 600, 600)),
    'blinking_guy': ('1lhvhp.jpg', (312, 0, 620, 372)),
    'cage_you_dont_say': ('ymso9.jpg', None), 'steve_harvey_shocked': ('16e0ir.jpg', None),
    'chris_pratt_surprised': ('t625c.jpg', None), 'ermahgerd': ('7hvst.jpg', None),
    'kombucha_disgust': ('3cpw06.png', (0, 0, 1000, 1473)),
    'kombucha_intrigued': ('3cpw06.png', (1000, 0, 2000, 1473)),
    'nick_young_confused': ('1jmzdm.jpg', None),
    'confused_math_lady': ('1tbys7.jpg', (400, 170, 656, 440)),
    'side_eye_chloe': ('7h6g4.jpg', None), 'skeptical_kid': ('265k.jpg', None),
    'skeptical_baby': ('26hb.jpg', None), 'rdj_eye_roll': ('5mcpl.jpg', None),
    'crying_jordan': ('ywfht.jpg', None), 'crying_kim': ('151biz.jpg', None),
    'woman_yelling': ('345v97.jpg', (0, 120, 208, 400)),
    'sad_keanu': ('26h9.jpg', None), 'pablo_waiting': ('1c1uej.jpg', None),
    'gordon_angry': ('115ah.jpg', None), 'michael_scott_no': ('34uf29.png', None),
    'jordan_peele_sweat': ('2bp4z7.jpg', None), 'jim_camera_look': ('26faci.jpg', None),
    'bernie_asking': ('3oevdk.jpg', None), 'one_does_not_simply': ('1bij.jpg', None),
    'gosling_stare': ('612dfk.png', None), 'dwight_false': ('1bhu.jpg', None),
    'mr_bean_waiting': ('21ajtl.jpg', None),
}
# Detected, but the face is too occluded / the "face" is a false positive -> unusable landmarks
REJECT = {'drake_disapprove', 'sad_keanu'}

here = Path(__file__).parent
raw, out = here / 'memes_raw', here / 'memes'
raw.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(here / 'face_landmarker.task')), num_faces=1))

rows = []
for _, r in pd.read_csv(here / 'memes_rough.csv').iterrows():
    fn, crop = SOURCES[r.id]
    url = f'https://i.imgflip.com/{fn}'
    if not (raw / fn).exists():
        subprocess.run(['curl', '-s', '-A', 'Mozilla/5.0', '-o', str(raw / fn), url], check=True)
    img = cv2.imread(str(raw / fn))
    if crop: img = img[crop[1]:crop[3], crop[0]:crop[2]]
    found = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(img[..., ::-1]))).face_landmarks
    if not found or r.id in REJECT:
        print('dropped:', r.id); continue
    cv2.imwrite(str(out / f'{r.id}.jpg'), img)
    rows.append(dict(id=r.id, meme=r.meme, expression=r.expression, file=f'memes/{r.id}.jpg', source=url))
pd.DataFrame(rows).to_csv(here / 'memes.csv', index=False)
print(f'kept {len(rows)} memes -> memes.csv')
