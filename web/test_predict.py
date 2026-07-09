import requests, base64
from io import BytesIO
from PIL import Image

img = Image.new('RGB', (160, 120), (255, 255, 255))
buf = BytesIO()
img.save(buf, format='PNG')
data = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

try:
    r = requests.post('http://127.0.0.1:7860/predict-frame', json={'image': data}, timeout=60)
    print('status', r.status_code)
    print('content-type', r.headers.get('content-type'))
    txt = r.text
    print(txt[:2000])
except Exception as e:
    print('request error', e)
