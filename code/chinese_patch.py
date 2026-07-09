
# ============================================================================
# Chinese Support Patch - PIL-based Chinese text rendering
# ============================================================================
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Chinese font path
_CHINESE_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_chinese_font = None

# Class name translation (English -> Chinese)
CLASS_NAME_CN = {
    'hand-raising': '举手',
    'reading': '阅读',
    'writing': '写字',
    'using phone': '用手机',
    'bowing the head': '低头',
    'leaning over the table': '趴桌子',
}

def _get_chinese_font(size=20):
    global _chinese_font
    if _chinese_font is None:
        try:
            _chinese_font = ImageFont.truetype(_CHINESE_FONT_PATH, size)
        except Exception:
            _chinese_font = ImageFont.load_default()
    return _chinese_font

def plot_with_chinese(result, font_size=22):
    img = result.orig_img
    if img is None:
        return result.plot()
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_chinese_font(font_size)
    
    colors = [
        (255, 87, 34),    # Deep Orange
        (76, 175, 80),    # Green
        (33, 150, 243),   # Blue
        (255, 193, 7),    # Amber
        (156, 39, 176),   # Purple
        (0, 188, 212),    # Cyan
    ]
    
    class_names = getattr(result, 'names', {})
    
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if isinstance(class_names, dict):
                cls_name_en = class_names.get(cls_id, str(cls_id))
            elif isinstance(class_names, list):
                cls_name_en = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            else:
                cls_name_en = str(cls_id)
            
            # Translate to Chinese
            cls_name = CLASS_NAME_CN.get(cls_name_en, cls_name_en)
            
            color = colors[cls_id % len(colors)]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = "%s %.0f%%" % (cls_name, conf * 100)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            label_y = max(0, y1 - text_h - 8)
            draw.rectangle([x1, label_y, x1 + text_w + 8, y1], fill=color)
            draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

print("[OK] Chinese patch loaded with CN class names")
