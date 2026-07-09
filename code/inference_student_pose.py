#!/usr/bin/env python3
"""
简单推理脚本：对指定文件夹下图片/视频做推理并把带预测框的文件 & 结果 CSV 存到目标文件夹。
支持：图片 + 视频
使用示例：
python inference_student_pose.py
"""
import argparse
from pathlib import Path
import csv
import os
from ultralytics import YOLO

# 获取当前脚本所在目录作为项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# simple model cache to avoid reloading for repeated calls
_MODEL_CACHE = {}


def run_inference(source, output_dir, model_path=None, imgsz=640, conf=0.25, iou=0.45, device='0', save=True):
    """Run inference on a source (file path, numpy array, PIL image, etc.) and save annotated outputs + CSV to output_dir.

    Returns the raw results list from YOLO.predict.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        raise ValueError("model_path must be provided")

    model = _MODEL_CACHE.get(model_path)
    if model is None:
        model = YOLO(model_path)
        _MODEL_CACHE[model_path] = model

    results = model.predict(
        source=source,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        save=save,
        project=str(out_dir),
        name="",
        exist_ok=True,
        verbose=False,
    )

    # write/update CSV with detections (append if file exists)
    csv_file = out_dir / 'predictions.csv'
    write_header = not csv_file.exists()
    with csv_file.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['file', 'class_id', 'class_name', 'conf', 'x1', 'y1', 'x2', 'y2', 'frame'])

        # When source is a file path, use its name; otherwise use str(source)
        source_name = os.path.basename(str(source)) if isinstance(source, (str, Path)) else str(source)

        class_names = getattr(model, 'names', {})
        for frame_idx, r in enumerate(results):
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clsids = boxes.cls.cpu().numpy()
            except Exception:
                continue

            for bb, cf, cid in zip(xyxy, confs, clsids):
                x1, y1, x2, y2 = map(float, bb)
                cid = int(cid)
                # model.names 是 dict[int, str]，用 int(cid) 查找
                if isinstance(class_names, (list, tuple)):
                    cls_name = class_names[cid] if cid < len(class_names) else str(cid)
                elif isinstance(class_names, dict):
                    cls_name = class_names.get(cid, str(cid))
                else:
                    cls_name = str(cid)
                writer.writerow([source_name, cid, cls_name, float(cf), x1, y1, x2, y2, frame_idx])

    return results


def is_media(p: Path) -> bool:
    # 支持 图片 + 视频
    ext = p.suffix.lower()
    return ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                   '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')


def main():
    p = argparse.ArgumentParser(description='Simple batch inference for student posture detection')
    p.add_argument('--input', '-i', default=os.path.join(BASE_DIR, 'TRY', 'origin'),
                   help='Input folder with images/videos')
    p.add_argument('--output', '-o', default=os.path.join(BASE_DIR, 'TRY', 'target'),
                   help='Output folder for annotated files and CSV')
    p.add_argument('--model', '-m', default=os.path.join(BASE_DIR, 'runs', 'detect', 'train-3', 'weights', 'best.pt'),
                   help='Model path')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    p.add_argument('--device', default='0', help='Device, e.g. cpu or 0 or cuda:0')
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 检查输入目录是否存在
    if not in_dir.is_dir():
        print(f'输入目录不存在：{in_dir}')
        return

    # 检查模型文件是否存在
    if not os.path.isfile(args.model):
        print(f'模型文件不存在：{args.model}')
        return

    model = YOLO(args.model)

    csv_file = out_dir / 'predictions.csv'
    with csv_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'class_id', 'class_name', 'conf', 'x1', 'y1', 'x2', 'y2', 'frame'])

        files = sorted([fp for fp in in_dir.iterdir() if fp.is_file() and is_media(fp)])
        if not files:
            print('输入文件夹没有找到图片/视频：', in_dir)
            return

        print(f"找到 {len(files)} 个文件，开始推理...")

        for file_path in files:
            try:
                results = model.predict(
                    source=str(file_path),
                    imgsz=args.imgsz,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    save=True,
                    project=str(out_dir),
                    name="",
                    exist_ok=True
                )
            except Exception as e:
                print(f'推理失败，跳过 {file_path.name}: {e}')
                continue

            # 遍历每一帧结果
            for frame_idx, r in enumerate(results):
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue

                try:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clsids = r.boxes.cls.cpu().numpy()
                except Exception:
                    continue

                class_names = model.names
                for bb, cf, cid in zip(xyxy, confs, clsids):
                    x1, y1, x2, y2 = map(float, bb)
                    cid = int(cid)
                    # model.names 是 dict[int, str]，用 int(cid) 查找
                    if isinstance(class_names, (list, tuple)):
                        cls_name = class_names[cid] if cid < len(class_names) else str(cid)
                    elif isinstance(class_names, dict):
                        cls_name = class_names.get(cid, str(cid))
                    else:
                        cls_name = str(cid)
                    writer.writerow([
                        file_path.name, cid, cls_name, float(cf), x1, y1, x2, y2, frame_idx
                    ])

    print('=' * 50)
    print('推理完成！')
    print('带框文件保存到：', out_dir)
    print('结果表格保存到：', csv_file)
    print('=' * 50)


if __name__ == '__main__':
    main()
