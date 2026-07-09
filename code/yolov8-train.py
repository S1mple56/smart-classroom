import os
from ultralytics import YOLO

# 获取当前脚本所在目录作为项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 使用 train-3 的 last.pt 恢复训练（会自动加载优化器/调度器状态）
model = YOLO(os.path.join(BASE_DIR, "runs", "detect", "train-3", "weights", "last.pt"))

# 恢复训练
model.train(
    data=os.path.join(BASE_DIR, "yolo-bvn.yaml"),
    workers=0,
    epochs=100,
    batch=16,
    device=0,
    resume=True,
)
