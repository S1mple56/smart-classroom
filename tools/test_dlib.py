"""端到端测试：构建特征库并测试匹配"""
import logging, json, os
logging.basicConfig(level=logging.INFO)
import dlib_face_recognizer as fr

upload_dir = os.path.join('data', 'upload')
students_info = os.path.join('data', 'students_info.json')
features_cache = os.path.join('data', 'face_features.json')

# 检查 upload 目录
print("=== 学生照片检查 ===")
if os.path.isdir(upload_dir):
    for p in os.listdir(upload_dir):
        d = os.path.join(upload_dir, p)
        if os.path.isdir(d):
            photos = [f for f in os.listdir(d) if f.endswith('.jpg')]
            print(f"  {p}: {len(photos)} photos")
else:
    print("  upload 目录不存在，跳过")

# 构建特征库
print("\n=== 构建特征数据库 ===")
db = fr.build_features_database(upload_dir, students_info, features_cache)
print(f"完成! 共 {len(db)} 人")

# 验证生成的 JSON
with open(features_cache, 'r', encoding='utf-8') as f:
    cached = json.load(f)
for pid, data in cached.items():
    nonzero = sum(1 for v in data["features"] if v != 0.0)
    print(f"  {pid}: {data['name']} ({data['student_id']}), "
          f"特征维度={len(data['features'])}, 非零值={nonzero}")

# 测试匹配（如果有学生照片）
if os.path.isdir(upload_dir):
    for p in os.listdir(upload_dir):
        d = os.path.join(upload_dir, p)
        if not os.path.isdir(d):
            continue
        photos = [f for f in os.listdir(d) if f.endswith('.jpg')]
        if photos:
            test_img = os.path.join(d, photos[0])
            print(f"\n=== 测试匹配: {test_img} ===")
            result = fr.match_face_dlib(test_img, features_cache, upload_dir, students_info)
            for k, v in result.items():
                print(f"  {k}: {v}")
            break
