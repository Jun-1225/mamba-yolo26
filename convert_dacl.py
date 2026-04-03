import json
import os
from pathlib import Path
from tqdm import tqdm

# 根据你提供的 meta.json 自动整理的 19 个类别映射
CLASS_MAP = {
    "alligator crack": 0, "bearing": 1, "cavity": 2, "crack": 3, "drainage": 4,
    "efflorescence": 5, "expansion joint": 6, "exposed rebars": 7, "graffiti": 8,
    "hollowareas": 9, "joint tape": 10, "protective equipment": 11, "restformwork": 12,
    "rockpocket": 13, "rust": 14, "spalling": 15, "washouts/concrete corrosion": 16,
    "weathering": 17, "wetspot": 18
}

def convert_dataset(base_dir, output_base):
    """
    处理结构: base_dir/train(val)/ann/*.json
    输出结构: output_base/labels/train(val)/*.txt
    """
    for split in ['train', 'val']:
        # 路径定位
        ann_dir = os.path.join(base_dir, split, 'ann')
        img_dir = os.path.join(base_dir, split, 'img')
        label_out_dir = os.path.join(output_base, 'labels', split)
        image_out_dir = os.path.join(output_base, 'images', split)
        
        os.makedirs(label_out_dir, exist_ok=True)
        os.makedirs(image_out_dir, exist_ok=True)

        if not os.path.exists(ann_dir):
            print(f"跳过 {split}: 文件夹不存在")
            continue

        json_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        
        for json_file in tqdm(json_files, desc=f"正在转换 {split} 标注"):
            with open(os.path.join(ann_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_w = data['size']['width']
            img_h = data['size']['height']
            
            yolo_lines = []
            for obj in data['objects']:
                # 仅处理赛题要求的多边形类型 [cite: 31]
                if obj.get('geometryType') != 'polygon':
                    continue
                
                class_title = obj.get('classTitle')
                if class_title not in CLASS_MAP:
                    continue
                
                class_id = CLASS_MAP[class_title]
                
                # 提取外部多边形点集
                points = obj['points']['exterior']
                if len(points) < 3: 
                    continue
                
                # 归一化坐标
                normalized_points = []
                for pt in points:
                    px = pt[0] / img_w
                    py = pt[1] / img_h
                    normalized_points.append(f"{px:.6f}")
                    normalized_points.append(f"{py:.6f}")
                
                # 写入格式: <class_id> <x1> <y1> <x2> <y2> ...
                line = f"{class_id} " + " ".join(normalized_points)
                yolo_lines.append(line)
            
            # 生成与图像同名的 TXT 文件
            # 兼容 image.jpg.json 或 image.json 命名方式
            base_name = json_file.replace('.json', '').replace('.jpg', '').replace('.png', '')
            txt_path = os.path.join(label_out_dir, f"{base_name}.txt")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))
                
    print(f"\n转换完成！请将原 images/{split} 下的图片手动复制或软链接到 {output_base}/images/{split}")

# 执行转换 (请确保 SOURCE 路径指向包含 train 和 val 的文件夹)
SOURCE_DATASET = '/root/mamba-yolo26' 
OUTPUT_DATASET = '/root/autodl-tmp/dacl10k' 

convert_dataset(SOURCE_DATASET, OUTPUT_DATASET)