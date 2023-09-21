import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from ultralytics import YOLO
import os
from glob import glob

def write_kitti_format(results, image_path,file_name):
    """
    将YOLOv8的检测结果保存为KITTI格式的TXT文件。
    
    参数:
    det_results : list
        YOLOv8的输出结果results[0]

    save_path : str
        保存TXT文件的路径。
    """
    # COCO 0: 'person', 1: 'bicycle', 2: 'car', 5: 'bus', 7: 'truck'
    # KITTI 1: 'Pedestrian', 2: 'car', 3: 'Cyclist'
    for cls,conf,xyxy in zip(results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xyxy):
        # KITTI格式：cls conf x1 y1 x2 y2
        cls = int(cls)
        if (cls in [0,1,2]):
            if cls == 0:
                cls = 1
            elif cls == 1:
                cls = 3
            xyxy = xyxy.cpu().numpy()
            xyxy = [str(x) for x in xyxy]
            xyxy = ' '.join(xyxy)
            line = f'{image_path} {cls} {conf} {xyxy}\n'
            file_name.write(line)
        else:
            continue
    


image_paths = sorted(glob("/mnt/lxc/kitti/object/training/image_2/*.png"))
os.makedirs('./rgb_detections',exist_ok=True)
# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')
file_train = open('./rgb_detections/rgb_detection_train.txt', 'w')
for image_path in image_paths:
    # 如果图片名字在这个train.txt里面，就执行
    if os.path.basename(image_path).split('.')[0]+'\n' in open('./ImageSets/train.txt').readlines(): 
        # Run inference on the source
        results = model(image_path)
        # 保存为KITTI格式的TXT文件
        write_kitti_format(results, image_path,file_train)
        print(f'Image {image_path} done.')
file_train.close()

file_val = open('./rgb_detections/rgb_detection_val.txt', 'w')
for image_path in image_paths:
    # 如果图片名字在这个val.txt里面，就执行
    if os.path.basename(image_path).split('.')[0]+'\n' in open('./ImageSets/val.txt').readlines(): 
        # Run inference on the source
        results = model(image_path)
        # 保存为KITTI格式的TXT文件
        write_kitti_format(results, image_path,file_val)
        print(f'Image {image_path} done.')
file_val.close()