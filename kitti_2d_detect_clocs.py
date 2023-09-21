import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from ultralytics import YOLO
import os
from glob import glob
from tqdm import tqdm
dataset_name = "kitti"
image_paths = sorted(glob(f"/mnt/lxc/{dataset_name}/object/training/image_2/*.png"))
os.makedirs(f'./clocs/{dataset_name}/data',exist_ok=True)
model = YOLO('yolov8x.pt')
for image_path in tqdm(image_paths):
    file_temp = open(f'./clocs/{dataset_name}/data/'+os.path.basename(image_path).split('.')[0]+'.txt', 'w')
    results = model(image_path)
    boxes = results[0].boxes.boxes.cpu().numpy()
    for t in range(boxes.shape[0]):
        if boxes[t, 5] == 2:
            file_temp.write('Car -1 -1 -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.4f\n' % (boxes[t, 0],boxes[t, 1],boxes[t, 2],boxes[t, 3],boxes[t,4]))
    file_temp.close()
