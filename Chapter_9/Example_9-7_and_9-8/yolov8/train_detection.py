
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型，从YAML构建并转移权重
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    # 训练模型
    model.train(data='ultralytics/cfg/datasets/crack-seg.yaml', imgsz=640, epochs=100, batch=16, workers=1, device='0', project='runs/train', name='Yuancrackstrain')