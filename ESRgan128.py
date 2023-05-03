import cv2
import numpy as np
import cv2
import numpy as np
import pandas as pd
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def load_yolo():
    net = cv2.dnn.readNet("weights/yolov7.weights", "cfg/yolov7.cfg")
    # Set target backend and target device
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open("data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()] 
    
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.85:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
model_path = 'weights\RealESRGAN_x4plus.pth'
# restorer
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    model=model,
)

model, classes, colors, output_layers = load_yolo()

actual_folder = "coco128/images/actual"
output_folder = "coco128/images/predicted"
output_text_folder = "coco128/labels/predicted"
for file_name in os.listdir(actual_folder):
    filename = file_name.strip('.jpg')
    actual_file = os.path.join(actual_folder, file_name)
    img = cv2.imread(actual_file)
    output, _ = upsampler.enhance(img, outscale=1)
    output_file = os.path.join(output_folder,file_name)
    cv2.imwrite(output_file, output)
    height, width, channels = output.shape
    blob, outputs = detect_objects(output, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    print(boxes)
    print(class_ids)
    file_path = os.path.join(output_text_folder,filename+".txt")
    f = open(file_path,"w")
    for i in range(len(boxes)):
        print(class_ids[i], ' '.join(map(str, boxes[i])))
        res = str(class_ids[i]) + " " + ' '.join(map(str, boxes[i])) + '\n'
        f.write(res)
