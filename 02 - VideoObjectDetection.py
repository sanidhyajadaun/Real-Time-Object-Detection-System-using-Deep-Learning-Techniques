import cv2
import numpy as np

class VideoObjectDetection:
    def __init__(self,path):
        self.temp = 0
        self.path = path

    #function to load data
    def load_data(self):
        #pretrained model is returned using readNet
        net = cv2.dnn.readNet("weights/yolov7.weights", "cfg/yolov7.cfg")
        # Set target backend and target device
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #list to store all classes in the dataset
        classes = []
        with open("data/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()] 
        #getting names of output layers
        output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
        #generating different colors for unique labels 
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers
 
    #function to perform preprocessing and feed forward in the net
    def detect_objects(self,img, net, outputLayers):	
        #blobfromImage performs preprocessing steps like scaling, resizing, swapping of channels on image		
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        #setting our input to the net
        net.setInput(blob)
        #feed forwarding in the net
        outputs = net.forward(outputLayers)
        return blob, outputs

    #getting dimensions of the bounding box
    def get_box_dimensions(self,outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        #iterating through each output in the list
        for output in outputs:
            #iterating through each row in output matrix
            for detect in output:
                scores = detect[5:]
                #finding index of maximum value in scores matrix
                class_id = np.argmax(scores)
                #conf stores the value at the index
                conf = scores[class_id]
                #if confidence is greater than 0.3 then, object detection is carried through
                if conf > 0.3:
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

    #function to find the label of the detected object          
    def draw_labels(self,boxes, confs, colors, class_ids, classes, img):
        #applying non-max supression - removing overlapping bounding boxes 
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
        cv2.imshow("Image", img)

    #function to start video in which objects have to be detected
    def start_video(self):
        model, classes, colors, output_layers = self.load_data()
        cap = cv2.VideoCapture(self.path)
        while True:
            #reading a tuple which returns a boolean value of whether a frame is present or not and frame is the matrix value of the frame 
            _, frame = cap.read()
            height, width, channels = frame.shape
            blob, outputs = self.detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
            self.draw_labels(boxes, confs, colors, class_ids, classes, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()

#path of the video for object detection
video_path = "video/cycling.mp4"
print('Opening '+video_path+" .... ")
ref = VideoObjectDetection(video_path)
ref.start_video()
cv2.destroyAllWindows()