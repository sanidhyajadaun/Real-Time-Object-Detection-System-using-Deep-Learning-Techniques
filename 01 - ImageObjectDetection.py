#importing required libraries
import cv2
import numpy as np

#defining the class ImageObjectDetection 
class ImageObjectDetection:
    
    #initializing the image path in the constructor
    def __init__(self,image_path):
        self.img_path = image_path
    
    #defining the load_yolo function
    def load_yolo(self):
        #reading the pre-trained model using the weight and cfg file 
        net = cv2.dnn.readNet("weights/yolov7.weights", "cfg/yolov7.cfg")
        #Setting target backend and target device
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #storing the class names in the list classes[]
        classes = []
        with open("data/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()] 
        
        #getting the names of the output layer
        output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
        #generating the colours for each class names
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers

    #defining the load_image()
    def load_image(self):
        #loading the image using imread() function
        img = cv2.imread(self.img_path)
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        return img, height, width, channels

    #defining the detect_objects() function
    def detect_objects(self,img, net, outputLayers):
        #performing pre-processing on the image like scaling,resizing and swapping the channels			
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        #setting the preprocessed image as input
        net.setInput(blob)
        #performing feed forward
        outputs = net.forward(outputLayers)
        return blob, outputs

    #defining get_box_dimensions() function
    def get_box_dimensions(self,outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        #iterating through each of the output in output layers
        for output in outputs:
            #iterating through each of the row in output matrix
            for detect in output:
                scores = detect[5:]
                #finding index of maximum value in scores matrix
                class_id = np.argmax(scores)
                #conf stores the value at the index
                conf = scores[class_id]
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
        # resized = cv2.resize(img,(500,500), interpolation = cv2.INTER_AREA)
        cv2.imwrite("predictions.jpg",img)

    #defining image detect function
    def image_detect(self): 
        #calling the load_yolo() function
        model, classes, colors, output_layers = self.load_yolo()
        image, height, width, channels = self.load_image()
        blob, outputs = self.detect_objects(image, model, output_layers)
        boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
        self.draw_labels(boxes, confs, colors, class_ids, classes, image)
        while True:
            key = cv2.waitKey(1)
            if key == 27:
                break

#defining the path of the image
image_path = "image/city.jpg"
print("Opening "+image_path+" .... ")

#creating an instance of image detection class
ref = ImageObjectDetection(image_path)

#calling the image_detect() function
ref.image_detect()

#to close the windows created by OpenCV
cv2.destroyAllWindows()