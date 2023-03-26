import cv2
import numpy as np

class RealTimeObjectDetection:
    def __init__(self,capture):
        self.capture = capture
        
    #function to load data
    def load_data(self):
        # readNet returns the pretrained model using weights and cfg
        net = cv2.dnn.readNet("weights/yolov7.weights", "cfg/yolov7.cfg")
        # Set target backend and target device if cude is available
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # list to store the names of labels in dataset
        classes = []
        with open("data/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()] 
        
        # getting the names of the output layers
        output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
        # generating a random color combination for each label
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers

    # function to get the capture object
    def start_webcam(self):
        cap = cv2.VideoCapture(self.capture)
        return cap

    # function to preprocess the image and getting the output from the model
    def detect_objects(self,img, net, outputLayers):			
        # getting the preprocessed image (binary large object)
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        # setting the input
        net.setInput(blob)
        # performing the feedforward in the model
        outputs = net.forward(outputLayers)
        return blob, outputs

    # function to create contour
    def get_box_dimensions(self,outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        # iterating to each output 
        for output in outputs:
            # iterating to each of the rows in output matrix
            for detect in output:
                scores = detect[5:]
                # getting the index of the maximum score
                class_id = np.argmax(scores)
                # getting the score value and storing it to the confidence
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
                
    def draw_labels(self,boxes, confs, colors, class_ids, classes, img): 
        # applying non max suppression 
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

    # function to perform real time object detection
    def webcam_detect(self):
        model, classes, colors, output_layers = self.load_data()
        cap = self.start_webcam()
        while True:
            # read() returns a tuple, first a boolean value which denotes whether the frame is read successfully or not and then returns the whole frame
            _, frame = cap.read()
            # getting the height and weight from the frame
            height, width, channels = frame.shape
            # getting the outputs 
            blob, outputs = self.detect_objects(frame, model, output_layers)

            boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
            # function called to plot the contours along with the labels
            self.draw_labels(boxes, confs, colors, class_ids, classes, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()

print('---- Real Time object detection ----')
ref = RealTimeObjectDetection(0)
ref.webcam_detect() 
cv2.destroyAllWindows()