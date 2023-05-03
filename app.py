from flask import Flask, render_template, Response, request
import cv2
import base64
import numpy as np

app = Flask(__name__)
# camera = cv2.VideoCapture(0)
model = ""

#function to load data
def load_data():
    weights = "weights/"+model.lower()+".weights"
    cfg = "cfg/"+model.lower()+".cfg"
    # readNet returns the pretrained model using weights and cfg
    # net = cv2.dnn.readNet("weights/yolov7.weights", "cfg/yolov7.cfg")
    net = cv2.dnn.readNet(weights,cfg)

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

# function to preprocess the image and getting the output from the model
def detect_objects(img, net, outputLayers):			
    # getting the preprocessed image (binary large object)
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    # setting the input
    net.setInput(blob)
    # performing the feedforward in the model
    outputs = net.forward(outputLayers)
    return blob, outputs

# function to create contour
def get_box_dimensions(outputs, height, width):
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

def gen_frames():
    camera = cv2.VideoCapture(0)
    model, classes, colors, output_layers = load_data()
    while True:
        success, frame = camera.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width, channels = frame.shape
        # getting the outputs 
        blob, outputs = detect_objects(frame, model, output_layers)

        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

        # applying non max suppression 
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 5), font, 1, color, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam_feed',methods=['post'])
def webcam():
    global model
    model = str(request.form['browser'])
    print(model.center(70,'-'))
    return render_template('webcam.html')

@app.route('/video_feed',methods=['GET', 'POST'])
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == '__main__':
    app.run(debug=True)
