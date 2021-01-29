import cv2
import numpy as np
import time

# yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
# anchors = read_anchors("model_data/yolo_anchors.txt")
# ancho boxes is already in yolovs.cfg

# Get frames of video/webcam
# use 1 if using external webcam
cap = cv2.VideoCapture(0)

# you can adjust confidence threshold and nms threshold
confThreshold = 0.5
nmsThreshold = 0


# collect names of coco.names classes and storing it in classNames
classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


# random colors
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# yolov3.cfg is the architecture of the yolo network
# yolov3.weights are the weights for the yolo network
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

# you can also use yolov3-tiny.cfg and yolov3-tiny.weights
#modelConfiguration = "yolov3-tiny.cfg"
#modelWeights = "yolov3-tiny.weights"

# create the network
net = cv2.dnn.readNet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



# go through each of the bounding boxes and see if the probablities are good enough. If so, store it in these lists
def findObjects(outputs, img):
    # heigh, width, and channel
    # channels/depth is for RGB values
    hT, wT, cT = img.shape
    # bounding box: will contain centerx, centery, width and height
    bbox = []
    # contain all the classids
    classIds = []
    # confidence values
    confs = []
    # go through each 3 outputs
    for output in outputs:
        # output is an (1, 85) array
        for detection in output:
            # remove the first 5 elements: centerx, centery, width, height, and confidence
            scores = detection[5:]
            # np.argmax returns indices of the max element of the array in a particular axis.
            # find index of the maximum confidence value
            classId = np.argmax(scores)
            # get the value
            confidence = scores[classId]

            # 0.5 is confidence threshold
            if confidence > confThreshold:
                # save width height and centerx, centery
                # get the pixel value
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                # get the centerx and centery
                x, y = int((detection[0] * wT) - w/2), int((detection[1] * hT) - h/2)

                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # non maximum suppression eliminates overlapping bounding boxes. It finds the overlapping boxes,
    # it picks the maximum confidence box, and suppress the non-maximum boxes
    # outputs which of the bounding boxes to keep by giving their indices
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    b = []
    # loop over the indices and draw the bounding boxes
    #for i in indices:
    for i in range(len(bbox)):
        if i in indices:
            #i = i[0]
            #box = bbox[i]
            x, y, w, h = bbox[i] #box[0], box[1], box[2], box[3]
            color = [int(c) for c in colors[classIds[i]]]



            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



            # printing the labels and confidences every iteration on the terminal
            label = classNames[classIds[i]]
            print(label, end = " ")
            print(f"{int(confs[i] * 100)}%")

            # storing every object it sees in the list b
            b.append(label)
            print(b)


            # get the fps
            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)



            for i in b:
                cv2.putText(img, f'{label.upper()} {b.count(i)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # b.count(i) is how many occurences of that object





starting_time = time.time()
frame_id = 0

while True:
    success, img = cap.read()
    # frame_id is used to calculate fps
    frame_id += 1

    # the network only accepts one image format. This format is "blob"
    blob = cv2.dnn.blobFromImage(img, 1/255,(256, 256), [0, 0, 0], 1, crop=False)
    # set blob as an input to the network
    net.setInput(blob)

    # yolo have 3 different outputs/3 different values coming out of the outwork
    # get layer names of the network
    layerNames = net.getLayerNames()
    # extract only the output layers
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    # send this image as a forward pass to the network and find the output of the last three output layers
    outputs = net.forward(outputNames)


    """outputs[0].shape is a matrix of (300, 85) 300 is the number of bounding boxes. In order to store the information
       of a bounding box, you need to know the centerx, centery, and the width and height. In 85, the first 4 values are
       the centerx, centery, and the width and height. THe 5th value is the confidence that there is an object present
       within this bounding box. The next 80 values are probablities of each of the class. 
       outputs[1].shape is a matrix of (1200, 85)
       outputs[2].shape is a matrix of (4800, 85)"""



    # initialize the object detection function
    findObjects(outputs, img)




    # show the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)






