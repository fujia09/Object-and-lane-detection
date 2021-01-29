import cv2
import numpy as np

# object detection function for videos
def objectDetector(img):

    # reading the weights
    # neural network using pretrained weights
    yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # read the coco names
    classes = []
    with open("coco.names", "r") as file:
        # saving each object in coco.names in the list
        classes = [line.strip() for line in file.readlines()]

    #extract from layer names and output layers
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

    # Color red for font and color green to highlight where the objects are
    colorRed = (0, 0, 255)
    colorGreen = (0, 255, 0)


    # extracting height, width of the image
    height, width, channels = img.shape

    # Detecting objects
    # blob removes the image and takes the average of the image. (mitigates light inconsistency)
    # bringing our images which are in different light intensities to the original network parameters
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # passing the image blob into the yolo network
    # so that the yolo network can use the blob image to detect
    yolo.setInput(blob)
    # reading the yolo network and saving it in "output"
    outputs = yolo.forward(output_layers)

    # identifying different object classes
    class_ids = []
    # confidence score
    confidences = []
    # defining all detected objects
    boxes = []

    # detecting objects
    # reading each output
    for output in outputs:
        for detection in output:
            # confidence socring
            scores = detection [5:]
            # np.argmax returns indices of the max element of the array in a particular axis.
            # for all the detection that was performed np.argmax takes the maximum score
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # only using confidence that is more than 50 %
            if confidence > 0.5:
                # defining where the objects are
                # finds the center point of the object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # in opencv we need to take the center of the object to the top left corner of the box
                x = int(center_x - w /2)
                y = int(center_y - h /2)

                # once we define the boxes of the objects we save it in an array called "boxes"
                # x, y, w, h are the coordinate variables of the boxes
                boxes.append([x, y, w, h])
                # store confidence value
                confidences.append(float(confidence))
                # store the class objects
                class_ids.append(class_id)

    # Performs non maximum suppression given boxes and corresponding scores.
    # erases some unncessary detections (noise/error(items that don't want to be detected)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # use opencv to draw the boxes after NMS removes the error
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # the label is extracting the text for the class from coco.names
            label = str(classes[class_ids[i]])
            # draw the rectangles
            cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
            # print the font, thickness and size of the words
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, colorRed, 2)
            #print(confidences[i])
            cv2.putText(img, label + " " + str(confidences[i]), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, colorRed, 2)

    #cv2.imshow("Image", img)
    # cv2.imwrite("output.jpg", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img





# using capture to read the frames of the video
cap = cv2.VideoCapture("video.mp4")
#cap = cv2.VideoCapture(0)

# read the height, width of the image
# cap.read() reads the frame
ret, frame = cap.read()
height, width, layers = frame.shape

# VideoWriter helps create a video file
# XVID is used for avi video files
fourcc = cv2.VideoWriter_fourcc(*'MP4')
# show you 20fps
video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))


# slowly read one by one frame
# and take that image and ass it to the yolo object detector
# and then pass that image back

# cap.isOpened() as long is video file is open
while(cap.isOpened()):
    # Capture frame by frame/image by image
    ret, frame = cap.read()
    # objectDetector passes the frame from the video file to the objectDetector function
    # output is the image with all the bounded rectangles
    output_img = objectDetector(frame)
    # save that image into the avi file
    video.write(output_img)


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
video.release()




