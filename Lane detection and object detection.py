import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math



# Get frames of video/webcam
# use 1 if using external webcam
file = "solidYellowLeft.mp4"
cap = cv2.VideoCapture(file)

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




def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # make a copy of the image before converting rgb image to gray
    lane_image = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    # blur the image before passing it into the algorithm. In order to reduce the noise level in the image
    # 5x5 kernelin order to blur
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    # canny function performs a derivative on both the x (width) and y (height) function.
    # - measuring the change in intensity with respect to adjacent pixels - large derivative = large change
    # by computing the derivative/gradient - the gradient is defined as the change in brightness over a series of pixels
    canny = cv2.Canny(blur, 50, 150)
    """cv2.Canny(img, low_threshold, high_threshold) - if it is below the low_threshold it is rejected
    - if it is above the high_threshold it is accepted and if it is between low and high_threshold it is only 
    accepted if it is connected to a strong edge.
    """
    return canny


def region_of_interest(img):
    # obtain height of image
    height = img.shape[0]
    # obtain width of image
    width = img.shape[1]
    # triangle removes everything except the points of interest
    # change triangle coordinates based on where the lanes are
    triangle = np.array([[
        (0, height),
        (width/1.8, height/2.5),
        (width, height), ]], np.int32)
    # apply the triangle onto a black mask - every pixel will have 0 intensity
    mask = np.zeros_like(img)
    # cv2.fillyPoly fills the triangle onto the black mask
    cv2.fillPoly(mask, triangle, 255)
    """return the image only where the mask pixel matches
    Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar.  
    - cancels out unless it is a pair of 1s"""
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# houghLines only works on straight lines, not curved roads
# bin with maximum number of votes in houghspace/intersection of the sinousidal graphs of rho and theta in houghspace
def houghLines(cropped_canny):
    """threshold is 100, which is the minimum number of intersections in a bin is 100
    minlinelength is 40 of any lines detected that are less than 40 pixels are rejected
    maxlinegap is the maximum distance in pixels between different lines which we will
    allow to be connected into a single line instead of them being broken up"""

    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100,
                           np.array([]), minLineLength=40, maxLineGap=100)


# adding the mask line image on the original frame
def addWeighted(frame, line_image):
    # taking the sum of the color image with the masked line image/adding the pixel intensities of masked image to the original frame
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)





def display_lines(img, lines):
    # create lists of previous in order to treat outliers so that we can access previous reliable coordinates
    left_prev_x1 = []
    left_prev_x2 = []
    left_prev_y1 = []
    left_prev_y2 = []
    right_prev_x1 = []
    right_prev_x2 = []
    right_prev_y1 = []
    right_prev_y2 = []
    # use count in order to make sure lines do not except outliers/jump around
    count = 1
    # obtain height of image
    height = img.shape[0]
    # obtain width of image
    width = img.shape[1]

    # declare a mask of zeros
    line_image = np.zeros_like(img)
    # check if it even detected any lines
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            if count == 1:
                count += 1
                if x1 < int(width//5):
                    x1 = int(width//5)
                if x2 < int(width//4):
                    x2 = int(width//4)
                if x1 > width:
                    x1 = width
                if x2 > int(width//1.8):
                    x2 = int(width//1.8)
                if x2 <= x1:
                    x2 = width//2
                left_prev_x1.append(x1)
                left_prev_x2.append(x2)
                left_prev_y1.append(y1)
                left_prev_y2.append(y2)

            elif count == 2:
                if x1 < int(width//4):
                    x1 = int(width//4)
                if x2 < int(width//4):
                    x2 = int(width//4)
                if x1 > width:
                    x1 = width
                if x1 < int(width//1.7):
                    x1 = width
                #if x2 > int(width//1.8):
                    #x2 = int(width//1.8)
                right_prev_x1.append(x1)
                right_prev_x2.append(x2)
                right_prev_y1.append(y1)
                right_prev_y2.append(y2)

            elif count % 2 == 1:
                count += 1
                if x1 < int(width//4):
                    x1 = left_prev_x1[-1]
                if x2 < int(width//4):
                    x2 = left_prev_x2[-1]
                if x1 > width:
                    x1 = left_prev_x1[-1]
                if x2 > int(width//1.8):
                    x2 = left_prev_x2[-1]
                if x2 <= x1:
                    x2 = left_prev_x2[-1]
                left_prev_x1.append(x1)
                left_prev_x2.append(x2)
                left_prev_y1.append(y1)
                left_prev_y2.append(y2)

            elif count % 2 == 0:
                count += 1
                if x1 < int(width//4):
                    x1 = right_prev_x1[-1]
                if x2 < int(width//4):
                    x2 = right_prev_x2[-1]
                if x1 > width:
                    x1 = right_prev_x1[-1]
                if x1 < int(width//1.7):
                    x1 = right_prev_x1[-1]
                if x2 > int(width//1.8):
                    x2 = right_prev_x2[-1]
                right_prev_x1.append(x1)
                right_prev_x2.append(x2)
                right_prev_y1.append(y1)
                right_prev_y2.append(y2)


            print(x1, y1, x2, y2)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    # line image is drawn on the mask
    return line_image




#previous_points_for_NaN = []
# get the coordinates of y1, y2, x1, x2
def make_points(image, line_parameters):
    if type(line_parameters) == np.ndarray:
        slope, intercept = line_parameters

    elif math.isnan(line_parameters) == True:
        slope = -0.17454980594255887
        intercept = 553.0363330336027

    y1 = int(image.shape[0])
    y2 = int(y1 * 3.0 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])




# average the lines from the left and right lanes in order to make only ONE line so it looks smoother
def average_slope_intercept(image, lines):
    # coordinates of lines on the left
    left_fit = []
    # coordinates of lines on the right
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # fit a polynomial (y=mx+b) onto the x and y points and return a vector of coefficients that describe the slop and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # acquire the slope and intercepts
        slope = parameters[0]
        intercept = parameters[1]
        # negative slope represents left lane line
        if slope < 0:
            left_fit.append((slope, intercept))
        # positive slope represents right lane line
        else:
            right_fit.append((slope, intercept))

    # get average slopes and y-intercept
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # get the points
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    return np.array([left_line, right_line])

############################################################################################################################################



starting_time = time.time()
frame_id = 0



while True:
    _, frame = cap.read()
    # frame_id is used to calculate fps
    frame_id += 1

    # the network only accepts one image format. This format is "blob"
    blob = cv2.dnn.blobFromImage(frame, 1/255,(256, 256), [0, 0, 0], 1, crop=False)
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
    findObjects(outputs, frame)

    # canny edge detection is an edge detection algorithm
    # reads the image and detects edges
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    # cv2.imshow("cropped_canny",cropped_canny)

    # gives an approximate presence of lines in an image
    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    # display the line on the image
    line_image = display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    # show the result
    cv2.imshow("result", combo_image)
    # plt.imshow(combo_image)

    # displays the image until "q" is pressed or video ends
    # wait 1 millisecond between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()