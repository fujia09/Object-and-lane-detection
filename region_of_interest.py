import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math

cap = cv2.VideoCapture("road_car_view.mp4")
#cap = cv2.imread("test.jpg")

def region_of_interest(canny):
    # obtain height of image
    height = canny.shape[0]
    # obtain width of image
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    # triangle removes everything except the points of interest
    # change triangle coordinates based on where the lanes are
    triangle = np.array([[
        (0, height),
        (width/1.8, height/2.5),
        (width, height), ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # convert rgb image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    # blur the image before passing it into the algorithm. In order to reduce the noise level in the image
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# houghLines only works on straight lines, not curved roads
# bin with maximum number of votes in houghspace/intersection of the sinousidal graphs of rho and theta in houghspace
def houghLines(cropped_canny):
    """threshold is 100, which is the minimum number of intersections in a bin is 100
    minlinelength is 40 of any lines detected that are less than 40 pixels are rejected
    maxlinegap is the maximum distance in pixels between different lines which we will
    allow to be connected into a single line instead of them being broken up"""

    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100,
                           np.array([]), minLineLength=40, maxLineGap=100)

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


while True:
    _, frame = cap.read()
    # frame_id is used to calculate fps


    # canny edge detection is an edge detection algorithm
    # reads the image and detects edges
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    cv2.imshow("cropped_canny",cropped_canny)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#plt.imshow(cropped_canny)
#plt.show()

