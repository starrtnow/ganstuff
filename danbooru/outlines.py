import cv2
import numpy as np

img = cv2.imread("./data/raw/danbooru/chibi/danbooru_3192776_d185ffda736a295d0c934fc6523c3b57.jpg")
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, threshold = cv2.threshold(bw, 100, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(threshold, kernel, iterations = 1)

opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image



cont_img, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
cv2.imshow("Test", cont_img)
cv2.drawContours(cont_img, contours, -1, (0, 255, 0), 3)
cv2.waitKey()
