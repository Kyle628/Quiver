import numpy as np
import cv2 as cv

im = cv.imread('surfboard.jpg')
cannyIm = cv.Canny(im,100,200)

# Perform morphology (close shape)
se = np.ones((7,7), dtype='uint8')
imageClose = cv.morphologyEx(cannyIm, cv.MORPH_CLOSE, se)

ret, thresh = cv.threshold(imageClose, 127, 255, 0)
#contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

mask = np.zeros(imageClose.shape[:2], np.uint8)
outerContour = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

longestContourLength = 0
for i, contour in enumerate(outerContour):
    if cv.arcLength(contour, True) > longestContourLength:
        longestContourLength = cv.arcLength(contour, True)
        longestContour = contour
        longestContourIndex = i

contourImg = cv.drawContours(mask, outerContour, longestContourIndex, (255,255,255), 3)
#contourImg = cv.drawContours(mask, contours, -1, (255,255,255), 3)
cv.imwrite('./contour.jpg', contourImg)


