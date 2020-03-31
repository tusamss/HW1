import cv2
import numpy as np
import time

img = cv2.imread('lena.jpg')
# -----------------------------------------------------
cv2.setUseOptimized(True) # open AVX
cv2.useOptimized()
# -----------------------------------------------------
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
tStart = time.time()
# -----------------------------------------------------
for i in range (1,100,1) :
    kernel    = np.ones((5,5),np.uint8)
    frame_mor = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    frame_mor = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
# -----------------------------------------------------
tEnd = time.time()
t_total = tEnd - tStart
print("With AVX : %f s" % t_total)
# -----------------------------------------------------
cv2.namedWindow("With_AVX",0)
cv2.imshow('With_AVX', frame_mor)
# ---------------------------------------------------------------------------------------------------------------------
img = cv2.imread('lena.jpg')
# -----------------------------------------------------
cv2.setUseOptimized(False) # close AVX
cv2.useOptimized()
# -----------------------------------------------------
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
tStart = time.time()
# -----------------------------------------------------
for i in range (1,100,1) :
    kernel    = np.ones((5,5),np.uint8)
    frame_mor = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    frame_mor = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
# -----------------------------------------------------
tEnd = time.time()
t_total = tEnd - tStart
print("Without AVX : %f s" % t_total)
# -----------------------------------------------------
cv2.namedWindow("Without_AVX",0)
cv2.imshow('Without_AVX', frame_mor)

cv2.waitKey(0)
cv2.destroyAllWindows()