import cv2
import numpy as np

img1 = cv2.imread('mask_paint.png')

img2 = cv2.imread('lena_opencv.png')

new_img = cv2.resize(img1, (len(img2), len(img2[0])))

mask_img = np.zeros((len(img2), len(img2[0]), 3))

for i in range(0, len(new_img)):
	for j in range(0, len(new_img[0])):
		if(new_img[i][j][0]==255 and new_img[i][j][1]==255 and new_img[i][j][2]==255):
			mask_img[i][j][0] = 0
			mask_img[i][j][1] = 0
			mask_img[i][j][2] = 0
		else:
			mask_img[i][j][0] = 255
			mask_img[i][j][1] = 255
			mask_img[i][j][2] = 255

cv2.imwrite('mask_img.png', mask_img)

for i in range(0, len(img2)):
	for j in range(0, len(img2[0])):
		if(mask_img[i][j][0]==255):
			img2[i][j][0] = 0
			img2[i][j][1] = 255
			img2[i][j][2] = 0

cv2.imwrite('corrupt_image.png', img2)