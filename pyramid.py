import cv2
import numpy as np

def zoomout(inp_img, zoom):
	new_img = np.zeros((int(len(inp_img)/zoom), int(len(inp_img[0])/zoom), 3))

	for i in range(len(new_img)):
		for j in range(len(new_img[0])):
			new_img[i][j][0] = inp_img[i*zoom][j*zoom][0]
			new_img[i][j][1] = inp_img[i*zoom][j*zoom][1]
			new_img[i][j][2] = inp_img[i*zoom][j*zoom][2]

	return new_img

def zoomin(inp_img, zoom):
	new_img = np.zeros((zoom*len(inp_img), zoom*len(inp_img[0]), 3))

	for i in range(len(new_img)):
		for j in range(len(new_img[0])):
			new_img[i][j][0] = inp_img[int(i/zoom)][int(j/zoom)][0]
			new_img[i][j][1] = inp_img[int(i/zoom)][int(j/zoom)][1]
			new_img[i][j][2] = inp_img[int(i/zoom)][int(j/zoom)][2]

	return new_img

def reduce(inp_img, size, fac):
	blur = cv2.GaussianBlur(inp_img,(size,size),0)
	return zoomout(blur, fac)

def expand(inp_img, fac):
	return zoomin(inp_img, fac)

img = cv2.imread('example.jpg')

lvls = 5

gaussian = []

for i in range(0, lvls):
	gaussian.append(img)
	cv2.imwrite('gauss' + str(i) + '.jpg', img)
	img = reduce(img, 5, 2)
