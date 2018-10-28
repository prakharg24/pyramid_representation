import numpy as np
import cv2 as cv
import sys

def get_avg(img, i, j, zoom, k):
	istart = max(0, i-int((zoom-1)/2))
	jstart = max(0, j-int((zoom-1)/2))

	iend = min(0, i+int(zoom/2))
	jend = min(0, j+int(zoom/2))

	sum_img = 0
	num_img = 0
	for i in range(istart, iend+1):
		for j in range(jstart, jend+1):
			sum_img += img[i][j][k]
			num_img += 1

	return sum_img/num_img

if(len(sys.argv)!=4):
	print("Required -> python Zout_inter.py <input_image> <zoom> <output_image>")
	exit()

img = cv.imread(sys.argv[1])
zoom = int(sys.argv[2])

img = np.array(img)

new_img = np.zeros((int(len(img)/zoom), int(len(img[0])/zoom), 3))

for i in range(len(new_img)):
	if(i%100==0):
		print(i, "out of", len(new_img), "done")
	for j in range(len(new_img[0])):
		new_img[i][j][0] = get_avg(img, i*zoom, j*zoom, zoom, 0)
		new_img[i][j][1] = get_avg(img, i*zoom, j*zoom, zoom, 1)
		new_img[i][j][2] = get_avg(img, i*zoom, j*zoom, zoom, 2)


cv.imwrite(sys.argv[3], new_img)
