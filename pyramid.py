import cv2
import numpy as np
import sys
from scipy import sparse

# inp = cv2.imread('apple_high.jpg')
# inp = cv2.resize(inp, (1024, 1024))
# cv2.imwrite('apple_high.jpg', inp)
# exit()

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

def expand(inp_img, size, fac):
	return cv2.GaussianBlur(zoomin(inp_img, fac), (size, size), 0)

def get_gaussian(img, lvls, size, fac):
	gaussian = []

	for i in range(0, lvls):
		gaussian.append(img)
		img = reduce(img, size, fac)

	return gaussian

def get_laplacian(img, lvls, size, fac):
	laplacian = []

	for i in range(0, lvls):
		temp_img = reduce(img, size, fac)
		laplacian.append(img - expand(temp_img, size, fac))
		img = temp_img
	laplacian.append(img)

	return laplacian

def recreate(lapl, size, fac):
	img = lapl[-1]

	for i in range(1, len(lapl)):
		ind = len(lapl) - i - 1
		img = expand(img, size, fac) + lapl[ind]

	return img

def mosaic(lapl1, lapl2, size, fac):
	new_lap = []
	for ele1, ele2 in zip(lapl1, lapl2):
		for i in range(0, len(ele1)):
			for j in range(0, int(len(ele1[0])/2)):
				ele1[i][j][0] = ele2[i][j][0]
				ele1[i][j][1] = ele2[i][j][1]
				ele1[i][j][2] = ele2[i][j][2]
		new_lap.append(ele1)

	return recreate(new_lap, size, fac)

def create_gaussian(img, lvls, size, fac):
	gaussian = get_gaussian(img, lvls, size, fac)

	for i in range(0, len(gaussian)):
		cv2.imwrite('gauss' + str(i) + '.jpg', gaussian[i])


def create_laplacian(img, lvls, size, fac):
	laplacian = get_laplacian(img, lvls, size, fac)
	
	for i in range(0, len(laplacian)):
		cv2.imwrite('lapl' + str(i) + '.jpg', laplacian[i])

	rec = recreate(laplacian, size, fac)
	cv2.imwrite('recreated.jpg', rec)

def compress(img, lvls, size, fac):
	print("Old Image Size : ", img.nbytes/1024, " kb")

	laplacian = get_laplacian(img, lvls, size, fac)

	uncom_size = 0
	com_size = 0
	for ele in laplacian:
		for i in range(0, 3):
			temp_ele = ele[:, :, i]
			com_ele = sparse.csr_matrix(temp_ele)
			uncom_size += temp_ele.nbytes
			com_size += com_ele.data.nbytes

	print("Laplacian Pyramid Size (Numpy Array) : ", uncom_size/1024, " kb")
	print("Laplacian Pyramid Size (Sparse Matrix) : ", com_size/1024, " kb")
	
def create_mosaic(img1, img2, lvls, size, fac):
	laplacian1 = get_laplacian(img1, lvls, size, fac)
	laplacian2 = get_laplacian(img2, lvls, size, fac)
	joined = mosaic(laplacian1, laplacian2, size, fac)

	joined = mosaic(laplacian1, laplacian2, size, fac)
	cv2.imwrite('mosaic.jpg', joined)


img1 = cv2.imread('apple.jpg')
img2 = cv2.imread('orange.jpg')

img1 = np.array(img1, dtype=float)
img2 = np.array(img2, dtype=float)

lvls = 5
size = 5
fac = 2

# create_gaussian(img1, lvls, size, fac)
# create_laplacian(img1, lvls, size, fac)
# compress(img1, lvls, size, fac)
create_mosaic(img1, img2, lvls, size, fac)