import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay
import math
import heapq

INF = 1e6
EPS = 1e-6
		# if ((i,j) in narrow_band):
		# 	img_typ[i][j] = 2
		# elif(polygon_band.contains(Point(i,j))):
		# 	img_typ[i][j] = 1
		# 	img_dis[i][j] = 1000000
		# else:
		# 	img_typ[i][j] = 0
		# 	img_dis[i][j] = 0
def pixel_gradient(x,y):
	dis = img_dis[x][y]
	y_prev = y - 1
	y_next = y + 1
	if y_prev < 0 and y_next >= len(img1[0]):
		y_grad = INF
	else:
		if img_typ[x][y_prev]!=1 and img_typ[x][y_next]!=1 :
			y_grad = (img_dis[x][y_next] - img_dis[x][y_prev])/2
		elif img_typ[x][y_prev] != 1:
			y_grad = dis - img_dis[x][y_prev]
		elif img_typ[x][y_next] != 1:
			y_grad = img_dis[x][y_next] - dis
		else:
			y_grad = 0.0
	x_prev = x - 1
	x_next = x + 1
	if x_prev < 0 and x_next >= len(img1):
		x_grad = INF
	else:
		if img_typ[x_prev][y]!=1 and img_typ[x_next][y]!=1 :
			x_grad = (img_dis[x_next][y] - img_dis[x_prev][y])/2
		elif img_typ[x_prev][y] != 1:
			x_grad = dis - img_dis[x_prev][y]
		elif img_typ[x_next][y] != 1:
			x_grad = img_dis[x_next][y] - dis
		else:
			x_grad = 0.0
	return x_grad,y_grad

def inpaint(x,y):
	radius = 4
	grad_x,grad_y = pixel_gradient(x,y)
	p = np.zeros((3), dtype=float)
	s = 0.0
	for x_n in range(x - radius,x + radius + 1):
		if(x_n<0 or x_n>len(img1)):
			continue
		for y_n in range(y - radius,y + radius + 1):
			if(y_n<0 or y_n>len(img1)):
				continue
			if(img_typ[x_n][y_n]==1):
				continue
			if x_n == x and y_n == y:
				continue
			r = (x-x_n,y-y_n)
			len_r = math.sqrt(r[0]*r[0]+r[1]*r[1])
			if len_r > radius:
				continue
			dir_r = abs(r[0]*grad_x+r[1]*grad_y)
			if dir_r == 0:
				dir_r = EPS
			dst = 1/(len_r*len_r)
			lev = 1/(1+math.fabs(img_dis[ele[0]][ele[1]]-img_dis[x][y]))
			w = abs(dir_r*dst*lev)
			p[0]  += w * img1[x_n][y_n][0]
			p[1]  += w * img1[x_n][y_n][1]
			p[2]  += w * img1[x_n][y_n][2]
			s += w
	return p/s



def solve(x,y,z,w):
	
	if x < 0 or x >= len(img1) or y < 0 or y >= len(img1[0]):
		return INF

	if z < 0 or z >= len(img1[0]) or w < 0 or w >= len(img1[0]):
		return INF

	sol = INF
	if(img_typ[x][y]!=1):
		if(img_typ[z][w]!=1):
			d = 2 - (img_dis[x][y]-img_dis[z][w])**2
			if d > 0.0:
				r = math.sqrt(d)
				s = (img_dis[x][y]+img_dis[z][w]-r)/2.0
				if (s>=img_dis[x][y] and s>=img_dis[z][w]):
					sol = s
				else:
					s += r
					if(s>=img_dis[x][y] and s>=img_dis[z][w]):
						sol = s
		else:
			sol = 1 + img_dis[x][y]
	elif(img_typ[z][w]==2):
		sol = 1 + img_dis[z][w]
	return sol






img1 = cv.imread('corrupt_image.png')
print(img1)
img_m = cv.imread('mask_img.png')
img_mask = np.zeros((len(img_m),len(img_m[0])))
for i in range(len(img_m)):
	for j in range(len(img_m[0])):
		if(img_m[i][j][0]>0):
			img_mask[i][j] = 1

#mask banana hai
# img_dis = np.zeros((len(img1), len(img1[0])),INF, dtype = img1.dtype)
#fill the outline of pixels
img_typ = img_mask.copy()

img_dis = img_mask * 1e6

heap_band = []

mask_x,mask_y = img_mask.nonzero()

for x,y in zip(mask_x,mask_y):
	neigh = [(x,y-1),(x-1,y),(x+1,y),(x,y+1)]
	for n_x,n_y in neigh:
		if n_x < 0 or n_x >= len(img_mask) or n_y < 0 or n_y >= len(img_mask[0]):
			continue
		if img_typ[n_x][n_y] == 2:
			continue
		if img_mask[n_x][n_y] == 0:
			img_typ[n_x][n_y] = 2
			img_dis[n_x][n_y] = 0.0
			heapq.heappush(heap_band,(0.0,n_x,n_y))

print(len(heap_band))
while(heap_band):
	print(len(heap_band))
	z,x,y = heapq.heappop(heap_band)
	img_typ[x][y] = 0

	for ele in [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]:
		if ele[0] < 0 or ele[0] >= len(img1) or ele[1] < 0 or ele[1] >= len(img1[0]):
			continue
		if(img_typ[ele[0]][ele[1]]!=1):
			continue
		m1 = solve(ele[0]-1,ele[1],ele[0],ele[1]-1)
		m2 = solve(ele[0]+1,ele[1],ele[0],ele[1]-1)
		m3 = solve(ele[0]-1,ele[1],ele[0],ele[1]+1)
		m4 = solve(ele[0]+1,ele[1],ele[0],ele[1]+1)
		img_dis[ele[0]][ele[1]] = min([m1,m2,m3,m4])
		new_pixel = inpaint(ele[0],ele[1])
		if(new_pixel[0]==0.0):
			break
		img1[ele[0]][ele[1]][0] = new_pixel[0]
		img1[ele[0]][ele[1]][1] = new_pixel[1]
		img1[ele[0]][ele[1]][2] = new_pixel[2]
		img_typ[ele[0]][ele[1]] = 2 
		heapq.heappush(heap_band,(img_dis[ele[0]][ele[1]],ele[0],ele[1]))



cv.imshow('det',img1)
cv.waitKey(0)