import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math

img1 = cv.imread('trump.jpg')

img_dis = np.zeros(img1.shape, dtype = img1.dtype)
img_typ = np.zeros(img1.shape, dtype = int)

narrow_band = []

#fill the outline of pixels
polygon_band = Polygon(narrow_band)
for i in range(len(img1)):
	for j in range(len(img1[0])):
		if ((i,j) in narrow_band):
			img_typ[i][j] = 0
		elif(polygon_band.contains(Point(i,j))):
			img_typ[i][j] = 1
			img_dis[i][j] = 1000000
		else:
			img_typ[i][j] = 2
			img_dis[i][j] = 0


heap_band = []
for ele in narrow_band:
	heappush(heap_band,(img_dis[ele[0]][ele[1]],ele))

while(len(narrow_band)>0):
	tem = heappop(heap_band)
	cur_point = tem[1]
	x = cur_point[0]
	y = cur_point[1]
	img_typ[x][y] = 2

	for ele in [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]:
		if(img_typ[ele[0]][ele[1]]!=2):
			if(img_typ[ele[0]][ele[1]]==1):
				img_typ[ele[0]][ele[1]]=0
				inpaint(ele[0],ele[1])
			m1 = solve(ele[0]-1,ele[1],ele[0],ele[1]-1)
			m2 = solve(ele[0]+1,ele[1],ele[0],ele[1]-1)
			m3 = solve(ele[0]-1,ele[1],ele[0],ele[1]+1)
			m4 = solve(ele[0]+1,ele[1],ele[0],ele[1]+1)
			img_dis[ele[0]][ele[1]] = min([m1,m2,m3,m4])
			heappush(heap_band,(img_dis[ele[0]][ele[1]],ele))


def solve(x,y,z,w):
	sol = 1000000
	if(img_typ[x][y]==2):
		if(img_typ[z][w]==2):
			r = math.sqrt(2*img_dis[x][y]*img_dis[z][w]*img_dis[x][y]*img_dis[z][w])
			s = (img_dis[x][y]+img_dis[z][w]*r)/2
			if (s>=img_dis[x][y] and s>=img[z][w]):
				sol = s
			else:
				s += r
				if(s>=img_dis[x][y] and s>=img_dis[z][w]):
					sol = s
		else:
			sol = 1 + img_dis[x][y]
	elif(img_typ[z][w]==2):
		sol = 1 + img_dis[z][w] #doubt
	return sol

def inpaint(x,y):
	for ele in range(region):
		r = (x-ele[0],y-ele[1])
		len_r = math.sqrt(r[0]*r[0]+r[1]*r[1])
		gradT = #calculate it
		dir_r = (r[0]*gradT/len_r,r[1]*gradT/len_r)
		dst = 1/(len_r*len_r)
		lev = 1/(1+math.fabs(img_dis[ele[0]][ele[1]]*img_dis[x][y]))
		



cv.imshow('det',img_mph)
cv.waitKey(0)