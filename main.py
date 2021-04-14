# -*- coding: utf-8 -*-

"""
RoadTrip
Roads & Cars Detection
"""

# Dependency: File management

import os
import sys
from termcolor import colored, cprint

""" add custom modules to PYTHONPATH """
where_am_i = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, where_am_i+"/python_modules")

from os import path
import getopt
from shutil import copyfile
import shutil
from skimage import data_dir
from contextlib import redirect_stdout
import io as ioo

# Dependency: Server

from flask import Flask, request, redirect, url_for, flash, Response, send_from_directory
from flask_mobility import Mobility
from flask_talisman import Talisman
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Dependency: Procceses

from threading import Thread
from tqdm import tqdm


# Dependency: Neural Nets

from classify import *
from count import *

# Dependency: Images and arrays

from PIL import Image
import numpy as np


# Dependency: SciKit

from skimage.morphology import skeletonize_3d
from skimage.data import load
from skimage import io

# Dependency: Math

from collections import namedtuple  
import matplotlib.pyplot as plt
from random import uniform
import math
from math import sin, cos, sqrt, atan2, radians, ceil

def cmd(line):
	os.system(line)

#Loads Car Results from Car Detection Model output
def carResults():
	csvPath = os.getcwd() + '/car_count.csv'

	raw = open(csvPath, "r").read()

	results = [car for car in raw.split('\n')]

	return results

#CLI: open car_predictions.csv
def openCarPredictions():
	cmd('open Car_Detection/car_predictions')

#CLI: open stitched-images.png
def openMap():
	cmd('open files/stitched-images.png')

#CLI: open map.svg'
def openRoadMap():
	cmd('open maps/map.svg')

#CLI: open premap.bmp
def openRoadPremap():
	cmd('open files/premap.bmp')

#CLI: run shell script form path
def runSh(path):
	lines = open(path, "r").read().split('\n')
	for l in lines:
		print(l)
		cmd(l)

#Square Root
def sqrt(n):
	return n**(0.5)

#Create missing directories for project
def createDirectories():

	for _dir in [data_dir+'/static/', data_dir+'/skeletons/', data_dir+'/files/', os.getcwd()+'/skeletons/', os.getcwd()+'/tiles/', os.getcwd()+'/files/', os.getcwd()+'/maps/', os.getcwd()+'/tests/', os.getcwd()+'/images/', os.getcwd()+'/dataset/', os.getcwd()+'/predictions/', os.getcwd()+'/Car_Detection/car_predictions']:
		if not os.path.exists(_dir):
			os.makedirs(_dir)
	"""
	if not os.path.exists(data_dir+'/static/'):
		os.makedirs(data_dir+'/static/')
	if not os.path.exists(data_dir+'/skeletons/'):
		os.makedirs(data_dir+'/skeletons/')
	if not os.path.exists(data_dir+'/files/'):
		os.makedirs(data_dir+'/files/')
	if not os.path.exists(os.getcwd()+'/skeletons/'):
		os.makedirs(os.getcwd()+'/skeletons/')
	if not os.path.exists(os.getcwd()+'/tiles/'):
		os.makedirs(os.getcwd()+'/tiles/')
	if not os.path.exists(os.getcwd()+'/files/'):
		os.makedirs(os.getcwd()+'/files/')
	if not os.path.exists(os.getcwd()+'/maps/'):
		os.makedirs(os.getcwd()+'/maps/')
	if not os.path.exists(os.getcwd()+'/tests/'):
		os.makedirs(os.getcwd()+'/tests/')
	if not os.path.exists(os.getcwd()+'/images/'):
		os.makedirs(os.getcwd()+'/images/')
	if not os.path.exists(os.getcwd()+'/dataset/'):
		os.makedirs(os.getcwd()+'/dataset/')
	if not os.path.exists(os.getcwd()+'/predictions/'):
		os.makedirs(os.getcwd()+'/predictions/')
	if not os.path.exists(os.getcwd()+'/Car_Detection/car_predictions'):
		os.makedirs(os.getcwd()+'/Car_Detection/car_predictions')
	"""

def oitc(i, width, height):
	col = width - ceil(width - i / height) - 1
	row = height - 1 - i % height if col % 2 == 0 else i % height
	return (row, col)

#Mirror matrix
def getMirrored(x = None, y = None):
	if x == None:
		x = y = 3

	arr = np.zeros((x,y))

	xSize, ySize = arr.shape[0], arr.shape[1]

	for i in range(xSize*ySize):
		x, y = oitc(i, xSize, ySize)
		arr[x][1+y] = i

	arr = np.flip(arr, (0,1))

	wrong = np.flip(arr, (1,0))

	dicty = {}

	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]):
			dicty[str(arr[i][j])[0:1]] = str(wrong[i][j])[0:1]

	return dicty

#Load georefernces from csv file. csv format: image, latTopLeftCorner, longTopLeftCorner, latDownRightCorner, longDownRightCorner
def loadLocations(path):
	csvPath = os.getcwd()+ path

	raw = open(csvPath, "r").read()

	return [loc.split(',')[1:] for loc in raw.split('\n')]

#Add georefernces to dictionary
def dictGeoreferences(path, correction=False):

	print('[INFO] Loading georeferences...')

	csvPath = os.getcwd() + path

	raw = open(csvPath, "r").read()

	dictionary = {}

	for loc in raw.split('\n'):

		imid = loc.split(',')[0]

		dictionary[str(imid)] = loc.split(',')[1:]

	if not correction:
		return dictionary
	else:
		print('[INFO] Correcting georeferences by mirror method...')
		new_dictionary = {}
		poss = getMirrored()
		print('[INFO] Done')
		for i in range(len(raw.split('\n'))):
			new_dictionary[str(poss[str(i)])] = dictionary[str(i)]

		return new_dictionary
		
#Calculate resolution in meters from georefernces
def calcRes(path, correction=False):
	dicty = dictGeoreferences(path, correction)

	x = [dicty['0'][1], dicty['0'][0]]
	y = [dicty['0'][3], dicty['0'][0]]

	meters = coordinatesToMeters(x, y)

	width = Image.open('images/image-0.jpg').size[0]

	print('[INFO] Width in meters of image', width/meters)
	return width/meters

#Random lat lon pair.
def randomPair(latRange, lonRange):
	lat = round(uniform(latRange[0], latRange[1]), 8)
	lon = round(uniform(lonRange[0], lonRange[1]), 8)

	return lat, lon

#Add earths radius as offset in meters to georefences
def offset(lat, lon, offset):
	"""Earth's radius, sphere"""

	lat = float(lat)
	lon = float(lon)

	R = 6378137

	"""offsets in meters"""

	dn = offset
	de = offset

	"""Coordinate offsets in radians"""

	dLat = dn/R
	dLon = de/(R*math.cos(math.pi*lat/180))

	"""OffsetPosition, decimal degrees"""

	latO = round(lat + dLat * 180/math.pi, 8)
	lonO = round(lon + dLon * 180/math.pi, 8)

	return latO, lonO

#Distance bewtween lat lon coordinates in meters.
def coordinatesToMeters(point, point2):
	
	lat1 = radians(float(point[1]))
	lon1 = radians(float(point[0]))
	lat2 = radians(float(point2[1]))
	lon2 = radians(float(point2[0]))

	"""approximate radius of earth in km"""

	R = 6378.137

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	distance = R * c

	return distance*1000
	

#Distance bewtween lat lon coordinates in pixels.
def coordinatesToPixels(coordinates):

	outmosts = outmostLatLong(coordinates, north = True, east = True)

	top = outmosts[0]
	left = outmosts[0]

	pixReferences = [(coordinatesToMeters([top, coord[1]], [coord[0], coord[1]]), coordinatesToMeters([coord[0], left], [coord[0], coord[1]])) for coord in coordinates]

	return pixReferences

#Order coorners out of 4 points
def outerPoints(path):

	coordinates = loadLocations(path)

	"""latTopLeftCorner, longTopLeftCorner, latDownRightCorner, longDownRightCorner"""

	maxLatTop = max([coord[0] for coord in coordinates])
	minLonTop = min([coord[1] for coord in coordinates])
	minLatDown = min([coord[2] for coord in coordinates])
	maxLonDown = max([coord[3] for coord in coordinates])

	full_Corner1 = [minLonTop,maxLatTop]
	full_Corner2 = [maxLonDown,maxLatTop]
	full_Corner3 = [maxLonDown,minLatDown]
	full_Corner4 = [minLonTop,minLatDown]

	return full_Corner1, full_Corner2, full_Corner3, full_Corner4

#Generate random rect
def randomRectangle(latRange, lonRange, size):

	x_Lat = round(uniform(latRange[0], latRange[1]), 8)
	y_100 = round(uniform(lonRange[0], lonRange[1]), 8)

	x_100 = offset(x_Lat, y_100, size)[0]
	y_Lon = offset(x_Lat, y_100, size)[1]

	return [[x_Lat, y_Lon], [x_100, y_Lon],	[x_100, y_100], [x_Lat, y_100]]

#Load tile georefernces
def tileReferences(path):

	"""latTopLeftCorner, longTopLeftCorner, latDownRightCorner, longDownRightCorner"""

	refs_0_0, full_Corner2, full_Corner3, full_Corner4 = outerPoints(path)

	full_Corner2 = None
	full_Corner3 = None
	full_Corner4 = None

	refs = [[]]

	coordinates = loadLocations(path)

	boardMinLon, boardMaxLat = refs_0_0[0], refs_0_0[1]

	for coord in coordinates:

		lonTopLeftCorner = coord[1]
		latTopLeftCorner = coord[0]

		x = coordinatesToMeters([boardMinLon, latTopLeftCorner], [lonTopLeftCorner, latTopLeftCorner])

		y = coordinatesToMeters([lonTopLeftCorner, boardMaxLat], [lonTopLeftCorner, latTopLeftCorner])

		refs.append([x,y])

	return refs

#Add tile georefernces to dictionary
def dictTileReferences(path, correction=False):

	print('[INFO] Asociation geoeferences to images')

	refs_0_0, full_Corner2, full_Corner3, full_Corner4 = outerPoints(path)

	full_Corner2 = None
	full_Corner3 = None
	full_Corner4 = None

	boardMinLon, boardMaxLat = refs_0_0[0], refs_0_0[1]

	"""
		image, latTopLeftCorner, longTopLeftCorner, latDownRightCorner, longDownRightCorner
		latTopLeftCorner, longTopLeftCorner, latDownRightCorner, longDownRightCorner
	"""
	refs = dictGeoreferences(path, correction)

	dictionary = {}

	print('[INFO] Measuring distances between georeferences...')
	
	for i in range(len(refs)):

		coord = refs[str(i)]

		lonTopLeftCorner = coord[1]
		latTopLeftCorner = coord[0]

		x = coordinatesToMeters([boardMinLon, latTopLeftCorner], [lonTopLeftCorner, latTopLeftCorner])

		y = coordinatesToMeters([lonTopLeftCorner, boardMaxLat], [lonTopLeftCorner, latTopLeftCorner])

		dictionary[i] = [x,y]

	print('[INFO] Done measuring distances between georeferences: ', x, y)

	return dictionary


Point = namedtuple('Point', 'x y')

#Convex Hull
class ConvexHull(object):  
    _points = []
    _hull_points = []

    def __init__(self):
        pass

    #Add points to convex hull
    def add(self, point):
        self._points.append(point)

    #Convex hull orientation
    def _get_orientation(self, origin, p1, p2):
        
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    #Computes convex hull
    def compute_hull(self):
        
        points = self._points

        """get leftmost point"""

        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:

            """get the first point (initial max) to use to compare with others"""

            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                """ensure we aren't comparing to self or pivot point"""

                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    #Returns Convex Hull points
    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    #Matplot lib display o Convex Hull Points
    def display(self):
        """all points"""

        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')

        """hull points"""

        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)

        plt.title('Convex Hull')
        plt.show()

#Get size of map out of georefences
def giftWrappedBoardSize(georeferencesPath):  
    
    ch = ConvexHull()

    locs = loadLocations(georeferencesPath)

    for loc in locs:
        ch.add(Point(float(loc[0]), float(loc[1])))
        ch.add(Point(float(loc[2]), float(loc[3])))

    hull_points = ch.get_hull_points()

    lons = [abs(point[0]) for point in ch.get_hull_points()]
    lats = [abs(point[1]) for point in ch.get_hull_points()]

    maxLon = max(lons)
    minLon = min(lons)
    maxLat = max(lats)
    minLat = min(lats)
    
    point_1 = [minLon, maxLat]
    point_2 = [maxLon, maxLat]
    point_3 = [maxLon, minLat]
    point_4 = [minLon, minLat]

    print('[INFO] Outer Points:')
    print(point_1, point_2)
    print(point_4, point_3)

    W = coordinatesToMeters(point_1, point_2)

    H = coordinatesToMeters(point_1, point_4)

    """
    ch.display() 
    optional to display convex hull
    """

    return H, W

#Convex Hull fromg georeferences file
def coordConvexHull(georeferencesPath):
    
    ch = ConvexHull()

    locs = loadLocations(georeferencesPath)

    for loc in locs:
        ch.add(Point(float(loc[0]), float(loc[1])))
        ch.add(Point(float(loc[2]), float(loc[3])))

    hull_points = ch.get_hull_points()

    lons = [abs(point[0]) for point in ch.get_hull_points()]
    lats = [abs(point[1]) for point in ch.get_hull_points()]

    maxLon = max(lons)
    minLon = min(lons)
    maxLat = max(lats)
    minLat = min(lats)
    
    point_1 = [minLon, maxLat]
    point_2 = [maxLon, maxLat]
    point_3 = [maxLon, minLat]
    point_4 = [minLon, minLat]

    print('[INFO] Outer Points:')
    print(point_1, point_2)
    print(point_4, point_3) 

    W = coordinatesToMeters(point_1, point_2)

    H = coordinatesToMeters(point_1, point_4)

    print(H, W)

    ch.display()

    plt.plot(minLon, maxLat, '-*')
    plt.plot(maxLon, minLat, '-*')
    plt.plot(minLon, minLat, 'o')
    plt.plot(maxLon, maxLat, 'o')
    plt.title('Outmosts')
    plt.show()

#CLI: potrace bitmap to vector graphics
def bmp_to_svg(path, pathOut):
	cmd('potrace --svg {} -o {}'.format(path, pathOut))

#Lines out of street detection output
def sekeletonize(path, pathOut):
	data = load(path, as_gray=True, as_grey=True)
	
	skeleton3d = skeletonize_3d(data)

	io.imsave(pathOut, skeleton3d)

#Load Tile images from path
def loadTiles(path, amt = None, correction=False):

	ls = []

	print('[INFO] Loading tiles...')

	refs = dictTileReferences(path, correction)

	print('[INFO] Loading tiles...')
	if amt == None:
		cant = len(list(filter(lambda s: s.endswith(".jpg"), os.listdir("images/"))))
	else:
		cant = amt
	for i in range(cant):
		image_filename = 'tiles/tile-{}.bmp'.format(i)
		ls.append([Image.open(image_filename), refs[i]])
	return ls


#Load Satelite images from path
def loadImages(path, amt = None, correction=False, whereFrom='images/image-{}.jpg'):

	print('[INFO] Loading images...')

	ls = []

	refs = dictTileReferences(path, correction)

	if amt == None:
		cant = len(list(filter(lambda s: s.endswith(".jpg"), os.listdir("images/"))))
	else:
		cant = amt

	print('[INFO] Loading images...')
	for i in range(cant):
		if whereFrom != 'images/image-{}.jpg':
			image_filename = whereFrom.format(i).replace('\u2069', '')
		else:
			image_filename = whereFrom.format(i)
		print(image_filename)
		ls.append([Image.open(image_filename), refs[i]])
	print('[INFO] Done loading images...')
	return ls

#Create Empty Map size of all georefences
def createBoard(path, res):

	print('[INFO] Gift wrapping...')
	H, W = giftWrappedBoardSize(path)
	
	W = H

	while H > 10000:
		print('[INFO] Board Height > 10000...')
		H = int(input('Insert Height (Mts): '))
		W = H

	return Image.new("RGB", (int(H * res), int(W * res)), "black")

#Empty Square Map
def createBoardPerfectSquares():

	H = sqrt((len(os.listdir(os.getcwd()+'/images/'))-1))*Image.open('images/image-0.jpg').size[0]

	W = H

	print('[INFO] Size: h=', H,'w=', W)

	if H > 10000:
		print('[WARNING] Board Height > 10000')
		H = int(input('Insert Height (Mts): '))
		W = H

	return Image.new("RGB", (int(H), int(W)), "black")

#Fit tiles to empty map
def fitTiles(board, tiles, res):

	print('[INFO] Stitching Tiles...')
	for i in range(len(tiles)):
		board.paste(tiles[i][0], (int(tiles[i][1][0] * res), int(tiles[i][1][1] * res)))
		board.save('files/premap-saved-{}.bmp'.format(i))
	return board

#Stitch images toghether out of georefences
def stitchImages(path, res, amt = None, correction=False, where='images/image-{}.jpg', outputPath='files/stitched-images.png'):

	images = loadImages(path, amt, correction, whereFrom=where)
	
	print('[INFO] Building board...')
	board = createBoard(path, res)

	print('[INFO] Fitting tiles...')
	full = fitTiles(board, images, res)

	print('[INFO] Saving composite...')
	full.save(outputPath)

	print('[INFO] Done')

#Stitched road detection output as masked map
def stitchMasks(path, res, amt = None, correction=False):

	images = loadTiles(path, amt, correction)
	board = createBoard(path, res)

	"""board = createBoardPerfectSquares() as alternative to createBoard(path, res)"""

	full = fitTiles(board, images, res)

	full.save('files/stitched-masks.png')

#Scale image
def scale(img, fullSize1, fullSize2, roadSize):

	ratio = img.size[0]

	new = Image.new('1', (fullSize1, fullSize2), 'black')

	roadblock = Image.new('1', (roadSize, roadSize), 'white')

	roadpxs = roadblock.load()

	for i in range(img.size[0]):
		for j in range(img.size[1]):
			if img.getpixel((i, j)) > 125:
				new.paste(roadblock, (int(i*fullSize1/img.size[0]), int(j*fullSize2/img.size[1])))

	return new	


def predictionDepiction(t, roadSize = 32):
	
	prediction = 'predictions/prediction-{}.bmp'.format(t)
	
	pathToSave = 'tiles/tile-{}.bmp'.format(t)
	predImage = Image.open(prediction)

	fullSize1 = predImage.size[0] * roadSize
	fullSize2 = predImage.size[1] * roadSize

	tile = scale(predImage, fullSize1, fullSize2, roadSize)

	tile.save(pathToSave)

#Multiple satelital images
def multiple(georeferencesPath, res, correction=False):

	"""Neural Net Predicts roads from /images and saves to /predictions"""

	print('[INFO] Enlarging Predictions...')

	cant = len(list(filter(lambda s: s.endswith(".bmp"), os.listdir("predictions/"))))

	for i in tqdm(range(cant)):
		predictionDepiction(i) 
	"""visualization.py Creates 'tiles/tile-i.bmp' from 'predictions/prediction-i.png'"""

	tiles = loadTiles(georeferencesPath, correction=correction) 
	"""stitching.py Returns list of PIL.Images from 'tiles/'' files and appends locations to each from '/dataset/georeferences.csv'"""

	board = createBoard(georeferencesPath, res) 
	"""stitching.py Returns PIl.Image of dimentions extracted from '/dataset/georeferences.csv' outmosts latitude and longitud converted to meters (Res of 1m^2/pix)"""

	premap = fitTiles(board, tiles, res) 
	"""stitching.py Pastes tiles into board with lat/long coordinates as reference"""

	premap.save('files/premap.bmp')
	
	premap.save(data_dir+'/files/premap.bmp')

	print('[INFO] Skeletonizing...')
	sekeletonize('files/premap.bmp', 'skeletons/map.bmp') 
	"""skeletonization.py Creates line image out of original squares."""

	print('[INFO] Vectorizing...')
	bmp_to_svg('skeletons/map.bmp', 'maps/map.svg') 
	"""vectorization.py Vectorizes line image."""

#Singe satelite image
def single():

	"""Neural Net Predicts roads from /images and saves to /predictions"""

	predictionDepiction(0)

	premap = Image.open('tiles/tile-0.bmp')

	premap.save(data_dir+'/files/premap.bmp')

	sekeletonize('files/premap.bmp', 'skeletons/map.bmp') 
	"""skeletonization.py Creates line image out of original squares."""

	bmp_to_svg('skeletons/map.bmp', 'maps/map.svg')

#Multiple satelite images that do not compose a map
def separated():

	"""Neural Net Predicts roads from /images and saves to /predictions"""	
	
	print('[INFO] Enlarging Predictions')
	cant = len(list(filter(lambda s: s.endswith(".bmp"), os.listdir("predictions/"))))

	for i in tqdm(range(cant)):
		predictionDepiction(i)

	print('[INFO] Creating Vector Maps...')
	for i in tqdm(range(len(os.listdir(os.getcwd()+'/predictions/')))):

		premap = Image.open('tiles/tile-{}.bmp'.format(i))

		premap.save(data_dir+'/files/premap-{}.bmp'.format(i))

		sekeletonize('files/premap-{}.bmp'.format(i), 'skeletons/map-{}.bmp'.format(i)) 
		"""skeletonization.py Creates line image out of original squares."""

		bmp_to_svg('skeletons/map-{}.bmp'.format(i), 'maps/map-{}.svg'.format(i))

#Multiple road detection from satelite images
def multipleNoSkeleton(georeferencesPath, res, correction=False):

	"""Neural Net Predicts roads from /images and saves to /predictions"""

	print('[INFO] Enlarging Predictions...')

	cant = len(list(filter(lambda s: s.endswith(".bmp"), os.listdir("predictions/"))))

	for i in tqdm(range(cant)):
		predictionDepiction(i)

	tiles = loadTiles(georeferencesPath, correction=correction) 

	board = createBoard(georeferencesPath, res) 

	premap = fitTiles(board, tiles, res) 
	premap.save('files/premap.bmp')
	premap.save(data_dir+'/files/premap.bmp')

	bmp_to_svg('files/premap.bmp', 'maps/map.svg')

#Single road detection from satelite image
def singleNoSkeleton():

	"""Neural Net Predicts roads from /images and saves to /predictions"""

	predictionDepiction(0)

	premap = Image.open('tiles/tile-0.bmp')

	premap.save(data_dir+'/files/premap.bmp')

	premap.save('files/premap.bmp')
	
	bmp_to_svg('files/premap.bmp', 'maps/map.svg')


#Multiple road detection satelite images that do not compose a map
def separatedNoSkeleton():

	"""Neural Net Predicts roads from /images and saves to /predictions"""

	print('[INFO] Enlarging Predictions...')
	for i in tqdm(range(len(os.listdir(os.getcwd()+'/predictions/')))):

		predictionDepiction(i)

	print('[INFO] Creating Vector Maps (No Skeletons)...')
	for i in tqdm(range(len(os.listdir(os.getcwd()+'/predictions/')))):

		premap = Image.open('tiles/tile-{}.bmp'.format(i))

		premap.save(data_dir+'/files/premap-{}.bmp'.format(i))

		premap.save('files/premap-{}.bmp'.format(i))

		bmp_to_svg('files/premap-{}.bmp'.format(i), 'maps/map-{}.svg'.format(i))

#Full standardize routine: creates directories, loads georeferences and calculates resolution, stitches images, detects roads, generates a single map out of all satellital images, detects cars on masked map, outputs map depicting both road detecion and car detection.
def executeRoutine():

	createDirectories()
	
	georeferencesPath = '/images/georeferences.csv'
	
	res = calcRes(georeferencesPath)
	
	corrections = True
	
	print('[INFO] Stitching images...')
	stitchImages(georeferencesPath, res, correction=corrections)
	print('[INFO] Done stitching images...')

	predictRoads(full = True)
	
	singleNoSkeleton()
	
	predictRoads()
	
	multipleNoSkeleton(georeferencesPath, res, corrections)
	
	predictCars()

	carResults() 
	
	stitchImages(georeferencesPath, res, correction=corrections, where='Car_Detection⁩/car_predictions⁩/'+'cars_image-{}.jpg', outputPath='static/carPrediction.jpg')

#CLI input
def cli(cmmd):

	f = ioo.StringIO()
	with redirect_stdout(f):

		createDirectories()

		georeferencesPath = '/images/georeferences.csv'

		res = calcRes(georeferencesPath)

		corrections = True

		cmmds = ['break', 'routine', 'createDirectories', 'roads', 'openPredictions', 'cars', 'viewCars', 'simulatePr', 'multiple', 'single', 'separated', 'multiple!s', 'single!s', 'separated!s', 'convexHull', 'createGeorefs', 'ls', 'changeRes', 'changeGeorefsPath', 'stitchImages', 'stitchMasks', 'openStitchedMasks', 'openRoadPremap', 'openStitchedImages', 'openMap', 'openRoadMap', 'stitchCars', 'printRefs', 'calcRes', 'newLoc', 'quit', 'correct', 'clear']
		
		if cmmd == 'break':
			return ''
		if cmmd == 'routine':
			executeRoutine()
		elif cmmd == 'createDirectories':
			createDirectories()
		elif cmmd == 'road':
			predictRoads(full = True)
		elif cmmd == 'roads':
			if os.path.exists(os.getcwd()+'/predictions/'):
				shutil.rmtree('predictions')
			os.makedirs(os.getcwd()+'/predictions/')
			predictRoads()
		elif cmmd == 'openPredictions':
			cmd('open predictions/')
		elif cmmd == 'cars':
			predictCars()
			carResults()
		elif cmmd == 'viewCars':
			openCarPredictions()
		elif cmmd == 'simulatePr':
			simulatePredictions(int(input('Ammount')))
		elif cmmd == 'multiple':
			multiple(georeferencesPath, res, corrections)
		elif cmmd == 'single':
			single()
		elif cmmd == 'separated':
			separated()
		elif cmmd == 'multiple!s':
			multipleNoSkeleton(georeferencesPath, res, corrections)
		elif cmmd == 'single!s':
			singleNoSkeleton()
		elif cmmd == 'separated!s':
			separatedNoSkeleton()
		elif cmmd == 'convexHull':
			coordConvexHull(georeferencesPath)
		elif cmmd == 'createGeorefs':
			createGeoreferences(int(input('Ammount')), georeferencesPath)
		elif cmmd == 'ls' or cmmd == 'help':
			for cm in cmmds:
				print(cm, '<br>')
		elif cmmd == 'changeRes':
			print('Current Resolution: {}'.format(res))
			res = float(input('Enter New Resolution: '))
		elif cmmd == 'changeGeorefsPath':
			print('Current Path: "{}"'.format(georeferencesPath))
			georeferencesPath = str(input('Enter New Path: '))
		elif cmmd == 'stitchImages':
			stitchImages(georeferencesPath, res, amt = None, correction=corrections)
		elif cmmd == 'stitchMasks':
			stitchMasks(georeferencesPath, res, amt = None, correction=corrections)
		elif cmmd == 'openStitchedMasks' or cmmd == 'openRoadPremap':
			openRoadPremap()
		elif cmmd == 'openStitchedImages' or cmmd == 'openMap':
			openMap()
		elif cmmd == 'openRoadMap':
			openRoadMap()
		elif cmmd == 'printRefs':
			print(dictGeoreferences(georeferencesPath))
		elif cmmd == 'calcRes':
			res = calcRes(georeferencesPath)
			print(res)
		elif cmmd == 'newLoc':
			latRange = [40, 40.0005]
			lonRange = [80, 80.0005]
			rect = randomRectangle(latRange, lonRange, 100)
			print(rect)
		elif cmmd == 'quit':
			quit()
		elif cmmd == 'correct':
			corrections = True
		elif cmmd == 'clear':
			cmd('clear')
		elif cmmd == 'stitchCars':
			stitchImages(georeferencesPath, res, correction=corrections, where='Car_Detection⁩/car_predictions⁩/'+'cars_image-{}.jpg', outputPath='../Web/static/carPrediction.jpg')
		elif cmmd[0:3] == 'cmd':
			cmd(cmmd[5:-2])
		else:
			print("'{}' is not a Command".format(cmmd))

	return f.getvalue()

html_headers = open("site/headers.html", "r", encoding='utf-8').read()


def load_html(path):
	return open(path, "r", encoding='utf-8').read().replace("REPLACE_HEADERS", html_headers)

app = Flask(__name__)
Mobility(app)
"""Talisman(app, content_security_policy=None)"""

@app.route("/")
#Web Interface main endpoint
def index():
	if request.MOBILE:
		return load_html("site/mobile/index.html")
	return load_html("site/index.html")

UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set(['csv', 'png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Web Interface check for file upload
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/favicon.ico')
#Web Interface favicon
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/upload', methods=['GET', 'POST'])
#Web Interface upload files
def upload_file():
	html = load_html("site/SelectFiles.html")
	if request.method == 'POST':
		try:
			if 'file' not in request.files:
				flash('No file part')
				return redirect(request.url)
			file = request.files['file']
	        
			if file.filename == '':
				flash('No selected file')
				return redirect(request.url)
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				return redirect(url_for('upload_file', filename=filename))
		except:
			return html.replace('<span>{}</span>', '<span>{}</span>'.format("<br>".join(['<a id="filenames" href="/delete/{}">{}</a>'.format(file, file) for file in os.listdir(app.config['UPLOAD_FOLDER'])])))

	return html.replace('<span>{}</span>', '<span>{}</span>'.format("<br>".join(['<a id="filenames" href="/delete/{}">{}</a>'.format(file, file) for file in os.listdir(app.config['UPLOAD_FOLDER'])])))

@app.route('/delete/<file>')
#Web Interface delete file
def delete_file(file):
	os.remove("images/"+file)
	return redirect("/upload")

@app.route("/run/")
#Web Interface Run main routine with executeRoutine()
def run():

	def process():
		first = open("site/FirstResponse.html", "r", encoding='utf-8').read()
		yield first
		try:
			print('[INFO] Starting routine...')
			executeRoutine()
			print('[INFO] Successful')
			yield '<script type="text/javascript">window.location.replace("/done");</script>'
		except:
			pass
		return
	return Response(process())

@app.route("/done/")
#Retun results
def done():
	return open("site/SecondResponse.html", "r", encoding='utf-8').read()

@app.route('/terminal/')
#Web CLI
def interface():
	html = load_html("site/CLI.html")
	cmmd = None

	try:
		cmmd = request.args.get('cmd')
	except:
		pass

	def responseGen():

		cmd = cmmd.split(',')[-1]

		first = html.split('<span>{c}')[0] + '<span>'
		second = '</span>' + html.split('{c}</span>')[1]
		yield first
		yield '''
				<style>
				#cmmds{
					font-family: Avenir, Futura;
					font-style: normal;
					font-weight: normal;
					font-size: 33px;
					color: rgba(255,255,255,1);
					display: inline;
				}
				</style>'''
		for cmd in cmmd.split(','):
			yield '<br><p id="cmmds">Satellogic:~ RoadTrip$ {}</p>'.format(cmd)
		consoleOutput = cli(cmd)
		yield '<p id="cmmds">' + consoleOutput.replace('\n', '</p><p id="cmmds">') + '</p><br><p id="cmmds">Satellogic:~ RoadTrip$</p>'
		yield second
		return

	def new():
		first = html.split('<span>{c}')[0] + '<span>'
		second = '</span>' + html.split('{c}</span>')[1]
		yield first
		yield '''
				<style>
				#cmmds{
					font-family: Avenir, Futura;
					font-style: normal;
					font-weight: normal;
					font-size: 33px;
					color: rgba(255,255,255,1);
					display: inline;
				}
				</style>'''
		yield '<br><p id="cmmds">Satellogic:~ RoadTrip$</p>'
		yield second
		return


	if cmmd is None:
		return Response(new())
	else:
		return Response(responseGen())

@app.route("/roads")
#Web Interface display road detection
def roads():
	if request.MOBILE:
		return load_html("site/mobile/roads.html")
	if path.exists("maps/map.svg"):
		html = load_html("site/Roads.html")
		copyfile('maps/map.svg', 'static/map.svg')
		return html
	else:
		return "maps/map.svg not found"

@app.route("/cars")
#Web Interface display car detection
def cars():
	if request.MOBILE:
		return load_html("site/mobile/cars.html")
	html = load_html("site/Cars.html")
	results = carResults()
	html = html.split("<p id='car_results'>{c}")[0] + "<p id='car_results'>" + str(results) + html.split("<p id='car_results'>{c}")[1]
	return html

port = int(os.environ.get("PORT", 5000))
#Launch WSGI server for Web Interface
def server():
	WSGIServer(('', port), app).serve_forever()

#Threading for server instance
def keep_alive():  
	t = Thread(target=server)
	t.start()
	cprint("Running at ", 'green', end='')
	cprint("127.0.0.1:"+str(port), 'cyan')

if __name__ == '__main__':
	argv = sys.argv[1:]
	cmd('clear')
	cprint("RoadTrip", 'white', 'on_grey')
	cprint("Roads & Cars Detection", 'white', 'on_grey')
	if len(argv) == 0:
		keep_alive()
	else:
		if "-s" in argv:
			keep_alive()
		if "-e" in argv:
			keep_alive()
			os.system("./node_modules/.bin/electron .")
		if "-c" in argv:
			while True:
				print(cli(input(">")).replace(" <br>", ""))
		if "-r" in argv:
			executeRoutine()

