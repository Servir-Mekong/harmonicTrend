# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:06 2017

@author: SERVIRWK
"""

import ee
import logging
import time
import math
from pylab import *
from scipy.optimize import newton_krylov
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr

class environment(object):
    
    
    def __init__(self):
        """Initialize the environment."""   
         
        # Initialize the Earth Engine object, using the authentication credentials.
        ee.Initialize()
        self.timeString = time.strftime("%Y%m%d_%H%M%S")

        # set dates
        self.startYear = 2007;
        self.endYear = 2007;

        # construct date objects
        startDate = ee.Date.fromYMD(self.startYear,1,1)
        endDate = ee.Date.fromYMD(self.endYear,12,31)   

	# datasets
	MOD = ee.ImageCollection("MODIS/006/MYD13Q1")
	MYD = ee.ImageCollection("MODIS/006/MYD13Q1")

        # set location 
        #self.location = ee.Geometry.Polygon([[[105.294,17.923],[105.294,17.923],[106.453,17.923],[106.453,19.469],[105.2941,19.469],[105.294,17.923]]])

	#self.location = ee.Geometry.Polygon([[[103.294,17.923],[103.294,17.923],[106.453,17.923],[106.453,20.469],[103.2941,20.469],[103.294,17.923]]])
	self.location = ee.Geometry.Polygon([[[105.294,18.923],[105.294,18.923],[105.753,18.923],[105.753,19.469],[105.2941,19.469],[105.294,18.923]]])

	mod13 = ee.ImageCollection(MOD.merge(MYD)).filterDate(startDate,endDate);
	
	self.mod13 = mod13.select("EVI");
	
	# This field contains UNIX time in milliseconds.
	self.timeField = 'system:time_start';
	
	# The dependent variable we are modeling.
	self.dependent = 'EVI';

	# The number of cycles per year to model.
	self.harmonics = 3;

        # pixel size
        self.pixSize = 500
	self.scale = 2000
        
        # user ID
        self.userID = "users/servirmekong/temp/"


	# Make a list of harmonic frequencies to model. These also serve as band name suffixes.
  


class harmonicTrend():
    
    def __init__(self):
        """Initialize the app."""  
        
        # import the log library
        
        import logging
	
	# get the environment
        self.env = environment()
    
    def runModel(self):
		
	EVI = self.env.mod13.map(self.multiply);
	
	self.harmonicFrequencies = ee.List.sequence(1, self.env.harmonics);
      
        # Construct lists of names for the harmonic terms.
	self.cosNames = ee.List(self.getNames('cos_', self.env.harmonics));
	self.sinNames = ee.List(self.getNames('sin_', self.env.harmonics));
	
	## Independent variables.
	self.independents = ee.List(['constant', 't']).cat(self.cosNames).cat(self.sinNames);

	## Filter to the area of interest, mask clouds, add variables.
	harmonicLandsat = EVI.map(self.addConstant).map(self.addTime).map(self.addHarmonics);
	
	## The output of the regression reduction is a 4x1 array image.
	harmonicTrend = harmonicLandsat.select(self.independents.add(self.env.dependent)).reduce(ee.Reducer.linearRegression(self.independents.length(), 1))

	## Turn the array image into a multi-band image of coefficients.
	harmonicTrendCoefficients = harmonicTrend.select('coefficients').arrayProject([0]).arrayFlatten([self.independents])
	
	# get the lat lon and add the ndvi
	latlon = ee.Image.pixelLonLat().addBands(harmonicTrendCoefficients).unmask(-9999)
				
	#print harmonicTrendCoefficients
	latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(),geometry=self.env.location,maxPixels=5e9,scale=self.env.scale);
			
	# get data into three different arrays
	cos_0 = np.array((ee.Array(latlon.get("cos_0")).getInfo()))
	cos_1 = np.array((ee.Array(latlon.get("cos_1")).getInfo()))
	cos_2 = np.array((ee.Array(latlon.get("cos_2")).getInfo()))
	sin_0 = np.array((ee.Array(latlon.get("sin_0")).getInfo()))
	sin_1 = np.array((ee.Array(latlon.get("sin_1")).getInfo()))
	sin_2 = np.array((ee.Array(latlon.get("sin_2")).getInfo()))
	constants =  np.array((ee.Array(latlon.get('constant')).getInfo()))	
	
	lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
	lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))
	 
	# get the unique coordinates
	self.uniqueLats = np.unique(lats)
	self.uniqueLons = np.unique(lons)
	 
	# get number of columns and rows from coordinates
	self.ncols = len(self.uniqueLons)    
	self.nrows = len(self.uniqueLats)
	 
	# determine pixelsizes
	self.ys = self.uniqueLats[1] - self.uniqueLats[0] 
	self.xs = self.uniqueLons[1] - self.uniqueLons[0]
	 
	# create an array with dimensions of image
	arr = np.zeros([self.nrows, self.ncols], np.float32) #-9999
	
	# start and end date are calculate from 1970 * 2* pi
	start = int(self.env.startYear - 1970)
	stop =  int(self.env.endYear - 1970)+1
	
	constants =  np.array(constants) / (2*np.pi)
	
	values = self.calculateDates(cos_0,cos_1,cos_2,sin_0,sin_1,sin_2,constants,start,stop)
		
	# fill the array with values
	for j in range(0,1,1):
	    arr = np.zeros([self.nrows,self.ncols])
	    counter =0
	    for y in range(0,len(arr),1):
		for x in range(0,len(arr[0]),1):
		    try:
			if lats[counter] == self.uniqueLats[y] and lons[counter] == self.uniqueLons[x] and counter < len(lats)-1:
			    counter+=1
			    val = unique(np.round(values[counter],3))
			    if len(val) == 6:
				arr[len(self.uniqueLats)-1-y,x] = int((val[j]-start)*365) # we start from lower left corner
		    except:
			pass
	    
	    self.exportToGeoTiff(arr,j)
	
	# in case you want to plot the image
	import matplotlib.pyplot as plt        
	plt.imshow(arr,  vmin=0, vmax=365)
	plt.show()
	return arr 
	


    def multiply(self,img):
	""" apply scaling factor """ 	
	image = img.multiply(0.0001);
	return image.set('system:time_start',img.get('system:time_start')) ;

    def getNames(self, base, freq):
      """ Function to get a sequence of band names for harmonic terms. """
      
      myList = []
      for i in range(0,freq,1):
	  myList.append(base + str(i))
	
      return myList

    def addConstant(self,img):
	""" Function to add a constant band. """
        return img.addBands(ee.Image(1))
  
    def addTime(self,img):
	""" Function to add a time band. """
  
	# Compute time in fractional years since the epoch.
	myDate = ee.Date(img.get('system:time_start'))
	years = ee.Number(myDate.difference(ee.Date('1970-01-01'), 'year'))
	timeRadians = ee.Image(years.multiply(2.0 * math.pi))
	return img.addBands(timeRadians.rename(['t']).float());
  
  
    def addHarmonics(self,img):
	""" Function to compute the specified number of harmonics and add them as bands.  Assumes the time band is present."""
	
	for i in range(1,self.env.harmonics+1 ,1):
	    frequencies = ee.Image.constant(i)
	    # This band should represent time in radians.
	    time = ee.Image(img).select('t');
	    # Get the cosine terms.
	    cosines = time.multiply(frequencies).cos().rename([self.cosNames.get(i-1)]);
	    # Get the sin terms.
	    sines = time.multiply(frequencies).sin().rename([self.sinNames.get(i-1)]);
	    img = img.addBands(cosines).addBands(sines) 
	
	return img
      
    def fitLandsat(self,img):
	""" Compute fitted values. """
	image = img.select(self.independents).multiply(self.harmonicTrendCoefficients).reduce('sum').rename(['fitted'])
	
	return img.addBands(image)


    def calculateDates(self,cos_0,cos_1,cos_2,sin_0,sin_1,sin_2,constants,start,stop):
	""" Calculate the min and max using non-linear solver. """
	
	# empty list to store results
	values = []
	
	# range of values for non-linear solver
	myRange = np.arange(start, stop, 0.1)  
	
	counter = 0
	for x in range(0,len(cos_0)-1,1):
	    counter +=1
	    if counter > 100:
		print x, "of .. ", len(cos_0)
		counter = 0
	    # empty list with interection points
	    intersections = []
	    
	    for i in myRange:
		try:
		    # get all values
		    cos1 = cos_0[x]
		    cos2 = cos_1[x]
		    cos3 = cos_2[x]

		    sin1 = sin_0[x]
		    sin2 = sin_1[x]
		    sin3 = sin_2[x]
		    constant = constants[x]
		    
		    # function to be solved		    
		    def F(t):
			return constant-2*np.pi*cos1*np.sin(2*np.pi*t) + 2*np.pi*sin1*np.cos(2*np.pi*t) - 4*np.pi*cos2*np.sin(4*np.pi*t) + 4*np.pi*sin2*np.cos(4*np.pi*t) -6*np.pi*cos3*np.sin(6*np.pi*t) + 6*np.pi*sin3*np.cos(6*np.pi*t)
			
		    # the non-linear solver
		    sol = newton_krylov(F, i,  verbose=0)
		    
		    # if the results is within the expected boundaries
		    if sol > start and sol < stop:
			intersections.append(float(sol))  
			
		
		except:
		    pass
	    values.append(intersections)
	
	return values

    def exportToGeoTiff(self,arr,number):
	""" export the result to a geotif. """

	#SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
	transform = (np.min(self.uniqueLons),self.xs,0,np.max(self.uniqueLats),0,-self.ys)

	# set the coordinate system
	target = osr.SpatialReference()
	target.ImportFromEPSG(4326)

	# set driver
	driver = gdal.GetDriverByName('GTiff')

	timestring = time.strftime("%Y%m%d_%H%M%S")
	outputDataset = driver.Create(r"d:\mydata/" + timestring + str(number) + "test.tif", self.ncols,self.nrows, 1,gdal.GDT_Float32)
	print "exporting .. " + str("d:\mydata/" + timestring + "test.tif")

	# add some metadata
	#outputDataset.SetMetadata( {'time': str(timestring), 'someotherInfo': 'lala'} )

	outputDataset.SetGeoTransform(transform)
	outputDataset.SetProjection(target.ExportToWkt())
	outputDataset.GetRasterBand(1).WriteArray(arr)
	outputDataset.GetRasterBand(1).SetNoDataValue(-9999)
	outputDataset = None

      
    def ExportToAsset(self,img,assetName):  
        """export to asset """
        
        outputName = self.env.userID + str(self.env.timeString) + assetName
        logging.info('export image to asset: ' + str(outputName))   
	
	#img = img.multiply(10000).int16()

                    
        task_ordered = ee.batch.Export.image.toAsset(image=ee.Image(img), description=str(self.env.timeString)+"_exportJob", assetId=outputName,region=self.env.location['coordinates'], maxPixels=1e13,scale=self.env.pixSize)
        
        # start task
        task_ordered.start()

vals = harmonicTrend().runModel()
