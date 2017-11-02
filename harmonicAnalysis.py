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
        self.startYear = 2005;
        self.endYear = 2005;

        # construct date objects
        startDate = ee.Date.fromYMD(self.startYear,1,1)
        endDate = ee.Date.fromYMD(self.endYear,12,31)   

	# datasets
	MOD = ee.ImageCollection("MODIS/006/MYD13Q1")
	MYD = ee.ImageCollection("MODIS/006/MYD13Q1")

        # set location 
        #self.location = ee.Geometry.Polygon([[[105.207,17.854],[105.597,17.869],[105.617,19.243],[105.257,19.241],[105.207,17.854]]])

	# Nghe an
	self.location =ee.Geometry.Polygon([[[103.294,17.923],[103.294,18.923],[106.453,17.923],[106.453,23.469],[103.2941,23.469],[103.294,17.923]]])

	
	#self.location = ee.Geometry.Polygon([[[104.8040771484375,10.428390862637604],[105.611572265625,10.201406794334043],[106.5069580078125,10.493213228263972], [106.710205078125,11.08138460241306],[104.7381591796875,11.016688524459864],[104.8040771484375,10.428390862637604]]])
	#self.location =  ee.Geometry.Polygon([[[105.12542724609375,18.651449894396634],[105.70770263671875,18.61501328321048],[105.6390380859375,19.21780295966795],[105.22430419921875,19.210022196386085],[105.12542724609375,18.651449894396634]]])
	# Vietnam
	#self.location =  ee.Geometry.Polygon([[[102.1405029,8.574163399999973],[109.4590988,8.574163399999973], [109.4590988,23.375820100000023], [102.1405029,23.375820100000023], [102.1405029,8.574163399999973]]])
	
	# cambodia
	#self.location =  ee.Geometry.Polygon([[[103.3514022,10.009719799999973],[107.63639830000001,10.009719799999973],[107.63639830000001,14.704959800000026],[103.3514022,14.704959800000026],[103.3514022,10.009719799999973]]])


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
	self.scale = 500
        
        # user ID
        self.userID = "users/servirmekong/temp/"
	
	self.exportName = ""


	# Make a list of harmonic frequencies to model. These also serve as band name suffixes.
  


class harmonicTrend():
    
    def __init__(self):
        """Initialize the app."""  
        
        # import the log library
        
        import logging
	
	# get the environment
        self.env = environment()
    
    def runModel(self,coord,row,col):
		
	print 'getting data'
	self.EVI = self.env.mod13.map(self.scaling);
	self.env.location = coord.bounds();
	
	
	self.env.exportName = "_2_" +  str(self.env.startYear) + "_" + str(row) + "_" + str(col)
	cycles = 2
	intersectionPoints, eviValues, lats, lons = self.applyHarmonics(cycles)
	
	
	
	self.exportToDrive(intersectionPoints,"days",cycles,lats,lons)	
	self.exportToDrive(eviValues,"EVI",cycles,lats,lons)	
	
	cycles = 3
	intersectionPoints, eviValues, lats, lons  = self.applyHarmonics(cycles)
	
	self.env.exportName = "_3_" +  str(self.env.startYear) + "_" + str(row) + "_" + str(col)	
	
	self.exportToDrive(intersectionPoints,"doy",cycles,lats,lons)		
	self.exportToDrive(eviValues,"EVI",cycles,lats,lons)	
	
	#return intersectionPoints


    def applyHarmonics(self,cycles):
	
	self.harmonicFrequencies = ee.List.sequence(1, cycles);
	
	self.env.harmonics = cycles
      
        # Construct lists of names for the harmonic terms.
	self.cosNames = ee.List(self.getNames('cos_', cycles));
	self.sinNames = ee.List(self.getNames('sin_', cycles));
	
	## Independent variables.
	self.independents = ee.List(['constant', 't']).cat(self.cosNames).cat(self.sinNames);
	
	## Filter to the area of interest, mask clouds, add variables.
	harmonicLandsat = self.EVI.map(self.addConstant).map(self.addTime).map(self.addHarmonics);
	
	## The output of the regression reduction is a 4x1 array image.
	harmonicTrend = harmonicLandsat.select(self.independents.add(self.env.dependent)).reduce(ee.Reducer.linearRegression(self.independents.length(), 1))

	## Turn the array image into a multi-band image of coefficients.
	harmonicTrendCoefficients = harmonicTrend.select('coefficients').arrayProject([0]).arrayFlatten([self.independents])
	
	# map with number of seasons = 
	seasonsinyear = "users/servirmekong/NumberOfSeasons/seasons_" + str(self.env.startYear )
	rsq = ee.Image(seasonsinyear).select('cycles')
	
	# get the lat lon and add the ndvi
	latlon = ee.Image.pixelLonLat().addBands(harmonicTrendCoefficients).addBands(rsq).unmask(-9999)
	
	#print harmonicTrendCoefficients
	latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(),geometry=self.env.location,maxPixels=5e9,scale=self.env.scale);
		
	# get the number of cycles
	rsqdata =  np.array((ee.Array(latlon.get('cycles')).getInfo()))

	print 'data to numpy'		
	# get data into three different arrays
	cos_0 = np.array((ee.Array(latlon.get("cos_0")).getInfo()))[rsqdata == cycles]
	cos_1 = np.array((ee.Array(latlon.get("cos_1")).getInfo()))[rsqdata == cycles]
	
	sin_0 = np.array((ee.Array(latlon.get("sin_0")).getInfo()))[rsqdata == cycles]
	sin_1 = np.array((ee.Array(latlon.get("sin_1")).getInfo()))[rsqdata == cycles]
	
	print [rsqdata == cycles]
	
	constants =  np.array((ee.Array(latlon.get('constant')).getInfo()))[rsqdata == cycles]	
	
	if cycles == 3:
	    cos_2 = np.array((ee.Array(latlon.get("cos_2")).getInfo()))[rsqdata == cycles]
	    sin_2 = np.array((ee.Array(latlon.get("sin_2")).getInfo()))[rsqdata == cycles]
		
	# get lat lon
	self.lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
	self.lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))
	
	print np.array((ee.Array(latlon.get("latitude")).getInfo()))
	 	 
	# get the unique coordinates
	self.uniqueLats = np.unique(self.lats)
	self.uniqueLons = np.unique(self.lons)
	 
	# get number of columns and rows from coordinates
	self.ncols = len(self.uniqueLons)    
	self.nrows = len(self.uniqueLats)
	 
	# determine pixelsizes
	self.ys = self.uniqueLats[1] - self.uniqueLats[0] 
	self.xs = self.uniqueLons[1] - self.uniqueLons[0]
	 
	# create an array with dimensions of image
	arr = np.zeros([self.nrows, self.ncols], np.float32) #-9999
	
	# start and end date are calculate from 1970 * 2* pi
	self.start = 0 #int(self.env.startYear - 1970)
	self.stop =  1 #int(self.env.endYear - 1970)+1

	# do two cycle analysis
	
	lats = self.lats[rsqdata == cycles]
	lons = self.lons[rsqdata == cycles]
	print lats
	
	if cycles == 2:
	    print 'calc inflectionpoints', cycles
	    intersectionPoints, eviValues = self.calculateDates2Cycles(cos_0,cos_1,sin_0,sin_1,constants)	
	    
	if cycles == 3:
	    print 'calc inflectionpoints', cycles
	    intersectionPoints, eviValues = self.calculateDates3Cycles(cos_0,cos_1,cos_2,sin_0,sin_1,sin_2,constants)
	    
	return intersectionPoints, eviValues, lats, lons



    def scaling(self,img):
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
	years = ee.Number(myDate.difference(ee.Date('2003-01-01'), 'year'))
	timeRadians = ee.Image(ee.Number(years.multiply(2.0 * math.pi)))
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


    def calculateDates2Cycles(self,cos_0,cos_1,sin_0,sin_1,constants):
	""" Calculate the min and max using non-linear solver. """
	
	# empty list to store results
	dayValues = []
	eviValues = []
	
	# range of values for non-linear solver
	myRange = np.arange(self.start, self.stop, 0.1)  

	counter = 0

	for x in range(0,len(cos_0)-1,1):
	    counter +=1
	    if counter > 2000:
		print round((float(x)/float(len(cos_0)))*100.0) , " .. %\n ", 
		counter = 0
	    
	    #empty list with interection points
	    intersections = []
	    evis = []
	    
	    for i in myRange:

		cos1 = cos_0[x]
		cos2 = cos_1[x]

		sin1 = sin_0[x]
		sin2 = sin_1[x]
		constant = constants[x]
		    
		    
		# function to be solved		    
		def derivative(t):
		 return constant-2*np.pi*cos1*np.sin(2*np.pi*t) + 2*np.pi*sin1*np.cos(2*np.pi*t) - 4*np.pi*cos2*np.sin(4*np.pi*t) + 4*np.pi*sin2*np.cos(4*np.pi*t)
		
		def func(t):
		    return constant + cos1*np.cos(2*np.pi*1*t) + sin1*np.sin(2*np.pi*1*t) + cos2*np.cos(2*np.pi*2*t) + sin2*np.sin(2*np.pi*2*t)  
			
		try:
		    # the non-linear solver
		    sol = float(newton_krylov(derivative,i)) #,  verbose=0,maxiter=100))
			    
		    # if the results is within the expected boundaries
		    if sol > self.start and sol < self.stop:
			intersections.append(sol)  
			evis.append(func(sol))
			
		
		except:
		    #print "error"
		    pass
		    
	    dayValues.append(intersections)
	    eviValues.append(evis)
	
	return dayValues, eviValues

    def calculateDates3Cycles(self,cos_0,cos_1,cos_2,sin_0,sin_1,sin_2,constants):
	""" Calculate the min and max using non-linear solver. """
	
	# empty list to store results
	dayValues = []
	eviValues = []
	
	# range of values for non-linear solver
	myRange = np.arange(self.start, self.stop, 0.1)  

	counter = 0

	for x in range(0,len(cos_0)-1,1):
	    counter +=1
	    if counter > 1000:
		print round((float(x)/float(len(cos_0)))*100.0) , " .. %\n ", 
		counter = 0
	    
	    #empty list with interection points
	    intersections = []
	    evis = []
	    
	    for i in myRange:
		cos1 = cos_0[x]
		cos2 = cos_1[x]
		cos3 = cos_2[x]

		sin1 = sin_0[x]
		sin2 = sin_1[x]
		sin3 = sin_2[x]
		constant = constants[x]
		    
	    #print cos1, cos2, sin1, sin2, constant
		    
	    # function to be solved		    
		def derivative(t):
		 return constant-2*np.pi*cos1*np.sin(2*np.pi*t) + 2*np.pi*sin1*np.cos(2*np.pi*t) - 4*np.pi*cos2*np.sin(4*np.pi*t) + 4*np.pi*sin2*np.cos(4*np.pi*t) -6*np.pi*cos3*np.sin(6*np.pi*t) + 6*np.pi*sin3*np.cos(6*np.pi*t)
		
		def func(t):
		    return constant + cos1*np.cos(2*np.pi*1*t) + sin1*np.sin(2*np.pi*1*t) + cos2*np.cos(2*np.pi*2*t) + sin2*np.sin(2*np.pi*2*t)  + cos3*np.cos(2*np.pi*3*t) + sin3*np.sin(2*np.pi*3*t)
			
		try:
		    # the non-linear solver
		    sol = float(newton_krylov(derivative,i,  verbose=0,maxiter=100))
	    
		    
		    # if the results is within the expected boundaries
		    if sol > self.start and sol < self.stop:
			intersections.append(sol)  
			evis.append(func(sol))
			
		
		except:
		    #print "error"
		    pass
		    
	    #print intersections
	    dayValues.append(intersections)
	    eviValues.append(evis)
	
	return dayValues, eviValues


    def exportToGeoTiff(self,arr,number,name):
	""" export the result to a geotif. """

	#SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
	transform = (np.min(self.uniqueLons),self.xs,0,np.max(self.uniqueLats),0,-self.ys)
	
	if name == "days":
	    arr = np.where(arr> 0,arr * 365,arr)
	
	# set the coordinate system
	target = osr.SpatialReference()
	target.ImportFromEPSG(4326)

	# set driver
	driver = gdal.GetDriverByName('GTiff')

	timestring = time.strftime("%Y%m%d_%H%M%S")
	outputDataset = driver.Create(r"d:\ate/maps/" + str(name) + self.env.exportName + "_" + str(number) + ".tif", self.ncols,self.nrows, 1,gdal.GDT_Float32)
	print "exporting .. " + str("d:\ate/maps/" + str(name) + self.env.exportName + "_" + str(number) +  ".tif")

	# add some metadata
	#outputDataset.SetMetadata( {'time': str(timestring), 'someotherInfo': 'lala'} )

	outputDataset.SetGeoTransform(transform)
	outputDataset.SetProjection(target.ExportToWkt())
	outputDataset.GetRasterBand(1).WriteArray(arr)
	outputDataset.GetRasterBand(1).SetNoDataValue(-9999)
	outputDataset = None

    def exportToDrive(self,values,name,nr,lats,lons):
	""" export the result to drive using geotiff export function. """
	
	
	# we expect six intersections
	for j in range(0,nr*2,1):
	    # make an empty array with nrows and ncols
	    
	    arr = np.zeros([self.nrows,self.ncols])-9999
	    # set counter at 0
	    counter =0
	    
	    	    # loop over the array
	    for y in range(0,len(arr),1):
		for x in range(0,len(arr[0]),1):
		    # if the lat and lon match the location in the grid
		    if lats[counter] == self.uniqueLats[y] and lons[counter] == self.uniqueLons[x] and counter < len(lats)-1:
			    
			# get unique intersection pints
			val = unique(np.round(values[counter],3))
			counter+=1
			# write to grid if we have six values.
			if len(val) == nr*2:
			    #if val[j] > -1 and  val[j] < 1:
				#print val[j]
			    arr[len(self.uniqueLats)-1-y,x] = (val[j]-self.start) # we start from lower left corner
	    # write the band to disk
	    self.exportToGeoTiff(arr,j+1,name)
	return arr

      
    def ExportToAsset(self,img,assetName):  
        """export to asset """
        
        outputName = self.env.userID + str(nr) + "_" + str(self.env.timeString) + assetName
        logging.info('export image to asset: ' + str(outputName))   
                    
        task_ordered = ee.batch.Export.image.toAsset(image=ee.Image(img), description=str(self.env.timeString)+"_exportJob", assetId=outputName,region=self.env.location['coordinates'], maxPixels=1e13,scale=self.env.pixSize)
        
        # start task
        task_ordered.start()


import ee
ee.Initialize()

xmin = 91.979254
xmax = 114.664186
ymin = 5.429208
ymax = 28.728774

n = 10
xs = (xmax - xmin) / n
ys = (ymax - ymin) / n

for i in range(0,10,1):
    for j in range(0,10,1):
	h = i
	x1 = xmin + h * xs
	x2 = x1

	x3 = xmin + (h+1) * xs
	x4 = x3
	
	print i

	v = j
	y1 = ymin + (v*ys)
	y2 = ymin + ys + (v*ys)
	y3 = y2
	y4 = y1
	geom =  ee.Geometry.Polygon( [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	#try:
	vals = harmonicTrend().runModel(geom,i,j)
	#except:
	#    pass
