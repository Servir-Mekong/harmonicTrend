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

class environment(object):
    
    
    def __init__(self):
        """Initialize the environment."""   
         
        # Initialize the Earth Engine object, using the authentication credentials.
        ee.Initialize()
        self.timeString = time.strftime("%Y%m%d_%H%M%S")

        # set dates
        self.startYear = 2003;
        self.endYear = 2003;

        # construct date objects
        startDate = ee.Date.fromYMD(self.startYear,1,1)
        endDate = ee.Date.fromYMD(self.endYear,12,31)   

	# datasets
	MOD = ee.ImageCollection("MODIS/006/MYD13Q1")
	MYD = ee.ImageCollection("MODIS/006/MYD13Q1")

        # set location 
        self.location = ee.Geometry.Polygon([[[103.294,17.923],[103.294,17.923],[106.453,17.923],[106.453,20.469],[103.2941,20.469],[103.294,17.923]]])
 

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
	harmonicTrendCoefficients = harmonicTrend.select('coefficients').arrayProject([0]).arrayFlatten([self.independents]).clip(self.env.location)
	#print harmonicTrendCoefficients
	raw_list = harmonicTrendCoefficients.reduceRegion(reducer=ee.Reducer.toList(),geometry=self.env.location,maxPixels=5e9,scale=500);
	
	start = 33 #*2*np.pi
	stop =  34 #*2*np.pi
	t = np.arange(start, stop, 0.005)  
	i = 20
	cos1 = ee.Array(raw_list.get('cos_0')).getInfo()[i]
	cos2 = ee.Array(raw_list.get('cos_1')).getInfo()[i]
	cos3 = ee.Array(raw_list.get('cos_2')).getInfo()[i]

	sin1 = ee.Array(raw_list.get('sin_0')).getInfo()[i]
	sin2 = ee.Array(raw_list.get('sin_1')).getInfo()[i]
	sin3 = ee.Array(raw_list.get('sin_2')).getInfo()[i]
	
	constant = ee.Array(raw_list.get('constant')).getInfo()[i] / (2*np.pi)
	print constant

	#print len(cos1),len(cos2), len(cos3), len(sin1),len(sin2), len(sin3), len(constant)
	
	w = 1                # number of cycles
	xt1 = cos1*np.cos(2*np.pi*w*t) + sin1*np.sin(2*np.pi*w*t)
	w = 2                # number of cycles
	xt2 = cos2*np.cos(2*np.pi*w*t) + sin2*np.sin(2*np.pi*w*t) 
	w = 3                # number of cycles
	xt3 = cos3*np.cos(2*np.pi*w*t) + sin3*np.sin(2*np.pi*w*t)
	
	
	xt =  constant*t +xt1+xt2+xt3
	
	DT1 = -2*np.pi*cos1*np.sin(2*np.pi*t) + 2*np.pi*sin1*np.cos(2*np.pi*t)
	DT2 = -4*np.pi*cos2*np.sin(4*np.pi*t) + 4*np.pi*sin2*np.cos(4*np.pi*t)
	DT3 = -6*np.pi*cos3*np.sin(6*np.pi*t) + 6*np.pi*sin3*np.cos(6*np.pi*t)


	dtdx =  constant-DT1 + DT2 + DT3
	
	fig, ax1 = plt.subplots()
	ax1.plot(t, xt, 'b-')

	ax2 = ax1.twinx()
	s2 = np.sin(2 * np.pi * t)
	ax2.plot(t, dtdx, 'r-')

	fig.tight_layout()
	plt.axhline(y=0, color='r', linestyle='-')

	def F(t):
	    #y = np.sin(np.pi*t)
	    #return np.power(y,2) - 1 + np.power(((b1+2*np.pi*y*(12*b6*np.power(y,2)-4*b5-y-b2+2*b5-9*b6)) / (2*np.pi*(b3-4*b4*y +3*b7*(1-4*np.power(y,2))))),2)
	    return constant-2*np.pi*cos1*np.sin(2*np.pi*t) + 2*np.pi*sin1*np.cos(2*np.pi*t)  - 4*np.pi*cos2*np.sin(4*np.pi*t) + 4*np.pi*sin2*np.cos(4*np.pi*t) -6*np.pi*cos3*np.sin(6*np.pi*t) + 6*np.pi*sin3*np.cos(6*np.pi*t)

	#"""
	intersections = []
	myRange = np.arange(start, stop, 0.1)    
	for i in myRange:
	    try:
		sol = newton_krylov(F, i,  verbose=1)
		if sol > start and sol < stop:
		    #print sol
		    plt.axvline(x=sol, color='k', linestyle='--')
		    intersections.append(float(sol))  
		    #print sol	
	    except:
		pass
	#"""		
	plt.show()
	
	
	#df = DataFrame(band1, columns = pts[0])
	#print df
	#outputName = "Harmonic_trend_" + str(self.env.startYear)
	#self.ExportToAsset(harmonicTrendCoefficients,outputName)         
	
	##fittedHarmonic = harmonicLandsat.map(self.fitLandsat)
	## Compute the number of observations in each pixel.
	##n = fittedHarmonic.select("EVI").count();
  
	## There are n-p degrees of freedom
	##dof = n.subtract(self.independents.length());


	
	#print self.harmonicTrendCoefficients
	
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
      
    def ExportToAsset(self,img,assetName):  
        """export to asset """
        
        outputName = self.env.userID + str(self.env.timeString) + assetName
        logging.info('export image to asset: ' + str(outputName))   
	
	#img = img.multiply(10000).int16()

                    
        task_ordered = ee.batch.Export.image.toAsset(image=ee.Image(img), description=str(self.env.timeString)+"_exportJob", assetId=outputName,region=self.env.location['coordinates'], maxPixels=1e13,scale=self.env.pixSize)
        
        # start task
        task_ordered.start()
harmonicTrend().runModel()
