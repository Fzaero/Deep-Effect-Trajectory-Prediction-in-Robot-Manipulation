#!/usr/bin/env python

import sys

import math
import sys
import signal
import random
import matplotlib.pyplot as plt
import numpy as np
import time
from shutil import copyfile
import copy
from pprint import pprint
import pickle

directory = 'Shapes/'
sampleCount = 500
samplePoints = np.zeros((sampleCount,2))
rings = 10
wedges = 20
maxRingDist = 2.2
def getShapeContext():
   global samplePoints,sampleCount,rings,wedges,maxRingDist
   shapeContext = np.zeros((rings,wedges))
   for i in range(sampleCount):
       #print 'Point ' + str(samplePoints[i,0]) + ' ' + str(samplePoints[i,1])
       dist = math.sqrt(samplePoints[i,0]**2+samplePoints[i,1]**2)
       angle = math.degrees(math.atan2(samplePoints[i,1],samplePoints[i,0]) - math.atan2(0,1))
       if angle < 0.0:
           angle+=360.0
       r= int(math.log1p(dist)/(math.log1p(maxRingDist)/rings))
       w= int(angle/(360.0/wedges))
       if w >= wedges-1:
           w = 0
       if w == wedges/2+1:
           w = wedges/2
       if samplePoints[i,1] < 0.00001:
           if samplePoints[i,0] >= 0:
               w = 0
           else:
               w = wedges/2
       shapeContext[r,w] += 1
       #print 'Ring = ' + str(r) + ' Wedge = ' + str(w)
   return shapeContext

def dist2(p1,p2):
   return math.sqrt( (p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def sign(c1,c2):
   return 1 if c2[0]-c1[0] > 0 else -1
def readShape(gen,shapeID,edgeNum,experimentNum):
   global directory, samplePoints,sampleCount
   #print '==============================================================='
   #print directory+str(gen)+'gen/shape'+str(shapeID)+'/edge'+str(edgeNum)+'/experiment'+str(experimentNum)+'/shape.txt'
   shapeFile = open(directory+str(gen)+'gen/shape'+str(shapeID)+'/edge'+str(edgeNum)+'/experiment'+str(experimentNum)+'/shape.txt' ,'r')
   N = int(shapeFile.readline())

   corners = np.zeros((N,2))
   meanDistance = 0.0
   for i in range(N):
       cornerText = shapeFile.readline().split()
       corners[i,0] = float(cornerText[0])
       corners[i,1] = float(cornerText[1])
       meanDistance = meanDistance + math.sqrt(corners[i,0]**2+corners[i,1]**2)

   meanDistance = meanDistance/N
   for i in range(N):
       corners[i,0] = corners[i,0]/meanDistance
       corners[i,1] = corners[i,1]/meanDistance
       #print corners[i]

   samplePoints = np.zeros((sampleCount,2))

   perimeter = 0.0
   for i in range(N):
       perimeter += dist2(corners[i], corners[(i + 1) % N])
   stepSize = perimeter/sampleCount

   currentPoint = corners[0].copy()
   currentEdge = 0
   stepLength = np.array([corners[1,0] - corners[0,0], corners[1,1] - corners[0,1]])
   edgeLength = math.sqrt(stepLength[0]**2 + stepLength[1]**2)
   stepLength[1] *= stepSize/edgeLength
   stepLength[0] *= stepSize/edgeLength
   currentEdgeLength = dist2(corners[0],corners[1])    

   for i in range(sampleCount):
       currentEdgeLength -= stepSize
       samplePoints[i] = currentPoint.copy()

       nextPoint = np.array([currentPoint[0] + stepLength[0], currentPoint[1] + stepLength[1]])

       if(currentEdgeLength < 0):
           currentEdge = (currentEdge+1)%N
           stepLength = np.array([corners[(currentEdge+1)%N][0] - corners[currentEdge][0], corners[(currentEdge+1)%N][1] - corners[currentEdge][1]])
           edgeLength = math.sqrt(stepLength[0]**2 + stepLength[1]**2)
           offset = np.array([stepLength[0]*(-currentEdgeLength/edgeLength), stepLength[1]*(-currentEdgeLength/edgeLength)])
           stepLength[1] *= stepSize / edgeLength
           stepLength[0] *= stepSize / edgeLength

           currentEdgeLength = dist2(corners[currentEdge],corners[(currentEdge+1)%N]) + currentEdgeLength
           currentPoint[0] = corners[currentEdge][0] + offset[0]
           currentPoint[1] = corners[currentEdge][1] + offset[1]
       else:
           currentPoint = nextPoint.copy()
   shapeContext = getShapeContext()
   return shapeContext

index=1
from shutil import copyfile
import os.path
genIndex=3
maxGenIndex=10
f = open('supportPoints.txt', 'w')
f2 = open('shapeContexts200.txt', 'w')
f3 = open('shapeContexts800.txt', 'w')
f4 = open('shapeContexts1600.txt', 'w')
rings_wedges=[(10,20),(20,40),(40,80)]
files=[f2,f3,f4]
for i in range(genIndex,maxGenIndex+1):
    for j in range(20):
        for k in range(i):
            for l in range(2):
                print "copy "+directory+str(i)+'gen/shape'+str(j)+'/edge'+str(k)+'/experiment'+str(l)+'/trajectoryForce.txt' + \
                ' to trajectories/'+str(index)+'.txt'
                target_name=directory + str(i)+'gen/shape'+str(j)+'/edge'+str(k)+'/experiment'+str(l)+'/'
                if os.path.isfile(target_name+'supportPoints.txt'):
                    copyfile(target_name+'trajectoryForce.txt', 'trajectories/'+str(index)+'.txt')
                    copyfile(target_name+'image.jpg', 'images/'+str(index)+'.jpg')
                    with open (target_name+'supportPoints.txt', "r") as myfile:
                        data=myfile.read().replace('\n',' ').split(' ')
                        strasd=data[2]+' '+data[3]+' '+data[6]+' '+data[7]+'\n'
                    
                    f.write(strasd)
                    for f_ind in range(3):
                        rings = rings_wedges[f_ind][0]
                        wedges = rings_wedges[f_ind][1]
                        files[f_ind].write(str((readShape(i,j,k,l)[:,:wedges/2+1].reshape(rings*(wedges/2+1))/float(sampleCount)).tolist()).replace('[','').replace(']','').replace(',',' ')+'\n')
                    index=index + 1
f.close()
f2.close()
f3.close()
f4.close()

print("Dataset Created")

