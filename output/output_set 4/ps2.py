"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

import cv2.cv as cv

from math import *


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    #print radii_range
    #img_in = cv2.imread('img1.png')
    print 'checking for traffic lights'
    canny_para1 = 50
    canny_para2 = 70
    img = img_in
    img2 = img_in

    img_in2 = img_in.copy()
    minRadius1 = min(radii_range)
    maxRadius1 = max(radii_range)
    img2 = np.asarray(img_in)

    cimg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #cimg2 = cv2.GaussianBlur(cimg2, (9, 9),2)
    cimg2 = cv2.Canny(cimg2,canny_para1,canny_para2,apertureSize = 3)
    #cimg2 = cv2.medianBlur(cimg2,5)
    cimg2 = cv2.adaptiveThreshold(cimg2,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
    circles = cv2.HoughCircles(cimg2,cv.CV_HOUGH_GRADIENT,1,20,param1=50,param2=18,minRadius=10,maxRadius=30)
    light_circles = []
    center_x = 0
    center_y = 0
    flag = 0
    #print ('params: thresh1 =' + str(thresh1) + '         thresh2 = ' + str(thresh2) + '           param2 = ' + str(param21))
#    for i in circles[0,:]:
#        # draw the outer circle
#        cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
#        cv2.circle(img2,(i[0],i[1]),2,(255,255,255),3)
#    
#    cv2.imshow('hough circles', img2)
#    cv2.waitKey(0)
    
    pos_x = []  
    for i in range(len(circles[0,:])):
        
        #print 'in for loop'
        #light_x = circles[0,0:i,0] + circles[0,i+1:-1,0]
        value_x = circles[0,i,0]
        radius = circles[0,i,2]
        value_y = circles[0,i,1]
        for j in range(len(circles[0,:])):
            if abs((value_x - circles[0,j,0]) < 5) and (abs(value_y-circles[0,j,1])>=20): 
                pos_x.append(j)
                flag = flag + 1
                #print pos_x
        if len(pos_x) > 2 and flag > 2:
            center_x = int(circles[0,pos_x[0],0])
            center_y = np.median([circles[0,pos_x[0],1],circles[0,pos_x[1],1],circles[0,pos_x[2],1]])
            center = (center_x, center_y)
            #print center
            light_circles.append([circles[0,pos_x[0],:],circles[0,pos_x[1],:],circles[0,pos_x[2],:]])
            break
        else:
            (center_x, center_y) = int(circles[0,1,0]), int(circles[0,1,1])
            center = (center_x, center_y)
            
    print light_circles
        #print center_y    
        #print len(num_x)
#    else:
#        
#        #print 'yoyo2'
#        center_x = int(circles[0,1,0])
#        center_y = np.median([circles[0,0,1],circles[0,1,1],circles[0,2,1]])
#        center = (center_x, center_y)
#        #print center_y    
#        light_circles.append(circles[0,:])
        #print center
    
    if ((center_x)>0 and (center_y)>0) and flag>2:
        
        state = colors_state(img_in2, light_circles, center)
        return (center, state)
    else:
        return ((0,0), 'None')
    #print state
    
    
    

def colors_state(img_in2, light_circles, center):
    canny_para1 = 100
    canny_para2 = 70
    #cv2.imshow('img_in2',img_in2)
    #cv2.waitKey(0)
    hsv = cv2.cvtColor(img_in2, cv2.COLOR_BGR2HSV)
    
    for i in range(len(light_circles[0])):
         if (center[1] > light_circles[0][i][1]):
             red = int(light_circles[0][i][1])
             #print ('red ' + str(red))
         elif (center[1] < light_circles[0][i][1]):
             green = int(light_circles[0][i][1])
             #print ('green ' + str(green))
    center = (int(center[0]),int(center[1]))
    
    red_intensity = hsv[red,center[0],2] 
    #print "val @", (red,center[0]), " is ", red_intensity
    yellow_intensity = hsv[center[1],center[0],2] 
    #print "val @", (center[1],center[0]), " is ", yellow_intensity
    green_intensity = hsv[green,center[0],2] 
    #print "val @", (green,center[0]), " is ", green_intensity
    
    color_intensities = {"red" : red_intensity, "yellow" : yellow_intensity, "green" : green_intensity}
    
    state = max(color_intensities, key=color_intensities.get)

    return state
    
    
def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    #cv2.imshow('img_in',img_in)
    #cv2.waitKey(0)
    canny_para1 = 100
    canny_para2 = 70
    print 'checking for yield sign'
    img = img_in.copy()
    mean_y = 0
    mean_x = 0
    line_length = 0
    #print('line_using_houghP3')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,canny_para1,canny_para2,apertureSize = 3)
    cimg2 = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
    #cv2.imshow('adaptiveThreshold.jpg',cimg2)
    #cv2.waitKey(0)
    minLineLength = 30
    maxLineGap = 100
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,80)
    #lines = cv2.HoughLinesP(edges,1,np.pi,5)
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,80)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80)
    line_yield = np.zeros((1,4),dtype=np.uint32)
    
    #print len(lines[0])
    #print lines
    
    slope = np.zeros((len(lines[0]),1))
    length = np.zeros((len(lines[0]),1))
    total_lines = np.zeros((len(lines[0]),4))
    i = 0
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
        if abs(x2-x1) > 0:
            temp_slope = (y2-y1)/(x2-x1)
            slope[i,0] = temp_slope
            length[i,0] = ((x2-x1)**2 + (y2-y1)**2)**0.5
            total_lines[i,0] = x1
            total_lines[i,1] = y1
            total_lines[i,2] = x2
            total_lines[i,3] = y2
            i = i + 1
    
    print length
    print slope
    for i in range(len(length)):
        if slope[i,0] == 0:
            if length[i,0] > 50 and length[i,0] < 100:
                mean_x = 0.5*total_lines[i,0] + 0.5*total_lines[i,2]
                mean_y = 0.5*total_lines[i,1] + 0.5*total_lines[i,3]
                #print ('color strenght =' + str(sum(img[int(mean_y-20),int(mean_x),:])))
                if ((img[int(mean_y-20),int(mean_x),0]) > 80) and ((img[int(mean_y-20),int(mean_x),1]) > 80) and ((img[int(mean_y-20),int(mean_x),2]) > 80):
                    line_yield[0,0] = total_lines[i,0]
                    line_yield[0,1] = total_lines[i,1]
                    line_yield[0,2] = total_lines[i,2]
                    line_yield[0,3] = total_lines[i,3]
                    line_length = length[i,0]
                    #print ('line_yield ='  +str(line_yield))
                    cv2.line(img,(line_yield[0,0],line_yield[0,1]),(line_yield[0,2],line_yield[0,3]),(0,255,255),2)
#    

#    for x1,y1,x2,y2 in lines[0]:
#        #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
#        if (abs((y1-y2) < 5)): 
#                if (abs(x1-x2)>5 and abs(x1-x2)<150):
#                    mean_x = 0.5*x1 + 0.5*x2
#                    mean_y = 0.5*y1 + 0.5*y2
#                    print sum(img[int(mean_y-20),int(mean_x),:])
#                    if (sum(img[int(mean_y-20),int(mean_x),:]) > 520):
#                        line_yield[0,0] = x1
#                        line_yield[0,1] = y1
#                        line_yield[0,2] = x2
#                        line_yield[0,3] = y2
#                        line_length = abs(x1 - x2)
#                        print ('line_yield ='  +str(line_yield))
#                        cv2.line(img,(line_yield[0,0],line_yield[0,1]),(line_yield[0,2],line_yield[0,3]),(0,255,255),2)
#    

    #line_length = x1-x2
    #print ('line_length' + str(line_length))
    
    if mean_x > 0 and mean_y > 0 and line_length > 0:
        centroid_y = mean_y + (line_length/(2*(3**0.5)))
        centroid_y = float("{0:.2f}".format(centroid_y))
        centroid_x = mean_x
        centroid_x = float("{0:.2f}".format(centroid_x))
        return (centroid_x, centroid_y)
    else:
        return (0,0)
    
    #cv2.imshow('houghlines5.jpg',img)
    #cv2.waitKey(0)
    
       
#    cv2.imshow('img_in',img_in)
#    cv2.waitKey(0)
#    img = img_in.copy()
#    
#    
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    edges = cv2.Canny(gray,3,3,apertureSize = 3)
#    #cimg2 = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
#    #cv2.imshow('adaptiveThreshold.jpg',cimg2)
#    #cv2.waitKey(0)
#    minLineLength = 30
#    maxLineGap = 100
#    #lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
#    lines = cv2.HoughLinesP(edges,1,np.pi/180,80)
#    print len(lines[0])
#    for x1,y1,x2,y2 in lines[0]:
#        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
#
#    cv2.imshow('houghlines5.jpg',img)
#    cv2.waitKey(0)
    #return 0
    
        
    


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
        
    """
    print 'checking for stop sign'
    img = img_in.copy()
    canny_para1 = 100
    canny_para2 = 70
    #print('line_using_houghP3')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,canny_para1,canny_para2,apertureSize = 3)
    #cimg2 = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
    #cv2.imshow('adaptiveThreshold.jpg',cimg2)
    #cv2.waitKey(0)
    minLineLength = 30
    maxLineGap = 100
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,80)
    #lines = cv2.HoughLinesP(edges,1,np.pi,5)
    lines = cv2.HoughLinesP(edges,1,np.pi*90/180,25)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow('stop sign',img)
    cv2.waitKey(0)
    #print len(lines[0])
    line_length = np.zeros((len(lines[0]),1))
    slope = np.zeros((len(lines[0]),1))
    lineofinterest = np.array([[0,0,0,0]])
    t = 0
    for i in range(len(lines[0])):
        x1 = lines[0,i,0]
        y1 = lines[0,i,1]
        x2 = lines[0,i,2]
        y2 = lines[0,i,3]
        length = ((x1-x2)**2  + (y1-y2)**2)**0.5
        if (x2 - x1) > 0:
            temp_slope = (y2 - y1)/(x2 - x1)
            
        else:
            temp_slope = 100
            
        if (length > 35 and length < 45) and (temp_slope == 0 or temp_slope == 100):
            line_length[t,0] = length
            slope[t,0] = temp_slope
            if t == 0:
                lineofinterest[t,0] = x1
                lineofinterest[t,1] = y1
                lineofinterest[t,2] = x2
                lineofinterest[t,3] = y2
            else:
                lineofinterest = np.append(lineofinterest,[[x1,y1,x2,y2]],0)
            t = t + 1
    
    lineofinterest2 = np.array([[0,0,0,0]])   
    m = 0   
    flag = 0
    for i in range(np.shape(lineofinterest)[0]):
        x1 = lineofinterest[i,0]
        y1 = lineofinterest[i,1]
        x2 = lineofinterest[i,2]
        y2 = lineofinterest[i,3]
        length1 = line_length[i,0]
        for j in range(i,np.shape(lineofinterest)[0]):
            x3 = lineofinterest[j,0]
            y3 = lineofinterest[j,1]
            x4 = lineofinterest[j,2]
            y4 = lineofinterest[j,3]
            length2 = line_length[j,0]
            if slope[i,0] == slope[j,0]:    
                if abs(length1 - length2)<7:
                        mean_length = (length1 + length2)/2
                        probable_dist = mean_length + 2*mean_length*cos(radians(135/2))
                        if slope[i,0] == 0:
                            if abs(x1 - x3) < 7 and abs(x2 - x4) < 7 :
                                #print '1'
                                #print (abs(y1 - y3) - probable_dist)
                                if abs((abs(y1 - y3) - probable_dist)) < 40:
                                    #print '2'
                                    flag = 1
                                    center_x12 = (x1+x2)/2
                                    center_y12 = (y2+y4)/2
                                    if m == 0:
                                        center_X = np.array([center_x12])
                                        center_Y = np.array([center_y12])
                                    else:
                                        center_X = np.append(center_X,center_x12)
                                        center_Y = np.append(center_Y,center_y12)
                                    m = m + 1
                            elif abs(x1 - x4) < 7 and abs(x2 - x3) < 7 :
                                #print '3'
                                #print (abs(y1 - y3) - probable_dist)
                                
                                if abs((abs(y1 - y3) - probable_dist)) < 40:
                                    #print'4'
                                    flag = 1
                                    center_x12 = (x1+x2)/2
                                    center_y12 = (y1+y4)/2
                                    if m == 0:
                                        center_X = np.array([center_x12])
                                        center_Y = np.array([center_y12])
                                    else:
                                        center_X = np.append(center_X,center_x12)
                                        center_Y = np.append(center_Y,center_y12)
                                    m = m + 1
                        elif slope[i,0] == 100:
                            if abs(y1 - y3) < 7 and abs(y2 - y4) < 7 :
                                #print '5'
                                #print (abs(x1 - x3) - probable_dist)
                                if abs((abs(x1 - x3) - probable_dist)) < 40:
                                    #print '6'
                                    flag = 1
                                    center_x12 = (x1+x3)/2
                                    center_y12 = (y2+y1)/2
                                    if m == 0:
                                        center_X = np.array([center_x12])
                                        center_Y = np.array([center_y12])
                                    else:
                                        center_X = np.append(center_X,center_x12)
                                        center_Y = np.append(center_Y,center_y12)
                                    m = m + 1
                            elif abs(y1 - y4) < 7 and abs(y2 - y3) < 7 :
                                #print '7'
                                #print (abs(x1 - x3) - probable_dist)
                                if abs((abs(x1 - x3) - probable_dist)) < 40:
                                    #print '8'
                                    flag = 1
                                    center_x12 = (x1+x4)/2
                                    center_y12 = (y1+y2)/2
                                    if m == 0:
                                        center_X = np.array([center_x12])
                                        center_Y = np.array([center_y12])
                                    else:
                                        center_X = np.append(center_X,center_x12)
                                        center_Y = np.append(center_Y,center_y12)
                                    m = m + 1
                        
    
    if flag == 1:                    
        center_xfinal = np.mean(center_X)
        center_yfinal = np.mean(center_Y)
        
        center = (center_xfinal, center_yfinal)
        return center
    else:
        return (0,0)
    #print center
    
    #print ('lineofinterest =' + str(lineofinterest))
    #print ('len lineofinterest =' + str(len(lineofinterest)))
        
    #line_yield = np.zeros((1,4),dtype=np.uint32)
    #print len(lines[0])
    #print lines

#    for x1,y1,x2,y2 in lines[0]:
#        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

    #cv2.imshow('houghlines5.jpg',img)
    #cv2.waitKey(0)
    
    
    
    

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def slope_cal(lines):
    slope = []
    for x1,y1,x2,y2 in lines[0]:
        slope.append((y2-y1)/(x2-x1))
    return slope

def simplify_lines(lines):
    print ('simplify lines =' + str(len(lines[0])))
    val = 20
    for x1,y1,x2,y2 in lines[0]:
        
            
        #print 'identified x1 x2 y1 y2'
#        if (y2 - y1) > 0:
#            x1c = int((y2*x1 - y1*x2)/(y2-y1))   
        for index, (x3,y3,x4,y4) in enumerate(lines[0]):

            #print ('index =' + str(index))
            #print ('x3 =' + str(x3))
            #print ('y3 =' + str(y3))
            #print ('x4 =' + str(x4))
            #print ('y4 =' + str(y4))
            
            if (((((x3-x1)**2 + (y3-y1)**2)**0.5) < val) and ((((x4-x2)**2 + (y4-y2)**2)**0.5)<val)) or (((((x3-x2)**2 + (y3-y2)**2)**0.5) < val) and ((((x4-x1)**2 + (y4-y1)**2)**0.5)<val)):
                 #print 'true'
                 new_lines = np.delete(lines[0], index, 0)
            if ((y4 - y3) > 0) and ((y2 - y1) > 0):
                x3c = int((y4*x3 - y3*x4)/(y4-y3))      
                
                if abs(x3c - x1c) < 30:
                    new_lines = np.delete(lines[0], index, 0)
    
    #gridsize = (len(lines) - 2) / 2
    print ('new lines =' + str(len(new_lines)))
    return new_lines

def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    #cv2.imshow('img_in',img_in)
    #cv2.waitKey(0)
    canny_para1 = 100
    canny_para2 = 70
    print 'checking for warning sign'
    green_range = 150
    red_range = 150
    img = img_in.copy()
    flag = 0
    #print('line_using_houghP3')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,canny_para1,canny_para2,apertureSize = 3)
    #cv2.imshow('edges', edges)
    #cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)) 
    dilated = cv2.dilate(edges, kernel, iterations=5)
    lines = cv2.HoughLinesP(edges,1,np.pi*45/180,45)
    length = np.zeros((len(lines[0]),1))
    slope = np.zeros((len(lines[0]),1))
    lineofinterest = np.zeros((len(lines[0]),4))
    i = 0
    for x1,y1,x2,y2 in lines[0]:
         cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
    #cv2.imshow('warning', img)
    #cv2.waitKey(0)
    for x1,y1,x2,y2 in lines[0]:
         #cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
         if abs(x2 - x1) > 0:
             temp_slope = (y2-y1)/(x2-x1)
             #if temp_slope == 1 or temp_slope == -1:
             length[i,0] = ((x1 - x2) **2 + (y1 - y2)**2)**0.5
             slope[i,0] = temp_slope
             
             lineofinterest[i,:] = [x1,y1,x2,y2]

             i = i + 1
    #print lineofinterest
    
    for i in range(np.shape(lineofinterest)[0]):
        x1 = lineofinterest[i,0]
        y1 = lineofinterest[i,1]
        x2 = lineofinterest[i,2]
        y2 = lineofinterest[i,3]
        length1 = length[i,0]
        for j in range(i,np.shape(lineofinterest)[0]):
            x3 = lineofinterest[j,0]
            y3 = lineofinterest[j,1]
            x4 = lineofinterest[j,2]
            y4 = lineofinterest[j,3]
            length2 = length[j,0]
            m =0
            if slope[i,0] == slope[j,0]: 
                if slope[i,0] == 1:
                    if abs(length1 - length2)<7:
                            mean_length = (length1 + length2)/2
                            probable_dist = mean_length * (2**0.5)
                            if abs(x1-x4)< 4 and abs(y2-y3)<4:
                                distance = ((x1-x4)**2 + (y1-y4)**2)**0.5
                                if abs(distance - probable_dist) < 14:
                                    
                                    center_x12 = (x1+x4)/2
                                    center_y12 = (y1+y4)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    if img_G > green_range and img_R > red_range:
                                        flag = 1
                                        if m == 0:
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
                            elif abs(x1-x3)< 4 and abs(y2-y4)<4:
                                distance = ((x1-x3)**2 + (y1-y3)**2)**0.5
                                if abs(distance - probable_dist) < 10:
                                    
                                    center_x12 = (x1+x3)/2
                                    center_y12 = (y1+y3)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    if img_G > green_range and img_R > red_range:
                                        flag = 1
                                        if m == 0:
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
                if slope[i,0] == -1:
                    if abs(length1 - length2)<7:
                            mean_length = (length1 + length2)/2
                            probable_dist = mean_length * (2**0.5)
                            if abs(x1-x4)< 4 and abs(y2-y3)<4:
                                distance = ((x1-x4)**2 + (y1-y4)**2)**0.5
                                if abs(distance - probable_dist) < 14:
                                    
                                    center_x12 = (x1+x4)/2
                                    center_y12 = (y1+y4)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    if img_G > green_range and img_R > red_range:
                                        if m == 0:
                                            flag = 1
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
                            elif abs(x1-x3)< 4 and abs(y2-y4)<4:
                                distance = ((x1-x3)**2 + (y1-y3)**2)**0.5
                                if abs(distance - probable_dist) < 10:
                                    
                                    center_x12 = (x1+x3)/2
                                    center_y12 = (y1+y3)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    if img_G > green_range and img_R > red_range:
                                        flag = 1
                                        if m == 0:
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
    
    if flag == 1:
        print 'shape idenfitied'
        center_xfinal = np.mean(center_X)
        center_yfinal = np.mean(center_Y)
        img_B = img_in[int(center_yfinal),int(center_xfinal),0]
        img_G = img_in[int(center_yfinal),int(center_xfinal),1]
        img_R = img_in[int(center_yfinal),int(center_xfinal),2]
    
        if img_G > green_range and img_R > red_range:
            center = (center_xfinal, center_yfinal)
            return center
        else:
            return (0,0)
    else:
        return (0,0)
            
    
    for x1,y1,x2,y2 in lines[0,:]:
         cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
#    #print lineofinterest
#    #cv2.imshow('houghlines5.jpg',img)
#    #cv2.waitKey(0)
#    
#    
#    img_B = img_in[int(center_yfinal),int(center_xfinal),0]
#    img_G = img_in[int(center_yfinal),int(center_xfinal),1]
#    img_R = img_in[int(center_yfinal),int(center_xfinal),2]
#    
#    if img_G > 100 and img_R > 100:
#        center = (center_xfinal, center_yfinal)
    
# =============================================================================
#     
# # =============================================================================
# #     lines = cv2.HoughLines(edges,1,np.pi/180,30)
# #     print ('original lines = ' + str(len(lines[0])))
# #     #print lines
# #     for rho,theta in lines[0]:
# #         #print 'loop1'
# #         a = np.cos(theta)
# #         b = np.sin(theta)
# #         x0 = a*rho
# #         y0 = b*rho
# #         x1 = int(x0 + 1000*(-b))
# #         y1 = int(y0 + 1000*(a))
# #         x2 = int(x0 - 1000*(-b))
# #         y2 = int(y0 - 1000*(a))
# #         for index, (rho1,theta1) in enumerate(lines[0]):
# #             #print 'loop2'
# #             a1 = np.cos(theta1)
# #             b1 = np.sin(theta1)
# #             x01 = a1*rho1
# #             y01 = b1*rho1
# #             x3 = int(x01 + 1000*(-b1))
# #             y3 = int(y01 + 1000*(a1))
# #             x4 = int(x01 - 1000*(-b1))
# #             y4 = int(y01 - 1000*(a1))
# #             print ('hi1')  
# #             if (((((x3-x1)**2 + (y3-y1)**2)**0.5) < 15) and ((((x4-x2)**2 + (y4-y2)**2)**0.5)<15)) or (((((x3-x2)**2 + (y3-y2)**2)**0.5) < 15) and ((((x4-x1)**2 + (y4-y1)**2)**0.5)<15)):
# #                 print ('hi2')  
# #                 del lines[index]
# #     
# #           
# #     print ('new lines = ' + str(len(lines[0])))
# #     for rho,theta in lines[0]:
# #         print 'loop1'
# #         a = np.cos(theta)
# #         b = np.sin(theta)
# #         x0 = a*rho
# #         y0 = b*rho
# #         x1 = int(x0 + 1000*(-b))
# #         y1 = int(y0 + 1000*(a))
# #         x2 = int(x0 - 1000*(-b))
# #         y2 = int(y0 - 1000*(a))
# #         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# # =============================================================================
#     minLineLength = 30
#     maxLineGap = 100
#     lines = cv2.HoughLinesP(dilated,1,np.pi/180,60)
#     #lines = cv2.HoughLinesP(edges,20,np.pi*455/180,50,maxLineGap=1)
#     
#     #print len(lines[0])
#     #print len(lines[0])
#     #print lines
#     #lines = simplify_lines(lines)
#     slope = slope_cal(lines)
#     #print slope
#     #print ('slope len =' + str(len(slope)))
#     
#     count45 = slope.count(1)
#     count135 = slope.count(-1)
#     line_len_45 = np.zeros((count45,1))
#     line_45 = []
#     line_len_135 = np.zeros((count135,1))
#     line_135 = []
#     t45 = 0
#     t135 = 0
#     t = 0
#     for x1,y1,x2,y2 in lines[0]:
#         if slope[t] == 1:
#             line_len_45[t45,0] = ((x1 - x2) **2 + (y1 - y2)**2)**0.5
#             line_45.append(lines[0,t])
#             t45 = t45 + 1
#         if slope[t] == -1:
#             line_len_135[t135,0] = ((x1 - x2) **2 + (y1 - y2)**2)**0.5
#             line_135.append(lines[0,t])
#             t135 = t135 + 1
#         t = t + 1   
#         
#     
#     line_mean_45 = np.mean(line_len_45)
#     line_mean_135 = np.mean(line_len_135)
#     #print ('line_45 =' + str(line_45))
#     #print ('line_len_45 =' + str(line_len_45))
#     #print ('mean line_len_45 =' + str(np.mean(line_len_45)))
#     
#     #print ('line_135 =' + str(line_135))
#     #print ('line_len_135 =' + str(line_len_135))
#     #print ('mean line_len_135 =' + str(np.mean(line_len_135)))
#     
#     shortlisted_lines_45 = []
#     shortlisted_lines_135 = []
# #    for i in range(len(line_45)):
# #        if ((line_len_45[i,0]) >= line_mean_45) and ((line_len_45[i,0]) < line_mean_45+15):
# #            shortlisted_lines_45.append(line_45[i])
# #    
# #    for i in range(len(line_135)):
# #        if ((line_len_135[i,0]) >= line_mean_135) and ((line_len_135[i,0]) < line_mean_135+15):
# #            shortlisted_lines_45.append(line_135[i])
#     center = (0,0)
#     for i in range(len(line_45)):
#         x1 = line_45[i][0]
#         y1 = line_45[i][1]
#         x2 = line_45[i][2]
#         y2 = line_45[i][3]
#         for j in range(len(line_135)):
#             x3 = line_135[j][0]
#             y3 = line_135[j][1]
#             x4 = line_135[j][2]
#             y4 = line_135[j][3]
#             
#             if (abs(line_len_45[i,0]  - line_len_135[j,0])) < 3:
#                 mean_distance = (line_len_45[i,0]*0.5 +  line_len_135[j,0] * 0.5)
#                 #print '1'
#                 if ((abs(x1-x3)<4) and (abs(y1-y3)<4)) or ((abs(x2-x4)<4) and (abs(y2-y4)<4)):
#                     #print '2'
#                     diagonal_dist = ((x1-x3)**2 + (x2-x4)**2 + (y1-y3)**2 + (y2-y4)**2)**0.5
#                     if (diagonal_dist - 1.414*mean_distance)<4:
#                         #print '3'
#                         if ((abs(x1-x3)<4) and (abs(y1-y3)<4)):
#                             center = (0.5*x2 + 0.5*x4 - 3,0.5*y2 + 0.5*y4 - 3) 
#                         else:
#                             center = (0.5*x1 + 0.5*x3 - 3,0.5*y1 + 0.5*y3 - 3) 
#                         #center = (((abs(x1-x3)<4) and (abs(y1-y3)<4)) * (0.5*x2 + 0.5*x4,0.5*y2 + 0.5*y4)) + (((abs(x2-x4)<4) and (abs(y2-y4)<4))* (0.5*x1 + 0.5*x3,0.5*y1 + 0.5*y3))
#                         #print center
#                 elif ((abs(x1-x4)<4) and (abs(y1-y4)<4)) or ((abs(x2-x3)<4) and (abs(y2-y3)<4)):
#                     #print '4'
#                     diagonal_dist = ((x1-x4)**2 + (x2-x3)**2 + (y1-y4)**2 + (y2-y3)**2)**0.5
#                     if (diagonal_dist - 1.414*mean_distance)<4:
#                         #print '5'
#                         if ((abs(x1-x4)<4) and (abs(y1-y4)<4)):
#                             center = (0.5*x2 + 0.5*x3 - 3,0.5*y2 + 0.5*y3 - 3) 
#                         else:
#                             center = (0.5*x1 + 0.5*x4 - 3,0.5*y1 + 0.5*y4 - 3) 
#                         #center = (((abs(x1-x4)<4) and (abs(y1-y4)<4)) * (0.5*x2 + 0.5*x3,0.5*y2 + 0.5*y3)) + (((abs(x2-x3)<4) and (abs(y2-y3)<4))* (0.5*x1 + 0.5*x4,0.5*y1 + 0.5*y4))
#                         #print center
# #    
#     
#     #print ('center =' +str(center))
#     new_line_len_45 = remove_duplicate_lines(line_45)
#     new_line_len_135 = remove_duplicate_lines(line_135)
#     
#     #print ('new_line_len_45 =' + str(new_line_len_45))
#     #print ('new_line_len_135 =' + str(new_line_len_135))
#     
#     for x1,y1,x2,y2 in lines[0]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
#     
#     
#     #cv2.imshow('houghlines5.jpg',img)
#     #cv2.waitKey(0)
# =============================================================================
    #return center

def remove_duplicate_lines(lines):
    
    new_lines = lines
    for i in range(len(lines)-2):
        x1 = lines[i][0]
        y1 = lines[i][1]
        x2 = lines[i][2]
        y2 = lines[i][3]
        for j in range(i+1,(len(lines)-1)):
            
            x3 = lines[j][0]
            y3 = lines[j][1]
            x4 = lines[j][2]
            y4 = lines[j][3]
            if (abs(x1 - x3) < 6) and ((abs(y1 - y3) < 6)) and (abs(x2 - x4) < 6) and (abs(y2 - y4) < 6):
                new_lines = new_lines
            elif (abs(x1 - x4) < 6) and ((abs(y1 - y4) < 6)) and (abs(x2 - x3) < 6) and (abs(y2 - y3) < 6):
                lines.pop(j)
    return lines
                

def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    print 'checking for construction sign'
    canny_para1 = 100
    canny_para2 = 70
    img = img_in.copy()
    flag = 0
    green_range = 160
    red_range = 150
    #print('line_using_houghP3')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,canny_para1,canny_para2,apertureSize = 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)) 
    dilated = cv2.dilate(edges, kernel, iterations=5)
    #lines = cv2.HoughLinesP(edges,1,np.pi*45/180,23)
    lines = cv2.HoughLinesP(edges,1,np.pi*45/180,45)
    length = np.zeros((len(lines[0]),1))
    slope = np.zeros((len(lines[0]),1))
    lineofinterest = np.zeros((len(lines[0]),4))
    i = 0
    
#    for x1,y1,x2,y2 in lines[0]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
#    cv2.imshow('construction.jpg',img)
#    cv2.waitKey(0)
    for x1,y1,x2,y2 in lines[0]:
         #cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
         
         if abs(x2 - x1) > 0:
             temp_slope = (y2-y1)/(x2-x1)
             #if temp_slope == 1 or temp_slope == -1:
             length[i,0] = ((x1 - x2) **2 + (y1 - y2)**2)**0.5
             slope[i,0] = temp_slope
             
             lineofinterest[i,:] = [x1,y1,x2,y2]

             i = i + 1
    #print lineofinterest
    
    for i in range(np.shape(lineofinterest)[0]):
        x1 = lineofinterest[i,0]
        y1 = lineofinterest[i,1]
        x2 = lineofinterest[i,2]
        y2 = lineofinterest[i,3]
        length1 = length[i,0]
        for j in range(i,np.shape(lineofinterest)[0]):
            x3 = lineofinterest[j,0]
            y3 = lineofinterest[j,1]
            x4 = lineofinterest[j,2]
            y4 = lineofinterest[j,3]
            length2 = length[j,0]
            m =0
            if slope[i,0] == slope[j,0]: 
                if slope[i,0] == 1:
                    if abs(length1 - length2)<7:
                            mean_length = (length1 + length2)/2
                            probable_dist = mean_length * (2**0.5)
                            if abs(x1-x4)< 4 and abs(y2-y3)<4:
                                distance = ((x1-x4)**2 + (y1-y4)**2)**0.5
                                if abs(distance - probable_dist) < 14:
                                    
                                    center_x12 = (x1+x4)/2
                                    center_y12 = (y1+y4)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    print('Construction Blue = ' + str(img_B))
                                    print('Construction Green = ' + str(img_G))
                                    print('Construction Red = ' + str(img_R))
                                    if img_R > red_range and img_G < green_range and img_G > 40:
                                        flag = 1
                                        if m == 0:
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
                            elif abs(x1-x3)< 4 and abs(y2-y4)<4:
                                distance = ((x1-x3)**2 + (y1-y3)**2)**0.5
                                if abs(distance - probable_dist) < 10:
                                    
                                    center_x12 = (x1+x3)/2
                                    center_y12 = (y1+y3)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    print('Construction Blue = ' + str(img_B))
                                    print('Construction Green = ' + str(img_G))
                                    print('Construction Red = ' + str(img_R))
                                    if img_R > red_range and img_G < green_range and img_G > 40:
                                        flag = 1
                                        if m == 0:
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
                if slope[i,0] == -1:
                    if abs(length1 - length2)<7:
                            mean_length = (length1 + length2)/2
                            probable_dist = mean_length * (2**0.5)
                            if abs(x1-x4)< 4 and abs(y2-y3)<4:
                                distance = ((x1-x4)**2 + (y1-y4)**2)**0.5
                                if abs(distance - probable_dist) < 14:
                                    
                                    center_x12 = (x1+x4)/2
                                    center_y12 = (y1+y4)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    print('Construction Blue = ' + str(img_B))
                                    print('Construction Green = ' + str(img_G))
                                    print('Construction Red = ' + str(img_R))
                                    if img_R > red_range and img_G < green_range and img_G > 40:
                                        if m == 0:
                                            flag = 1
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
                            elif abs(x1-x3)< 4 and abs(y2-y4)<4:
                                distance = ((x1-x3)**2 + (y1-y3)**2)**0.5
                                if abs(distance - probable_dist) < 10:
                                    
                                    center_x12 = (x1+x3)/2
                                    center_y12 = (y1+y3)/2
                                    img_B = img_in[int(center_y12),int(center_x12),0]
                                    img_G = img_in[int(center_y12),int(center_x12),1]
                                    img_R = img_in[int(center_y12),int(center_x12),2]
                                    print('Construction Blue = ' + str(img_B))
                                    print('Construction Green = ' + str(img_G))
                                    print('Construction Red = ' + str(img_R))
                                    if img_R > red_range and img_G < green_range and img_G > 40:
                                        flag = 1
                                        if m == 0:
                                            center_X = np.array([center_x12])
                                            center_Y = np.array([center_y12])
                                        else:
                                            center_X = np.append(center_X,center_x12)
                                            center_Y = np.append(center_Y,center_y12)
                                        m = m + 1
    if flag == 1:
        print 'shape idenfitied'
        center_xfinal = np.mean(center_X)
        center_yfinal = np.mean(center_Y)
        img_B = img_in[int(center_yfinal),int(center_xfinal),0]
        img_G = img_in[int(center_yfinal),int(center_xfinal),1]
        img_R = img_in[int(center_yfinal),int(center_xfinal),2]
        
        if  img_R > red_range and img_G < green_range:
            center = (center_xfinal, center_yfinal)
            return center
        else:
            return (0,0)
    else:
        return (0,0)
    #print 'green'
    #print img_G
    

    
#    img_B = img_in[int(center_yfinal),int(center_xfinal),0]
#    img_G = img_in[int(center_yfinal),int(center_xfinal),1]
#    img_R = img_in[int(center_yfinal),int(center_xfinal),2]
#    
#    #print 'green'
#    #print img_G
#    if  img_R > 180 and img_G < 150:
#        center = (center_xfinal, center_yfinal)
#    
#    
#    

#    return center


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    print 'checking for Do not Enter sign'
    canny_para1 = 100
    canny_para2 = 70
    img = img_in
    img2 = img_in
    flag = 0
    #cv2.imshow('img_in',img_in)
    #cv2.waitKey(0)
    img_in2 = img_in.copy()
    
    img2 = np.asarray(img_in)

    cimg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #cimg2 = cv2.GaussianBlur(cimg2, (9, 9),2)
    cimg2 = cv2.Canny(cimg2,canny_para1,canny_para2,apertureSize = 3)
    #cimg2 = cv2.medianBlur(cimg2,5)ok
    cimg2 = cv2.adaptiveThreshold(cimg2,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
    #cv2.imshow('cimg2',cimg2)
    #cv2.waitKey(0)
    circles = cv2.HoughCircles(cimg2,cv.CV_HOUGH_GRADIENT,1,20,param1=30,param2=18,minRadius=10,maxRadius=42)
    
    #print ('params: thresh1 =' + str(thresh1) + '         thresh2 = ' + str(thresh2) + '           param2 = ' + str(param21))
#    for i in circles[0,:]:
#        # draw the outer circle
#        cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
        #cv2.circle(img2,(i[0],i[1]),2,(255,255,255),3)
    
    for i in circles[0,:]:
        if (img[int(i[1]),int(i[0]),0] > 180) and (img[int(i[1]),int(i[0]),1] > 180) and (img[int(i[1]),int(i[0]),2] > 180):
            if (img[int(i[1]),int(i[0])-5,0] > 180) and (img[int(i[1]),int(i[0])-5,1] > 180) and (img[int(i[1]),int(i[0])-5,2] > 180):
                if (img[int(i[1])+15,int(i[0])-5,0] < 180) and (img[int(i[1])+15,int(i[0])-5,1] < 180) and (img[int(i[1])+15,int(i[0])-5,2] > 180):
                
                    flag = 1
                    center = (i[0],i[1])
    #cv2.imshow('img2',img)
    #cv2.waitKey(0)
    #print circles
    if flag == 1:
        return center
    else:
        return (0,0)


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    signs_dict = {}
    #cv2.imshow('img_in', img_in)
    #cv2.waitKey(0)
    radii_range = [10,40]
    (center, state) = traffic_light_detection(img_in, radii_range)
    if center[0] > 0 and center[1] > 0 :
        signs_dict.update({'traffic_light': (center, state)})
    else:
        print 'Traffic Light not found'
    
    center_yield = yield_sign_detection(img_in)
    if center_yield[0] > 0 and center_yield[1] > 0 :
        signs_dict.update({'yield': center_yield})
    else:
        print 'Yield sign not found'
        
        
    center_stop = stop_sign_detection(img_in)
    if center_stop[0] > 0 and center_stop[1] > 0 :
        signs_dict.update({'stop': center_stop})
    else:
        print 'Stop sign not found'
        
    center_warn = warning_sign_detection(img_in)
    if center_warn[0] > 0 and center_warn[1] > 0 :
        signs_dict.update({'warning': center_warn})
    else:
        print 'Warning sign not found'
        
    center_cons = construction_sign_detection(img_in)
    if center_cons[0] > 0 and center_cons[1] > 0 :
        signs_dict.update({'construction': center_cons})
    else:
        print 'Construction sign not found'
    
    
    center_dne = do_not_enter_sign_detection(img_in)
    if center_dne[0] > 0 and center_dne[1] > 0 :
        signs_dict.update({'no_entry': center_dne})
    else:
        print 'DNE sign not found'
    
    return signs_dict


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    
    img2 = img_in.copy()
    signs_dict = {}
    img_in = cv2.bilateralFilter(img_in,9,100,120)
    cv2.imshow('img_in', img_in)
    cv2.waitKey(0)
    radii_range = [10,40]
    (center, state) = traffic_light_detection(img_in, radii_range)
    if center[0] > 0 and center[1] > 0 :
        signs_dict.update({'traffic_light': (center, state)})
    else:
        print 'Traffic Light not found'
    
    center_yield = yield_sign_detection(img_in)
    if center_yield[0] > 0 and center_yield[1] > 0 :
        signs_dict.update({'yield': center_yield})
    else:
        print 'Yield sign not found'
        
        
    center_stop = stop_sign_detection(img_in)
    if center_stop[0] > 0 and center_stop[1] > 0 :
        signs_dict.update({'stop': center_stop})
    else:
        print 'Stop sign not found'
        
    center_warn = warning_sign_detection(img_in)
    if center_warn[0] > 0 and center_warn[1] > 0 :
        signs_dict.update({'warning': center_warn})
    else:
        print 'Warning sign not found'
        
    center_cons = construction_sign_detection(img_in)
    if center_cons[0] > 0 and center_cons[1] > 0 :
        signs_dict.update({'construction': center_cons})
    else:
        print 'Construction sign not found'
    
    
    center_dne = do_not_enter_sign_detection(img_in)
    if center_dne[0] > 0 and center_dne[1] > 0 :
        signs_dict.update({'no_entry': center_dne})
    else:
        print 'DNE sign not found'
    
    return signs_dict


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError
