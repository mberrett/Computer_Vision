import cv2
import numpy as np
from deepgaze.color_detection import BackProjectionColorDetector, RangeColorDetector
from deepgaze.mask_analysis import BinaryMaskAnalyser
from deepgaze.motion_tracking import ParticleFilter
import matplotlib.pyplot as plt
import pandas as pd

# Set filename
filename = "BlueBounce.mp4"

# Set template (to identify object, i.e. blue ball)
template = cv2.imread('BlueBounceTemplate.png') #Load the image

# Capture video
video = cv2.VideoCapture(filename)
#video.set(1,10)

# Get video frame
video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

#Declaring the binary mask analyser object
my_mask_analyser = BinaryMaskAnalyser()

# Defining the deepgaze color detector object
my_back_detector = BackProjectionColorDetector()
my_back_detector.setTemplate(template)

# Filter parameters
n_particles = 500 # set number of particles

# Probability of error (a.k.a. added noise)
noise_probability = 0.015

# Spread of the particles in prediction phase
std = 25

# Initialize model within video frame
tracking_particles = ParticleFilter(video_width, video_height, n_particles)

# Had to adjust the following three deepgaze functions manually to account for new update in OpenCV 3.4.1
# (deepgaze library is built on top of OpenCV)
# wherein cv2.findCountours returned 3 values rather than 2 values
# I made the same adjustment for all three functions

def max_area_rectangle(mask, color=[0, 0, 255]):
        """it returns the rectangle sorrounding the contour with the largest area.

        This method could be useful to find a face when a skin detector filter is used.
        @param mask the binary image to use in the function
        @return get the coords of the upper corner of the rectangle (x, y) and the rectangle size (widht, hight)
            In case of error it returns a tuple (None, None, None, None)
        """
        if(mask is None): return (None, None, None, None)
        mask = np.copy(mask)
        if(len(mask.shape) == 3):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(mask, 1, 2)
        area_array = np.zeros(len(contours)) #contains the area of the contours
        counter = 0
        for cnt in contours:
                area_array[counter] = cv2.contourArea(cnt)
                counter += 1
        if(area_array.size==0):
            return (None, None, None, None) #the array is empty
        max_area_index = np.argmax(area_array) #return the index of the max_area element
        cnt = contours[max_area_index]
        (x, y, w, h) = cv2.boundingRect(cnt)
        return (x, y, w, h)

def total_contour(mask, color=[0, 0, 255]):
        """it returns the total number of contours present on the mask

        this method must be used during video analysis to check if the frame contains
        at least one contour before calling the other function below.
        @param mask the binary image to use in the function
        @return get the number of contours
        """
        if(mask is None):
            return None
        mask = np.copy(mask)
        if(len(mask.shape) == 3):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(mask, 1, 2)
        if(hierarchy is None):
            return 0
        else: return len(hierarchy)

def max_area_center(mask, color=[0, 0, 255]):
        """it returns the centre of the contour with largest area.

        This method could be useful to find the center of a face when a skin detector filter is used.
        @param mask the binary image to use in the function
        @return get the x and y center coords of the contour whit the largest area.
            In case of error it returns a tuple (None, None)
        """
        if(mask is None): return (None, None)
        mask = np.copy(mask)
        if(len(mask.shape) == 3):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(mask, 1, 2)
        area_array = np.zeros(len(contours)) #contains the area of the contours
        counter = 0
        for cnt in contours:
                #cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
                #print("Area: " + str(cv2.contourArea(cnt)))
                area_array[counter] = cv2.contourArea(cnt)
                counter += 1
        if(area_array.size==0):
            return (None, None) #the array is empty
        max_area_index = np.argmax(area_array) #return the index of the max_area element
        #Get the centre of the max_area element
        cnt = contours[max_area_index]
        M = cv2.moments(cnt) #calculate the moments
        if(M['m00'] == 0):
            return (None, None)
        cx = int(M['m10']/M['m00']) #get the center from the moments
        cy = int(M['m01']/M['m00'])
        return (cx, cy) #return the center coords

trajectory_measurements_x = []
trajectory_measurements_y = []
try:
    while(True):

        # Capture frame-by-frame
        ret, frame = video.read()
        if(frame is None): 
            break #check for empty frames

        #Return the binary mask from the backprojection algorithm
        mask = my_back_detector.returnMask(frame, morph_opening=True, blur=True, kernel_size=5, iterations=2)
        frame_mask = cv2.bitwise_not(mask) # invert mask so that it tracks ball and not background


        if(total_contour(frame_mask) > 0):
            # Use the binary mask to find the contour with largest area
            # then find the center of this contour (what we want to track)
            x_rect,y_rect,w_rect,h_rect = max_area_rectangle(frame_mask)
            x_center, y_center = max_area_center(frame_mask)

            #Add noise
            coin = np.random.uniform()
            if(coin >= 1.0-noise_probability):
                x_noise = int(np.random.uniform(-300, 300))
                y_noise = int(np.random.uniform(-300, 300))
            else:
                x_noise = 0
                y_noise = 0
            x_rect += x_noise
            y_rect += y_noise
            x_center += x_noise
            y_center += y_noise
            cv2.rectangle(frame, (x_rect,y_rect), (x_rect+w_rect,y_rect+h_rect), [255,0,0], 2) # blue rectangle

        # Predict target's position
        tracking_particles.predict(x_velocity=0, y_velocity=0, std=std)

        # Draw the particles
        tracking_particles.drawParticles(frame)

        # Estimate the object's position in the next frame
        x_estimated, y_estimated, _, _ = tracking_particles.estimate()
        cv2.circle(frame, (x_estimated, y_estimated), 3, [0,255,0], 5) # green dot

        # Save trajectory coordiantes frame by frame
        trajectory_measurements_x.append(x_estimated)
        trajectory_measurements_y.append(y_estimated)

        # Update the filter with the last measurements
        tracking_particles.update(x_center, y_center)

        #Resample the particles
        tracking_particles.resample()

        #Show the frame and wait for the exit command
        cv2.imshow('Original', frame) #show on window
        cv2.imshow('Mask', frame_mask) #show on window
        if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed
except: # sometimes thi part of the code needs to be run twice to work 
    while(True):

        # Capture frame-by-frame
        ret, frame = video.read()
        if(frame is None): 
            break #check for empty frames

        #Return the binary mask from the backprojection algorithm
        mask = my_back_detector.returnMask(frame, morph_opening=True, blur=True, kernel_size=5, iterations=2)
        frame_mask = cv2.bitwise_not(mask) # invert mask so that it tracks ball and not background


        if(total_contour(frame_mask) > 0):
            # Use the binary mask to find the contour with largest area
            # then find the center of this contour (what we want to track)
            x_rect,y_rect,w_rect,h_rect = max_area_rectangle(frame_mask)
            x_center, y_center = max_area_center(frame_mask)

            #Add noise
            coin = np.random.uniform()
            if(coin >= 1.0-noise_probability):
                x_noise = int(np.random.uniform(-300, 300))
                y_noise = int(np.random.uniform(-300, 300))
            else:
                x_noise = 0
                y_noise = 0
            x_rect += x_noise
            y_rect += y_noise
            x_center += x_noise
            y_center += y_noise
            cv2.rectangle(frame, (x_rect,y_rect), (x_rect+w_rect,y_rect+h_rect), [255,0,0], 2) # blue rectangle

        # Predict target's position
        tracking_particles.predict(x_velocity=0, y_velocity=0, std=std)

        # Draw the particles
        tracking_particles.drawParticles(frame)

        # Estimate the object's position in the next frame
        x_estimated, y_estimated, _, _ = tracking_particles.estimate()
        cv2.circle(frame, (x_estimated, y_estimated), 3, [0,255,0], 5) # green dot

        # Save trajectory coordiantes frame by frame
        trajectory_measurements_x.append(x_estimated)
        trajectory_measurements_y.append(y_estimated)

        # Update the filter with the last measurements
        tracking_particles.update(x_center, y_center)

        #Resample the particles
        tracking_particles.resample()

        #Show the frame and wait for the exit command
        cv2.imshow('Original', frame) #show on window
        cv2.imshow('Mask', frame_mask) #show on window
        if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed
    
    
#Release the camera
video.release()

# save x and y coordinates of particle filter trajectory into pandas dataframe
unbounded_particleDF = pd.DataFrame({"x":trajectory_measurements_x, "y":trajectory_measurements_y})
#unbounded_particleDF.to_csv('unbounded_particle.csv') # unbounded particle trajectory

particleDF = pd.DataFrame({"x":trajectory_measurements_x[9:-20], "y":trajectory_measurements_y[9:-20]})
len(particleDF) # 60 frames
#particleDF.to_csv('particle_trajectory.csv') # clipped particle trajectory
# save x and y coordinates of particle filter trajectory into pandas dataframe
plt.figure()
plt.axis([0,video_width,video_height, 0])
plt.plot(particleDF['x'], particleDF['y'], '-g')
