# Download necessary libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Set filename
filename ="BlueBounce.mp4"

# Capture video
video = cv2.VideoCapture(filename)
video.set(1,10)

# start at a point where the ball is already entirely visibile for fair comparison with ground truth
# frame number 10 will do
#video.set(1,10)

# get video height and width to set appropriate plot axes
video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
numframes = video.get(7)

# Set Model Parameters
count = 0
history = 15
nGauss = 3
bgThresh = 0.5
noise = 30
bgs = cv2.bgsegm.createBackgroundSubtractorMOG(history,nGauss,bgThresh,noise)

ball_trajectory = np.zeros((int(numframes), 2)) - 1

while count < numframes:

    count += 1

    img2 = video.read()[1]

    try:
        cv2.imshow('Video', img2) # some times will generate error -- ignore
    except:
        pass

    # apply background subtractor
    formatting = bgs.apply(img2)

    # set threshold for black and white masking
    ret,threshold = cv2.threshold(formatting,127,255,0)

    #_ is added to account for update in OpenCV's findContours function
    # which now returns 3 rather than 2 elements

    _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    len_contours = len(contours)

    if len_contours > 0:

        # find center of the ball (contour)
        center = np.mean(contours[0], axis = 0)
        ball_trajectory[count - 1, :] = center[0]

    try:
        cv2.imshow('Foreground',formatting) # some times will generate error -- ignore
    except:
        pass
    cv2.waitKey(80)

video.release()

measured = ball_trajectory

while True:

    if measured[0,0] == -1.:
        measured = np.delete(measured, 0, 0)
    else:

        break

numMeas = measured.shape[0]

# new section explain
marked_measure = np.ma.masked_less(measured,0)

transition_matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
observation_matrix=[[1,0,0,0],[0,1,0,0]]

X_init = marked_measure[0,0]
Y_init = marked_measure[0,1]
VX_init = marked_measure[1,0] - marked_measure[0,0]
VY_init = marked_measure[1,1] - marked_measure[0,1]

init_state = [X_init, Y_init, VX_init, VY_init]

covariance_init = 1.0e-3 * np.eye(4)
covariance_transit = 1.0e-4 * np.eye(4)
covariance_observed = 1.0e-1 * np.eye(2)

kf = KalmanFilter(transition_matrices= transition_matrix,
            observation_matrices = observation_matrix,
            initial_state_mean = init_state,
            initial_state_covariance = covariance_init,
            transition_covariance = covariance_transit,
            observation_covariance = covariance_observed)

# Refine with Kalman Filter using pykalman
(filtered_state_means, filtered_state_covariances) = kf.filter(marked_measure)

kalmanDF = pd.DataFrame({'x':filtered_state_means[:,0], 'y':filtered_state_means[:,1]})
#kalmanDF.to_csv('kalman_unbounded.csv') # unbounded kalman trajectory
len(kalmanDF)

kdf = kalmanDF.iloc[9:-20,:].reset_index().iloc[:,1:]
#kdf.to_csv('kalman_trajectory.csv') # clipped kalman trajcetory
len(kdf)

plt.axis([0,video_width ,video_height,0])
plt.plot(kdf['x'], kdf['y'], '-b' ,label = 'kalman prediction')
plt.legend(loc = 1)
plt.title("Constant Velocity Kalman Filter")
plt.show()
