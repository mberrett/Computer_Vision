import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import cv2

# Capture video and get frame measurements
filename = "BlueBounce.mp4"
video = cv2.VideoCapture(filename)
video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# import trajectories for different filters + tracker
kalman = pd.read_csv('kalman_trajectory.csv', index_col = 0)
particle = pd.read_csv('particle_trajectory.csv', index_col = 0)
csrt = pd.read_csv('csrt_trajectory.csv', index_col = 0)

# unbounded trajectories
unbounded_kalman = pd.read_csv('unbounded_kalman.csv', index_col = 0)
unbounded_particle = pd.read_csv('unbounded_particle.csv', index_col = 0)

sns.set_style('darkgrid')

plt.plot(unbounded_kalman['x'], unbounded_kalman['y'])

plt.plot(unbounded_particle['x'], unbounded_particle['y'])

plt.title('Blue Ball Tracking Unbounded \n (89 Frames)', size = 20)
plt.xlabel('Video Width')
plt.ylabel('Video Height')
plt.legend(['Kalman Filter','Particle Filter'])
#plt.savefig('Filter_Accuracy_by_Frame.jpg')
plt.show()

sns.set_style('darkgrid')

plt.plot(unbounded_kalman['x'][9:-20], unbounded_kalman['y'][9:-20])

plt.plot(unbounded_particle['x'][9:-20], unbounded_particle['y'][9:-20])

plt.title('Blue Ball Tracking Bounded \n (60 Frames)', size = 20)
plt.xlabel('Video Width')
plt.ylabel('Video Height')
plt.legend(['Kalman Filter','Particle Filter'])
#plt.savefig('Filter_Accuracy_by_Frame.jpg')
plt.show()

def eucledian_distance(df1, df2):

    """
    Takes in two sets of coordinates and calculates the eucledian distance between them
    returns a list reflecting the eucledian point at each distance
    Used to determine how close each filter is to the ground truth.
    """

    distances = []

    for i in range(len(df1)):

        x = df1['x'][i]
        y = df1['y'][i]
        a = df2['x'][i]
        b = df2['y'][i]

        dist = math.sqrt(math.pow(np.abs(x-a),2)+math.pow(np.abs(y-b),2))
        distances.append(dist)

    return distances

#csrt['x_adjust'] = csrt['x'] + 60
sns.set_style('whitegrid')
plt.figure()
plt.axis([0,video_width,video_height, 0])
plt.title('Blue Ball Tracking', size = 20)
plt.xlabel('video width')
plt.ylabel('video height')
plt.plot(kalman['x'], kalman['y'], '--b')
plt.plot(particle['x'], particle['y'], '--g')
plt.plot(csrt['x'], csrt['y'], '-xr')
plt.legend(['Kalman Filter','Particle Filter', 'Ground Truth'])

particle_eval = eucledian_distance(csrt, particle)
kalman_eval = eucledian_distance(csrt, kalman)

print("Kalman Particle Mean Euclidean Score:", str(np.mean(kalman_eval)), "\nParticle Filter Mean Euclidean Score:", str(np.mean(particle_eval)))

plt.plot(kalman_eval)
plt.plot(particle_eval)
plt.title('Filter Accuracy by Frame')
plt.xlabel('Frame')
plt.ylabel('Eucledian Distance')
plt.legend(['Kalman Filter','Particle Filter'])
plt.show()

evalDF = pd.DataFrame({"kalman":kalman_eval, "particle":particle_eval})

sns.set_style('darkgrid')
sns.lineplot(data = evalDF)
plt.title('Filter Accuracy by Frame', size = 20)
plt.xlabel('Frame')
plt.ylabel('Eucledian Distance')
plt.legend(['Kalman Filter','Particle Filter'])
#plt.savefig('Filter_Accuracy_by_Frame.jpg')
plt.show()
