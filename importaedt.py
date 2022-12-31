from dv import AedatFile
import numpy as np

with AedatFile("dvSave-2022_08_12_18_32_08.aedat4") as f:
    # events will be a named numpy array
    events = np.hstack([packet for packet in f['events'].numpy()])
    
    # Access information of all events by type
    timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']

    events = np.zeros((timestamps.shape[0], 4))
    timestamps = timestamps - timestamps[0]
    events[:, 0] = timestamps * 1e-6
    events[:, 1] = x
    events[:, 2] = y
    events[:, 3] = polarities

    time = []
    accelerometer = []
    gyro = []
    for imu in f["imu"]:
        time.append(imu.timestamp)
        accelerometer.append(imu.accelerometer)
        gyro.append(imu.gyroscope)

    time = np.array(time)
    time = time - time[0]
    imu_data = np.zeros((time.shape[0], 7))
    imu_data[:, 0] = time * 1e-6
    imu_data[:, 1:4] = accelerometer
    imu_data[:, 4:] = gyro

np.savetxt("events.txt", events[:])
np.savetxt("imu.txt", imu_data[:])