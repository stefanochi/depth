import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

class TimeDifference2d:
    def __init__(self, shape, events, poses, f):
        self.shape = shape
        self.events = events
        self.poses = poses
        self.f = f

        self.directions = self.get_directions()

        return

    def get_pose_time(self, time):
        idx = np.searchsorted(self.poses[:, 0], time)
        pose = self.poses[idx]

        return pose

    def get_directions(self):
        # Measure the time difference to find the velocity in each direction and then use arctan
        # to find the direction of the motion

        # poses = self.poses
        # dt = poses[1:, 0] - poses[:-1, 0]
        # vel = (poses[1:] - poses[:-1]) / dt[:, None]
        #
        # self.vel = vel
        #
        # directions = np.zeros((dt.size, 2))
        # directions[:,0] = poses[1:, 0]
        # directions[:,1] = np.arctan(vel[:, 2], vel[:, 1])

        gt = self.poses
        speed = np.array([gt[1:, 1] - gt[:-1, 1], gt[1:, 2] - gt[:-1, 2]]) / (gt[1:, 0] - gt[:-1, 0])
        # speed# = np.abs(speed)

        directions = np.zeros((speed.shape[1], 2))
        print(speed.shape)
        directions[:,0] = self.poses[1:, 0]
        directions[:,1] = np.mod(np.arctan2(speed[1], speed[0]), 2*np.pi)
        return directions

    def get_direction_time(self, time):
        idx = np.searchsorted(self.directions[:,0], time) + 1
        return self.directions[idx, 1]

    def compute_time_difference(self, dist=1, px_range_pred=5, delay=0.02, avg_n=5, std_mul=1, debug=False, start_delay=0.001):
        events = self.events
        poses = self.poses
        shape = self.shape

        last_time = np.full(shape, -1.0)
        td_predictions = {}
        predictions = {}
        event_sign = np.full(shape, -1)

        U = np.zeros(shape)
        V = np.zeros(shape)
        ofs = np.zeros(shape)
        final_td = np.zeros(shape)

        td_p = np.linspace(0.0, 0.1, 1000)

        count = 0
        discarded_neg = 0
        discarded_dif = 0
        discarded_q = 0
        discarded_img = 0
        discarded_ind = 0
        discarded_ang = 0
        discarded_neg = 0
        count_filtered = 0

        start_time = events[0, 0]

        diff_list = []

        ang_flow_u, ang_flow_v = self.get_angular_flow(start_time)

        for e in tqdm(events):
            x = int(e[1])
            y = int(e[2])

            # if e[3] == 0:
            #     continue

            count += 1

            last_sign = event_sign[y, x]
            last_time[y, x] = e[0]
            event_sign[y, x] = e[3]

            if e[0] < start_time + start_delay:
                continue

            u_td = (e[0] - last_time[y - dist, x]
                    if y - dist >= 0 and last_time[y - dist, x] != -1.0 and event_sign[y - dist, x] == e[3]
                    else -1.0)
            d_td = (e[0] - last_time[y + dist, x]
                    if y + dist < shape[0] and last_time[y + dist, x] != -1.0 and event_sign[y + dist, x] == e[3]
                    else -1.0)
            r_td = (e[0] - last_time[y, x + dist]
                    if x + dist < shape[1] and last_time[y, x + dist] != -1.0 and event_sign[y, x + dist] == e[3]
                    else -1.0)
            l_td = (e[0] - last_time[y, x - dist]
                    if x - dist >= 0 and last_time[y, x - dist] != -1.0 and event_sign[y, x - dist] == e[3]
                    else -1.0)

            cam_dir = self.get_direction_time(e[0])
            cam_dir = np.mod(cam_dir, 2*np.pi)
            direction = np.mod(cam_dir - np.pi, 2*np.pi)
            #direction = np.pi
            # print(np.rad2deg(direction))

            if u_td <= 0.0:
                if d_td <= 0.0:
                    v_td = 0.0
                    continue
                else:
                    v_td = d_td
            else:
                if d_td > 0:
                    #v_td = d_td if d_td >= u_td else -u_td
                    v_td = 0.0
                    discarded_ind += 1
                    continue
                else:
                    v_td = -u_td

            if r_td <= 0.0:
                if l_td <= 0.0:
                    h_td = 0.0
                    continue
                else:
                    h_td = l_td
            else:
                if l_td > 0:
                    #h_td = l_td if l_td >= r_td else -r_td
                    h_td = 0.0
                    discarded_ind += 1
                    continue
                else:
                    h_td = -r_td


            # h_td -= ang_flow_u[y, x]
            # v_td -= ang_flow_v[y, x]

            # U[y, x] = h_td
            # V[y, x] = v_td


            m = np.sqrt(np.square(v_td) + np.square(h_td))
            a = direction - np.mod(np.arctan2(v_td, h_td), np.pi * 2)
            a = np.mod(a, 2*np.pi)

            # if a > np.pi:
            #     continue

            of = m * np.cos(a)
            #of = h_td * np.cos(direction) + v_td * np.sin(direction)
            ofs[y, x] = of
            time_difference = of / dist

            if time_difference < 0.0005 or np.isclose(time_difference, 0.0):
                discarded_neg += 1
                continue

            #time_difference = td_p[np.digitize(time_difference, td_p) - 1]

            if (y, x) in td_predictions and len(td_predictions[y, x]) > avg_n:
                mean_pred = np.mean(td_predictions[y, x])
                std_pred = np.std(td_predictions[y, x])
                # time_arr_diff = np.abs(predictions[y, x] - e[0])
                # td_m = np.ma.masked_where(time_arr_diff > 0.1, td_predictions[y, x])
                # mean_pred = np.ma.mean(td_m)
                # if len(td_predictions[y, x]) - td_m.count() > 0:
                #     count_filtered += 1#len(td_predictions[y, x]) - td_m.count()
                #    # print(len(td_predictions[y, x]) - td_m.count())
                #
                # if td_m.count() < avg_n:
                #     continue

                if np.abs(mean_pred - time_difference) > std_mul:
                    last_time[y, x] = -1.0
                    event_sign[y, x] = last_sign
                    discarded_dif += 1
                    continue

                if np.isclose(np.abs(mean_pred - time_difference), 0.0):
                    continue

                if e[0] > start_time + delay:
                    final_td[y, x] = time_difference
                    #print(mean_pred)
                    # U[y, x] = np.cos(direction) * time_difference
                    # V[y, x] = np.sin(direction) * time_difference
                    U[y, x] = dist / h_td if h_td != 0.0 else 0.0
                    V[y, x] = dist / d_td if v_td != 0.0 else 0.0

                diff_list.append(std_pred)



                #td_predictions[y, x] = []
                #predictions[y, x] = []
            #a = direction

            for i in range(-int(px_range_pred / 2), int(px_range_pred / 2)):
                for k in range(-int(px_range_pred / 2), int(px_range_pred / 2)):

                    x_p = x + i
                    y_p = y + k

                    next_dist = np.sqrt(i**2 + k**2)

                    pred_time = last_time[y, x] + time_difference * next_dist

                    if x_p < 0 or x_p >= shape[1] or y_p < 0 or y_p >= shape[0]:
                        discarded_img += 1
                        continue

                    if (y_p, x_p) in predictions:
                        predictions[y_p, x_p].append(pred_time)
                        td_predictions[y_p, x_p].append(time_difference)
                    else:
                        predictions[y_p, x_p] = [pred_time]
                        td_predictions[y_p, x_p] = [time_difference]
        if debug:
            print("total: {}".format(count))
            print("diff: {}".format(discarded_dif))
            print("negative: {}".format(discarded_neg))
            print("q: {}".format(discarded_q))
            print("ang: {}".format(discarded_ang))
            print("ind: {}".format(discarded_ind))
            print("img: {}".format(discarded_img))
            print("filtered: {}".format(count_filtered))
        return final_td, td_predictions, U, V, diff_list


    def get_angular_flow(self, time):
        U = np.zeros(self.shape)
        V = np.zeros(self.shape)
        f = self.f
        w = self.get_angular_vel(time)

        for x in range(U.shape[1]):
            for y in range(U.shape[0]):
                xi = x - 90
                yi = y - 90
                # if np.mod(x, 10) != 0 or np.mod(y, 10) != 0:
                #     continue
                m = np.array([
                    [(xi * yi) / f, -(f + xi ** 2 / f), yi],
                    [f + yi ** 2 / f, -xi * yi / f, -xi]
                ])
                r = m @ w
                U[y, x] = 1 / r[0] if r[0] != 0.0 else 0.0
                V[y, x] = 1 / r[1] if r[1] != 0.0 else 0.0

        return U, V

    def get_angular_vel(self, time):
        poses = self.poses

        k = np.ones(10)
        poses[:, 1] = np.convolve(poses[:, 1], k, mode="same")
        poses[:, 2] = np.convolve(poses[:, 2], k, mode="same")
        poses[:, 3] = np.convolve(poses[:, 3], k, mode="same")
        poses[:, 4] = np.convolve(poses[:, 4], k, mode="same")
        poses[:, 5] = np.convolve(poses[:, 5], k, mode="same")
        poses[:, 6] = np.convolve(poses[:, 6], k, mode="same")
        poses[:, 7] = np.convolve(poses[:, 7], k, mode="same")

        idx = np.searchsorted(poses[:, 0], time)
        pose1 = poses[idx]
        pose2 = poses[idx + 1]

        ang1 = R.from_quat(pose1[4:]).as_euler("xyz")
        ang2 = R.from_quat(pose2[4:]).as_euler("xyz")

        vel = (ang2 - ang1) / (pose2[0] - pose1[0])

        print(vel)

        return vel