import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib
import matplotlib.cm as cm

def events2img(events, img_shape, polarity=0, filter_polarity=True):
    img = np.zeros(img_shape)

    for e in events:
        if filter_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        img[y, x] += 1

    return img

def events2time_surf(events, img_shape, polarity=0, filter_polarity=True):
    img = np.zeros(img_shape)

    for e in events:
        if filter_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        img[y, x] = e[0]

    return img


def time_surface(events, img_shape, polarity=0, filter_polarity=True):
    img = np.zeros(img_shape)

    for e in events:
        if filter_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        img[y, x] = e[0]

    return img

def _image2pointcloud(d, v_range=None):
    points = []
    for x in range(d.shape[1]):
        for y in range(d.shape[0]):
            if d.mask[y, x]:
                continue
            if v_range != None:
                if v_range[0] > d.data[y, x] or d.data[y, x] > v_range[1]:
                    continue
            points.append([x, y, d.data[y, x]])
    return np.array(points)

def _set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def visualize3d(depth_map, s=10, range=None):
    points = _image2pointcloud(depth_map, range)

    points = np.array(points)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    if range != None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s, c=points[:, 2], cmap="jet_r", vmin=range[0], vmax=range[1])
        ax.set_zlim3d([range[0], range[1]])
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s, c=points[:, 2], cmap="jet_r")
    # set_axes_equal(ax)
    plt.show()

def get_open3d_pointcloud(depth_map):
    points = _image2pointcloud(depth_map)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    minima = min(points[:, 2])
    maxima = max(points[:, 2])

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="jet_r")

    colors = []
    for v in points[:, 2]:
        colors.append(mapper.to_rgba(v)[:3])
    colors = np.array(colors)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def points_to_open3d_pointcloud(points):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    minima = min(points[:, 2])
    maxima = max(points[:, 2])

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="jet_r")

    colors = []
    for v in points[:, 2]:
        colors.append(mapper.to_rgba(v)[:3])
    colors = np.array(colors)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd