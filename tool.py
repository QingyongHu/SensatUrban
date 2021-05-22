from os.path import join, exists, dirname, abspath
import numpy as np
import colorsys, random, os, sys
import open3d as o3d
from helper_ply import read_ply, write_ply

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigSensatUrban:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.2  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 14  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_SensatUrban'
    saving = True
    saving_path = None


class DataProcessing:

    @staticmethod
    def get_num_class_from_label(labels, total_class):
        num_pts_per_class = np.zeros(total_class, dtype=np.int32)
        # original class distribution
        val_list, counts = np.unique(labels, return_counts=True)
        for idx, val in enumerate(val_list):
            num_pts_per_class[val] += counts[idx]
        # for idx, nums in enumerate(num_pts_per_class):
        #     print(idx, ':', nums)
        return num_pts_per_class

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def read_ply_data(path, with_rgb=True, with_label=True):
        data = read_ply(path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        if with_rgb and with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['class']
            return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
        elif with_rgb and not with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            return xyz.astype(np.float32), rgb.astype(np.uint8)
        elif not with_rgb and with_label:
            labels = data['class']
            return xyz.astype(np.float32), labels.astype(np.uint8)
        elif not with_rgb and not with_label:
            return xyz.astype(np.float32)

    @staticmethod
    def random_sub_sampling(points, features=None, labels=None, sub_ratio=10, verbose=0):
        num_input = np.shape(points)[0]
        num_output = num_input // sub_ratio
        idx = np.random.choice(num_input, num_output)
        if (features is None) and (labels is None):
            return points[idx]
        elif labels is None:
            return points[idx], features[idx]
        elif features is None:
            return points[idx], labels[idx]
        else:
            return points[idx], features[idx], labels[idx]

    @staticmethod
    def get_class_weights(num_per_class, name='sqrt'):
        # # pre-calculate the number of points in each category
        frequency = num_per_class / float(sum(num_per_class))
        if name == 'sqrt' or name == 'lovas':
            ce_label_weight = 1 / np.sqrt(frequency)
        elif name == 'wce':
            ce_label_weight = 1 / (frequency + 0.02)
        else:
            raise ValueError('Only support sqrt and wce')
        return np.expand_dims(ce_label_weight, axis=0)


class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])

        o3d.geometry.PointCloud.estimate_normals(pc)
        o3d.visualization.draw_geometries([pc], width=1000, height=1000)
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        # only visualize a number of points to save memory
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            # ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=1)
            ins_colors = [[85, 107, 47],  # ground -> OliveDrab
                          [0, 255, 0],  # tree -> Green
                          [255, 165, 0],  # building -> orange
                          [41, 49, 101],  # Walls ->  darkblue
                          [0, 0, 0],  # Bridge -> black
                          [0, 0, 255],  # parking -> blue
                          [255, 0, 255],  # rail -> Magenta
                          [200, 200, 200],  # traffic Roads ->  grey
                          [89, 47, 95],  # Street Furniture  ->  DimGray
                          [255, 0, 0],  # cars -> red
                          [255, 255, 0],  # Footpath  ->  deeppink
                          [0, 255, 255],  # bikes -> cyan
                          [0, 191, 255]  # water ->  skyblue
                          ]

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins

    @staticmethod
    def save_ply_o3d(data, save_name):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
        if np.shape(data)[1] == 3:
            o3d.io.write_point_cloud(save_name, pcd)
        elif np.shape(data)[1] == 6:
            if np.max(data[:, 3:6]) > 20:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.)
            else:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
            o3d.io.write_point_cloud(save_name, pcd)
        return

