import numpy as np

import geometry

class Camera:
    def __init__(self, tstamps, poses):
        assert len(tstamps) == len(poses)
        self._contents = dict(zip(tstamps, poses))

    def poses(self):
        return self._contents.values()

    def tstamp(self):
        return self._contents.keys()

    def __getitem__(self, tstamps):
        if isinstance(tstamps, list):
            poses = [self._contents[tstamp] for tstamp in tstamps]
        else:
            poses = self._contents[tstamps]
        return poses

class Traverse:
    def __init__(self, fpath, **cameras):
        self.fpath = fpath
        for cam_name in cameras:
            setattr(self, key, cameras[cam_name])

    def NN(self, poses, k=1, alpha=15):
        """
        Returns the top-k closest images within the traverse for each pose
        in list of poses. 
        """
        all_poses, all_tstamps, all_cam_names = self._aggregate()
        all_poses = geometry.combine(all_poses)

        poses_NN = []
        tstamps_NN = []
        cam_names_NN = []

        for pose in poses:
            dists = geometry.metric(pose, all_poses, alpha)
            ind = np.argpartition(dists, k)[:k]
            ind = ind[np.argsort(dist[ind])]
            poses_NN.append(all_poses[ind])
            tstamps_NN.append(all_tstamps[ind])
            cam_names_NN.append(all_cam_names[ind])
        return poses_NN, tstamps_NN, cam_names_NN

    def query_radius(self, poses, radius, alpha=15):
        """
        Return all images within a given radius to each pose in list of poses. 
        """
        all_poses, all_tstamps, all_cam_names = self._aggregate()
        all_poses = geometry.combine(all_poses)

        poses_NN = []
        tstamps_NN = []
        cam_names_NN = []

        for pose in poses:
            dists = geometry.metric(pose, all_poses, alpha)
            ind = np.squeeze(np.argwhere(dists < radius))
            poses_NN.append(all_poses[ind])
            tstamps_NN.append(all_tstamps[ind])
            cam_names_NN.append(all_cam_names[ind])
        return poses_NN, tstamps_NN, cam_names_NN

    def _aggregate(self):
        all_poses = []
        all_tstamps = []
        all_cam_names = []
        for cam_name, camera in self.__dict__:
            if cam_name != 'fpath':
                all_poses.extend(camera.poses())
                all_tstamps.extend(camera.tstamps())
                all_cam_names.extend([cam_name] * len(camera))
        return all_poses, all_tstamps, all_cam_names
