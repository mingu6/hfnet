import numpy as np

from . import geometry

class Camera:
    def __init__(self, tstamps, poses):
        assert len(tstamps) == len(poses)
        self._contents = {}
        for i in range(len(tstamps)):
            self._contents[tstamps[i]] = poses[i]

    def poses(self):
        return self._contents.values()

    def tstamps(self):
        return self._contents.keys()

    def __len__(self):
        return len(self._contents)

    def __getitem__(self, tstamps):
        if isinstance(tstamps, list):
            poses = [self._contents[tstamp] for tstamp in tstamps]
        else:
            poses = self._contents[tstamps]
        return poses

class Traverse:
    def __init__(self, name, **cameras):
        self.name = name
        for cam_name, camera in cameras.items():
            setattr(self, cam_name, camera)

    def NN(self, pose, k=1, alpha=15):
        """
        Returns the top-k closest images within the traverse for each pose
        in list of poses.
        """
        all_poses, all_tstamps, all_cam_names = self._aggregate()
        all_poses = geometry.combine(all_poses)

        dists = geometry.metric(pose, all_poses, alpha)
        ind = np.argpartition(dists, k)[:k]
        ind = ind[np.argsort(dists[ind])]
        return [all_poses[i] for i in ind], [all_tstamps[i] for i in ind], \
                    [all_cam_names[i] for i in ind]

    def query_radius(self, pose, radius, alpha=15):
        """
        Return all images within a given radius to each pose in list of poses. 
        """
        all_poses, all_tstamps, all_cam_names = self._aggregate()
        all_poses = geometry.combine(all_poses)

        dists = geometry.metric(pose, all_poses, alpha)
        ind = np.squeeze(np.argwhere(dists < radius))
        return [all_poses[i] for i in ind], [all_tstamps[i] for i in ind], \
                    [all_cam_names[i] for i in ind]

    def _aggregate(self):
        all_poses = []
        all_tstamps = []
        all_cam_names = []
        for cam_name, camera in self.__dict__.items():
            if cam_name != 'name':
                all_poses.extend(camera.poses())
                all_tstamps.extend(camera.tstamps())
                all_cam_names.extend([cam_name] * len(camera))
        return all_poses, all_tstamps, all_cam_names
