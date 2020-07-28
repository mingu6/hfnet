from collections import namedtuple
from bisect import bisect_left

import numpy as np

from . import geometry
from .geometry import SE3

Camera = namedtuple("Camera", ["poses", "timestamps"])


class Traverse:
    def __init__(self, name, **camera_dfs):
        self.name = name
        for i, (cam_name, camera_df) in enumerate(camera_dfs.items()):
            camera_df.sort_values(by=["timestamp"], inplace=True)
            # set traverse timestamps
            if i == 0:
                self.timestamps = np.squeeze(camera_df[["timestamp"]].to_numpy())
            else:
                curr_tstamps = np.squeeze(camera_df[["timestamp"]].to_numpy())
                assert np.array_equal(self.timestamps, curr_tstamps), (
                    "Timestamps between cameras are inconsistent, please"
                    "check that the cameras come from the same traverse!"
                )
            xyzrpy = camera_df[
                ["northing", "easting", "down", "roll", "pitch", "yaw"]
            ].to_numpy()
            poses = SE3.from_xyzrpy(xyzrpy)
            camera = Camera(poses=poses, timestamps=self.timestamps)
            setattr(self, cam_name, camera)

    def __len__(self):
        return len(self.timestamps)

    def query_img(self, camera, timestamp):
        """
        Return pose for camera/timestamp pair within traverse.
        """
        camera = getattr(self, camera)
        # locate camera timestamp index and return associated pose
        i = bisect_left(camera.timestamps, timestamp)
        if i != len(camera.timestamps) and camera.timestamps[i] == timestamp:
            return camera.poses[i]
        return None

    def kNN(self, pose, k, alpha=5):
        retrieved = []
        # aggregate all cameras and find NN images
        poses = []
        timestamps = []
        cameras = []
        for cam_name, camera in self.__dict__.items():
            if cam_name not in ["name", "timestamps"]:
                poses.extend(camera.poses)
                timestamps.extend(self.timestamps)
                cameras.extend(len(self.timestamps) * [cam_name])
        poses = geometry.combine(poses)
        # find NNs
        t_err, R_err = geometry.error(pose, poses)
        dist = geometry.metric(pose, poses, alpha)
        match_ind = np.argpartition(dist, k)[:k]
        for ind in match_ind:
            retrieved.append(
                {
                    "camera": cameras[ind],
                    "timestamp": timestamps[ind],
                    "t_err": t_err[ind],
                    "R_err": R_err[ind] * 180 / np.pi,
                }
            )
        return retrieved

    def query_tolerance(self, pose, t, R):
        """
        Return all images inside given error tolerances to given pose.
        """
        retrieved = []
        for cam_name, camera in self.__dict__.items():
            if cam_name not in ["name", "timestamps"]:
                t_err, R_err = geometry.error(pose, camera.poses)
                match = np.logical_and(t_err < t, R_err * 180 / np.pi < R)
                match_ind = np.squeeze(np.argwhere(match))
                for ind in match_ind:
                    retrieved.append(
                        {
                            "camera": cam_name,
                            "timestamp": self.timestamps[ind],
                            "t_err": t_err[ind],
                            "R_err": R_err[ind] * 180 / np.pi,
                        }
                    )
        return retrieved
