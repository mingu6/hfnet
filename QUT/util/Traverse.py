from collections import namedtuple
from bisect import bisect_left

import numpy as np

from . import geometry
from .geometry import SE3, metric

Camera = namedtuple("Camera", ["poses", "timestamps"])


class Traverse:
    def __init__(self, name, **camera_dfs):
        self.name = name
        for i, (cam_name, camera_df) in enumerate(camera_dfs.items()):
            # set traverse timestamps
            if i == 0:
                self.timestamps = camera_df[["timestamp"]].to_numpy()
            else:
                curr_tstamps = camera_df[["timestamp"]].to_numpy()
                assert np.array_equal(self.timestamps, curr_tstamps), (
                    "Timestamps between cameras are inconsistent, please"
                    "check that the cameras come from the same traverse!"
                )
            # set each camera as traverse attributes
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
                            "pose": camera.poses[ind],
                        }
                    )
        return retrieved
