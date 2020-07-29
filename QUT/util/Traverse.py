import os
from collections import namedtuple
from bisect import bisect_left

import numpy as np
import pandas as pd
from tqdm import tqdm

from hfnet.settings import DATA_PATH, EXPER_PATH
from . import geometry
from .geometry import SE3

Camera = namedtuple("Camera", ["poses", "timestamps", "descriptors"])
attributes = ["name", "timestamps", "traverse_name", "dataset_name", "experiment_name"]


class Traverse:
    def __init__(self, dataset_name, traverse_name, experiment_name):
        self.dataset_name = dataset_name
        self.traverse_name = traverse_name
        self.experiment_name = experiment_name
        # import INS data
        gps_dir = os.path.join(DATA_PATH, dataset_name, "gps", traverse_name)
        for i, gpsname in enumerate(os.listdir(gps_dir)):
            camera_df = pd.read_csv(os.path.join(gps_dir, gpsname))
            cam_name = gpsname[:-10]
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
            # read descriptors
            descriptor_dir = os.path.join(
                EXPER_PATH,
                "exports",
                experiment_name,
                traverse_name,
                cam_name,
            )
            example_fname = os.listdir(descriptor_dir)[0]
            D = len(np.load(os.path.join(descriptor_dir, example_fname))
                    ["global_descriptor"])
            descriptors = np.empty((len(self.timestamps), D))
            for i, tstamp in enumerate(tqdm(self.timestamps)):
                dpath = os.path.join(descriptor_dir, str(tstamp) + ".npz")
                descriptors[i, :] = np.load(dpath)["global_descriptor"]
            poses = SE3.from_xyzrpy(xyzrpy)
            camera = Camera(poses=poses, timestamps=self.timestamps,
                            descriptors=descriptors)
            setattr(self, cam_name, camera)

    def __len__(self):
        return len(self.timestamps)

    def topk_descriptors(self, query_attr, k):
        # retrieve attributes
        query_desc = query_attr["descriptor"]
        query_pose = query_attr["pose"]
        # top k most similar descriptors
        poses, timestamps, cameras, descriptors = self._aggregate()
        dist_sq = 2 - 2 * descriptors @ query_desc
        match_ind = np.argpartition(dist_sq, k)[:k]
        match_ind = match_ind[np.argsort(dist_sq[match_ind])]
        # extract INS information
        t_err, R_err = geometry.error(query_pose, poses)
        retrieved = []
        for ind in match_ind:
            retrieved.append(
                {
                    "camera": cameras[ind],
                    "timestamp": timestamps[ind],
                    "t_err": t_err[ind],
                    "R_err": R_err[ind] * 180 / np.pi,
                    "ind": ind,
                }
            )
        return retrieved

    def retrieve_distractors(self, query_attr, k):
        # identify relevant images
        query_pose = query_attr["pose"]
        relevant = self.kNN(query_pose, 10)
        relevant_ind = [attr["ind"] for attr in relevant]
        # image retrieval
        img_retrieval = self.topk_descriptors(query_attr, 10 + k)
        retrieval_ind = [attr["ind"] for attr in img_retrieval]
        # compute INS error
        poses, timestamps, cameras, descriptors = self._aggregate()
        t_err, R_err = geometry.error(query_pose, poses)
        # cull retrieved images that are close to gt ("relevant")
        distractors = []
        for ind in retrieval_ind:
            if ind not in relevant_ind and len(distractors) < k:
                distractors.append(
                    {
                        "camera": cameras[ind],
                        "timestamp": timestamps[ind],
                        "t_err": t_err[ind],
                        "R_err": R_err[ind] * 180 / np.pi,
                        "ind": ind,
                    })
        return distractors

    def query_attr(self, camera, timestamp):
        """
        Return pose  and descriptor for camera/timestamp pair within traverse.
        """
        camera = getattr(self, camera)
        # locate camera timestamp index and return associated pose
        i = bisect_left(camera.timestamps, timestamp)
        if i != len(camera.timestamps) and camera.timestamps[i] == timestamp:
            return {"pose": camera.poses[i],
                    "descriptor": camera.descriptors[i],
                    "ind": i}
        return None

    def kNN(self, pose, k, alpha=5, imperfect=False):
        retrieved = []
        # aggregate all cameras and find NN images
        poses, timestamps, cameras, _ = self._aggregate()
        # find NNs
        t_err, R_err = geometry.error(pose, poses)
        dist = geometry.metric(pose, poses, alpha)
        if imperfect:
            # imperfect retrieval retrieves k random relevant images
            match_ind = np.argpartition(dist, max(10, k))[:max(10, k)]
            match_ind = np.random.choice(match_ind, k)
        else:
            match_ind = np.argpartition(dist, k)[:k]
        match_ind = match_ind[np.argsort(dist[match_ind])]
        for ind in match_ind:
            retrieved.append(
                {
                    "camera": cameras[ind],
                    "timestamp": timestamps[ind],
                    "t_err": t_err[ind],
                    "R_err": R_err[ind] * 180 / np.pi,
                    "ind": ind,
                }
            )
        return retrieved

    def query_tolerance(self, pose, t, R):
        """
        Return all images inside given error tolerances to given pose.
        """
        retrieved = []
        for cam_name, camera in self.__dict__.items():
            if cam_name not in attributes:
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
                            "ind": ind,
                        }
                    )
        return retrieved

    def _aggregate(self):
        poses = []
        timestamps = []
        cameras = []
        descriptors = []
        for cam_name, camera in self.__dict__.items():
            if cam_name not in attributes:
                poses.extend(camera.poses)
                timestamps.extend(self.timestamps)
                cameras.extend(len(self.timestamps) * [cam_name])
                descriptors.append(camera.descriptors)
        poses = geometry.combine(poses)
        descriptors = np.concatenate(descriptors, axis=0)
        return poses, timestamps, cameras, descriptors
