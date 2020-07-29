import numpy as np
import cv2


def topk_matching_gps(query_pose, db_gps, k, imperfect=False):
    """for a query image, retrieve timestamps of top k nearest db
    images by INS pose. Return all relevant/overlapping cameras
    cameras_agg also.
    """
    cameras_agg = db_gps.kNN(query_pose, max(k, 20))
    if imperfect:
        cameras = np.choice(cameras, size=k)
    else:
        cameras = db_gps.kNN(query_pose, k)
    return cameras, cameras_agg


def retrieve_indices(dataset_name, db_names, topk_cameras):
    if dataset_name == "robotcar":
        reference = "overcast-reference"
    keys = [
        reference + "/" + camera["camera"] + "/" + str(camera["timestamp"]) + ".jpg"
        for camera in topk_cameras
    ]
    indices = [db_names.index(key) for key in keys]
    return indices
