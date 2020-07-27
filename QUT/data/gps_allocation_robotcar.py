import os

import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd

from hfnet.settings import DATA_PATH, RAW_PATH
import thirdparty.robotcar_dataset_sdk as sdk
from thirdparty.robotcar_dataset_sdk.python.interpolate_poses import (
    interpolate_ins_poses,
)
from thirdparty.robotcar_dataset_sdk.python.transform import (
    build_se3_transform,
    se3_to_components,
)
from QUT.settings import traverses, camera_names
from QUT.util.geometry import SE3
from QUT.util.Traverse import Camera, Traverse


if __name__ == "__main__":
    sdk_path = os.path.abspath(sdk.__file__)
    extrinsics_dir = os.path.join(os.path.dirname(sdk_path), "extrinsics")
    save_dir = os.path.join(DATA_PATH, "robotcar/gps")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for traverse, datetime in traverses.items():
        for cam_name in camera_names:
            # retrieve list of image tstamps
            img_folder = os.path.join(DATA_PATH, "robotcar/images", traverse, cam_name)
            img_paths = os.listdir(img_folder)
            tstamps = [int(os.path.basename(img_path)[:-4]) for img_path in img_paths]
            ins_path = os.path.join(RAW_PATH, datetime, "gps/ins.csv")
            interp_poses = np.asarray(interpolate_ins_poses(ins_path, tstamps))
            # apply camera extrinsics to INS for abs camera poses. Note extrinsics are relative
            # to the STEREO camera, so compose extrinsics for ins and mono to get true cam pose
            with open(os.path.join(extrinsics_dir, "ins.txt")) as extrinsics_file:
                extrinsics = next(extrinsics_file)
            T_ext_ste_ins = np.asarray(
                build_se3_transform([float(x) for x in extrinsics.split(" ")])
            )
            with open(
                os.path.join(extrinsics_dir, "mono_{}.txt".format(cam_name))
            ) as extrinsics_file:
                extrinsics = next(extrinsics_file)
            T_ext_ste_mono = np.asarray(
                build_se3_transform([float(x) for x in extrinsics.split(" ")])
            )
            T_ext_ins_mono = np.linalg.solve(T_ext_ste_ins, T_ext_ste_mono)
            mono_poses = interp_poses @ T_ext_ins_mono
            xyzrpys = [se3_to_components(pose) for pose in mono_poses]
            df = pd.DataFrame(
                xyzrpys, columns=["northing", "easting", "down", "roll", "pitch", "yaw"]
            )
            df.insert(0, "timestamp", tstamps, True)
            trav_save_dir = os.path.join(save_dir, traverse)
            if not os.path.exists(trav_save_dir):
                os.makedirs(trav_save_dir)
            df.to_csv(os.path.join(trav_save_dir, "{}_poses.csv".format(cam_name)))
