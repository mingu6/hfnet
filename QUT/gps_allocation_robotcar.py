import os

import numpy as np
import pickle
from scipy.spatial.transform import Rotation

from hfnet.settings import DATA_PATH, RAW_PATH
import thirdparty.robotcar_dataset_sdk as sdk
from thirdparty.robotcar_dataset_sdk.python.interpolate_poses import interpolate_ins_poses
from thirdparty.robotcar_dataset_sdk.python.transform import build_se3_transform
from QUT.geometry import SE3Poses
from QUT.datasets import Camera, Traverse

traverses = {
    'overcast-reference': '2014-11-28-12-07-13',
    'dawn': '2014-12-16-09-14-09',
    'dusk': '2015-02-20-16-34-06',
    'night': '2014-12-10-18-10-50',
    'night-rain': '2014-12-17-18-18-43',
    #'overcast-summer': '2015-05-22-11-14-30',
    #'overcast-winter': '2015-11-13-10-28-08',
    'snow': '2015-02-03-08-45-10',
    'sun': '2015-03-10-14-18-10'
}

camera_names = ['left', 'right', 'rear']

def main(args):
    sdk_path = os.path.abspath(sdk.__file__)
    extrinsics_dir = os.path.join(os.path.dirname(sdk_path), 'extrinsics')
    save_dir = os.path.join(DATA_PATH, 'robotcar/gps')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for name, datetime in traverses.items():
        if name in ['overcast-reference', 'night']:
            cameras = {}
            for cam_name in camera_names:
                # retrieve list of image tstamps
                img_folder = os.path.join(DATA_PATH, 'robotcar/images', name, cam_name)
                img_paths = os.listdir(img_folder)
                tstamps = [int(os.path.basename(img_path)[:-4]) for img_path in img_paths]
                ins_path = os.path.join(RAW_PATH, datetime, 'gps/ins.csv')
                interp_poses = np.asarray(interpolate_ins_poses(ins_path, tstamps))
                # apply camera extrinsics to INS for abs camera poses. Note extrinsics are relative
                # to the STEREO camera, so compose extrinsics for ins and mono to get true cam pose
                with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
                    extrinsics = next(extrinsics_file)
                T_ext_ste_ins = np.asarray(build_se3_transform([float(x) for x in extrinsics.split(' ')]))
                with open(os.path.join(extrinsics_dir, 'mono_{}.txt'.format(cam_name))) as extrinsics_file:
                    extrinsics = next(extrinsics_file)
                T_ext_ste_mono = np.asarray(build_se3_transform([float(x) for x in extrinsics.split(' ')]))
                T_ext_ins_mono = np.linalg.solve(T_ext_ste_ins, T_ext_ste_mono)
                mono_poses = interp_poses @ T_ext_ins_mono
                # add poses to camera object
                mono_poses_SE3 = SE3Poses(mono_poses[:, :3, 3], Rotation.from_matrix(mono_poses[:, :3, :3]))
                mono_cam = Camera(tstamps, mono_poses_SE3)
                cameras[cam_name] = mono_cam
            traverse = Traverse(name, **cameras)
            with open(os.path.join(save_dir, name + '.pickle'), 'wb') as f:
                pickle.dump(traverse, f)

if __name__ == "__main__":
    args = None
    main(args)
