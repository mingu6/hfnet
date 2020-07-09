import os

import numpy as np

from hfnet.settings import DATA_PATH, RAW_PATH
from thirdparty.robotcar_dataset_sdk.python.interpolate_poses import \
        interpolate_ins_poses

traverses = {
    'overcast-reference': '2014-11-28-12-07-13',
    'dawn': '2014-12-16-09-14-09',
    'dusk': '2015-02-20-16-34-06',
    'night': '2014-12-10-18-10-50',
    'night-rain': '2014-12-17-18-18-43',
    'overcast-summer': '2015-05-22-11-14-30',
    'overcast-winter': '2015-11-13-10-28-08',
    'snow': '2015-02-03-08-45-10',
    'sun': '2015-03-10-14-18-10'
}

cameras = ['left', 'right', 'rear']

def main(args):
    for name, datetime in traverses.items():
        for cam_name in cameras:
            # retrieve list of image tstamps
            img_folder = os.path.join(DATA_PATH, 'robotcar/images', name, cam_name)
            img_paths = os.listdir(img_folder)
            tstamps = [int(os.path.basename(img_path)[:-4]) for img_path in img_paths]
            ins_path = os.path.join(RAW_PATH, datetime, 'gps/ins.csv')
            interp_poses = interpolate_ins_poses(ins_path, tstamps)
            # apply camera extrinsics to INS for abs camera poses

if __name__ == "__main__":
    args = None
    main(args)
