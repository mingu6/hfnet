import os
import logging
from pathlib import Path
import argparse
from pprint import pformat
import yaml
import numpy as np
from pyquaternion import Quaternion

from QUT.evaluation.localizationOpt import LocalizationOpt, evaluate
from hfnet.evaluation.loaders import export_loader
from hfnet.settings import EXPER_PATH


configs_global = {
    "netvlad": {
        "db_name": "globaldb_netvlad.pkl",
        "experiment": "netvlad/robotcar",
        "predictor": export_loader,
        "has_keypoints": False,
        "has_descriptors": False,
        "pca_dim": 1024,
        "num_prior": 10,
    },
    "hfnet": {
        "db_name": "globaldb_hfnet.pkl",
        "experiment": "hfnet/robotcar",
        "predictor": export_loader,
        "has_keypoints": False,
        "has_descriptors": False,
        "pca_dim": 1024,
        "num_prior": 10,
    },
}

configs_local = {
    "superpoint": {
        "db_name": "localdb_superpoint.pkl",
        "experiment": "superpoint/robotcar",
        "predictor": export_loader,
        "has_keypoints": True,
        "has_descriptors": True,
        "binarize": False,
        # 'do_nms': True,
        # 'nms_thresh': 4,
        "num_features": 2000,
        "ratio_thresh": 0.9,
    },
    "sift": {
        "db_name": "localdb_sift.pkl",
        "colmap_db": "overcast-reference.db",
        "colmap_db_queries": "query.db",
        "broken_db": True,
        "root": False,
        "ratio_thresh": 0.7,
        "fast_matching": False,
    },
    "hfnet": {
        "db_name": "localdb_hfnet.pkl",
        "experiment": "hfnet/robotcar",
        "predictor": export_loader,
        "has_keypoints": True,
        "has_descriptors": True,
        # 'do_nms': True,
        # 'nms_thresh': 4,
        "num_features": 2000,
        "ratio_thresh": 0.9,
    },
}

config_pose = {
    "reproj_error": 12,
    "min_inliers": 15,
}

config_robotcar = {
    "resize_max": 960,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("eval_name", type=str)
    parser.add_argument("--local_method", type=str)
    parser.add_argument("--global_method", type=str)
    parser.add_argument("--build_db", action="store_true")
    parser.add_argument("--queries", type=str, default="dusk_left")
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--export_poses", action="store_true")
    parser.add_argument("--cpp_backend", action="store_true")
    parser.add_argument("--num_nearest", default=1, type=int)
    parser.add_argument("--num_distractors", default=1, type=int)
    parser.add_argument("--imperfect", action="store_true")
    args = parser.parse_args()

    config = {
        "global": configs_global[args.global_method],
        "local": configs_local[args.local_method],
        "robotcar": config_robotcar,
        "pose": config_pose,
        "model": args.model,
        "max_iter": args.max_iter,
        "queries": args.queries,
        "use_cpp": args.cpp_backend,
        "num_nearest": args.num_nearest,
        "num_distractors": args.num_distractors,
        "imperfect": args.imperfect,
    }
    logging.info("Evaluating Robotcar with configuration: \n" + pformat(config))
    loc = LocalizationOpt("robotcar", args.model, config, build_db=args.build_db)

    query_file = f"queries/{args.queries}_queries_with_intrinsics.txt"
    queries, query_dataset, query_gps = loc.init_queries(query_file, config_robotcar)

    logging.info("Starting evaluation")
    metrics, results = evaluate(
        loc, queries, query_dataset, query_gps, max_iter=args.max_iter
    )
    logging.info("Evaluation metrics: \n" + pformat(metrics))

    output = {"config": config, "metrics": metrics}
    output_dir = Path(EXPER_PATH, "eval/robotcar")
    output_dir.mkdir(exist_ok=True, parents=True)
    if "night" in args.queries:
        query_cat = "night"
    else:
        query_cat = "day"
    eval_filename = (
        f"{args.eval_name}_{query_cat}_{args.num_nearest}NN_{args.num_distractors}d"
    )
    eval_filename_yaml = (
        f"{args.eval_name}_{args.queries}_{args.num_nearest}NN_{args.num_distractors}d"
    )
    eval_path = Path(output_dir, f"{eval_filename_yaml}.yaml")
    with open(eval_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)

    if args.export_poses:
        poses_path = Path(output_dir, f"{eval_filename}_poses.txt")
        if os.path.exists(poses_path):
            append_write = "a"
        else:
            append_write = "w"
        with open(poses_path, append_write) as f:
            for query, result in zip(queries, results):
                query_T_w = np.linalg.inv(result.T)
                qvec_nvm = list(Quaternion(matrix=query_T_w))
                pos_nvm = query_T_w[:3, 3].tolist()
                name = "/".join(query.name.split("/")[-2:])
                line = name + " " + " ".join(map(str, qvec_nvm + pos_nvm))
                f.write(line + "\n")
