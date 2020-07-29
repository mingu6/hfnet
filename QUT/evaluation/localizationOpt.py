import numpy as np
import pandas as pd
import logging
from pathlib import Path
import os
import pickle
import sys
from tqdm import tqdm

from hfnet.datasets import get_dataset
from hfnet.evaluation.utils import db_management
from hfnet.evaluation.utils.descriptors import topk_matching
from hfnet.evaluation.utils.db_management import (
    read_query_list,
    extract_query,
    build_localization_dbs,
    colmap_image_to_pose,
)
from hfnet.evaluation.utils.localization import (
    covis_clustering,
    match_against_place,
    do_pnp,
    preprocess_globaldb,
    preprocess_localdb,
    loc_failure,
    LocResult,
)
from hfnet.evaluation.localization import Localization
from hfnet.datasets.colmap_utils.read_model import read_model
from hfnet.evaluation.cpp_localization import CppLocalization
from hfnet.utils.tools import Timer
from hfnet.settings import DATA_PATH
from QUT.util.Traverse import Traverse
from QUT.evaluation.utils.descriptors import topk_matching_gps, retrieve_indices

sys.modules["hfnet.evaluation.db_management"] = db_management  # backward comp


class LocalizationOpt(Localization):
    def __init__(self, dataset_name, model_name, config, build_db=False):
        super().__init__(dataset_name, model_name, config, build_db=build_db)
        # load GPS data for reference
        if dataset_name == "robotcar":
            traverse = "overcast-reference"
        self.gps = Traverse(dataset_name, traverse, config["global"]["experiment"])

    def init_queries(self, query_file, query_config, prefix=""):
        queries = read_query_list(Path(self.base_path, query_file), prefix=prefix)
        Dataset = get_dataset(query_config.get("name", self.dataset_name))
        query_config = {**query_config, "image_names": [q.name for q in queries]}
        query_dataset = Dataset(**query_config)
        # load GPS data for queries
        query_gps = Traverse(self.dataset_name, self.config["queries"],
                             self.gps.experiment_name)
        return queries, query_dataset, query_gps

    def localize(self, query_info, query_data, query_gps, debug=False):
        config_global = self.config["global"]
        config_local = self.config["local"]
        config_pose = self.config["pose"]
        timings = {}

        # Fetch data
        query_item = extract_query(query_data, query_info, config_global, config_local)

        # C++ backend
        if self.use_cpp:
            assert not debug
            assert hasattr(self, "cpp_backend")
            return self.cpp_backend.localize(
                query_info, query_item, self.global_transform, self.local_transform
            )

        # Global matching
        with Timer() as t:
            global_desc = self.global_transform(
                query_item.global_desc[np.newaxis])[0]
            splits = query_info.name.split("/")
            query_attr = query_gps.query_attr(splits[1], int(splits[2][:-4]))
            if self.config["num_nearest"] > 0:
                distractors = self.gps.retrieve_distractors(
                    query_attr, self.config["num_distractors"])
                nearest = self.gps.kNN(query_attr["pose"],
                                       self.config["num_nearest"],
                                       self.config["imperfect"])
                relevant_cameras = nearest + distractors
            else:
                relevant_cameras = self.gps.topk_descriptors(
                    query_attr, self.config["num_distractors"])
            indices = retrieve_indices(self.dataset_name,
                                       self.db_names, relevant_cameras)
            prior_ids = self.db_ids[indices]
        timings["global"] = t.duration

        # Clustering
        with Timer() as t:
            clustered_frames = covis_clustering(prior_ids, self.local_db, self.points)
            local_desc = self.local_transform(query_item.local_desc)
        timings["covis"] = t.duration

        # Iterative pose estimation
        dump = []
        results = []
        timings["local"], timings["pnp"] = 0, 0
        for place in clustered_frames:
            # Local matching
            matches_data = {} if debug else None
            matches, place_lms, duration = match_against_place(
                place,
                self.local_db,
                local_desc,
                config_local["ratio_thresh"],
                do_fast_matching=config_local.get("fast_matching", True),
                debug_dict=matches_data,
            )
            timings["local"] += duration

            # PnP
            if len(matches) > 3:
                with Timer() as t:
                    matched_kpts = query_item.keypoints[matches[:, 0]]
                    matched_lms = np.stack(
                        [self.points[place_lms[i]].xyz for i in matches[:, 1]]
                    )
                    result, inliers = do_pnp(
                        matched_kpts, matched_lms, query_info, config_pose
                    )
                timings["pnp"] += t.duration
            else:
                result = loc_failure
                inliers = np.empty((0,), np.int32)

            results.append(result)
            if debug:
                dump.append(
                    {
                        "query_item": query_item,
                        "prior_ids": prior_ids,
                        "places": clustered_frames,
                        "matching": matches_data,
                        "matches": matches,
                        "inliers": inliers,
                    }
                )
            if result.success:
                break

        # In case of failure we return the pose of the first retrieved prior
        if not result.success:
            result = results[0]
            result = LocResult(
                False,
                result.num_inliers,
                result.inlier_ratio,
                colmap_image_to_pose(self.images[prior_ids[0]]),
            )

        if debug:
            debug_data = {
                **(dump[-1 if result.success else 0]),
                "index_success": (len(dump) - 1) if result.success else -1,
                "dumps": dump,
                "results": results,
                "timings": timings,
            }
            return result, debug_data
        else:
            return result, {"timings": timings}


def evaluate(loc, queries, query_dataset, query_gps, max_iter=None):
    results = []
    all_stats = []
    query_iter = query_dataset.get_test_set()

    for query_info, query_data in tqdm(zip(queries, query_iter), total=len(queries)):
        result, stats = loc.localize(query_info, query_data, query_gps, debug=False)
        results.append(result)
        all_stats.append(stats)

        if max_iter is not None:
            if len(results) == max_iter:
                break

    success = np.array([r.success for r in results])
    num_inliers = np.array([r.num_inliers for r in results])
    ratios = np.array([r.inlier_ratio for r in results])

    metrics = {
        "success": np.mean(success),
        "inliers": np.mean(num_inliers[success]),
        "inlier_ratios": np.mean(ratios[success]),
        "failure": np.arange(len(success))[np.logical_not(success)],
    }
    metrics = {k: v.tolist() for k, v in metrics.items()}
    metrics["all_stats"] = all_stats
    return metrics, results
