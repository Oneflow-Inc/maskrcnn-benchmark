from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import torch
import json
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--py", type=str, required=True)
parser.add_argument("--flow", type=str, required=True)
parser.add_argument("--map", default=None, type=str, required=True, help="model_name_map.txt.")

args = parser.parse_args()


def get_torch_model_name(name):
    if name.startswith("module."):
        return name[len("module."):]
    return name


def get_pytorch_state_dict(path, rm_prefix=False):
    return torch.load(path)


def get_pytorch_model_dict(state_dict, rm_prefix=False):
    iteration = state_dict["iteration"]
    print(iteration)
    model_dict = state_dict["model"]
    if rm_prefix:
        model_dict_new = model_dict.copy()
        for old_name, _ in model_dict.items():
            if old_name.startswith("module."):
                new_name = old_name[len("module.") :]
                model_dict_new[new_name] = model_dict[old_name]
        return model_dict_new
    else:
        return model_dict


def get_torch_model_name2optm_stat_dict(state_dict):
    return state_dict["optimizer"]


def load_flow_npy(root, sub_path):
    joined = os.path.join(root, sub_path)
    if os.path.isfile(joined):
        return np.fromfile(joined, dtype=np.float32)
    else:
        # print(joined, "not found")
        return None


def load_torch_npy(dic, key):
    key_without_prefix = get_torch_model_name(key)
    if key in dic:
        return dic[key].cpu().numpy()
    if key_without_prefix in dic:
        return dic[key_without_prefix].cpu().numpy()
    return None


def load_torch_optm_state_npy(state_dict, model_name):
    if model_name.startswith("module."):
        model_name_without_prefix = model_name[len("module."):]
    else:
        model_name_without_prefix = model_name
    if model_name in state_dict["model_name2state"]:
        optm_state_id = state_dict["model_name2state"][model_name]
    elif model_name_without_prefix in state_dict["model_name2state"]:
        optm_state_id = state_dict["model_name2state"][model_name_without_prefix]
    else:
        raise ValueError
    return state_dict["state"][optm_state_id]["momentum_buffer"].cpu().numpy()


def get_corespondent_flow_path(flow_all_iters_path, torch_path):
    latest = flow_all_iters_path
    wild_card = glob.glob(os.path.join(flow_all_iters_path, "model_save-*"))
    if len(wild_card) > 0:
        wild_card.sort(key=os.path.getmtime)
        latest = wild_card[-1]
    iter_num = int(os.path.splitext(os.path.basename(torch_path))[0][-5:])
    joined = os.path.join(latest, "iter-{}".format(iter_num))
    if os.path.isdir(joined):
        return joined
    else:
        return flow_all_iters_path


def relative_diff(arr, abs_diff):
    return np.max(abs_diff / (np.abs(arr) + 1e-10))


def compare_npy(arr1, arr2, bn=None):
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    relative_max = max([relative_diff(arr1, abs_diff), relative_diff(arr2, abs_diff)])
    return pd.DataFrame(
        {
            "bn": bn,
            "shape": str(arr2.shape),
            "equal": np.array_equal(arr1, arr2),
            "diff_max": np.max(diff),
            "diff_min": np.min(diff),
            "abs_diff_max": np.max(abs_diff),
            "relative_max": relative_max,
        },
        index=[0],
    )


def compare_model(flow_path, py_model_dict, map_json_path, optm_state_dict=None):
    with open(map_json_path) as json_file:
        model_map = json.load(json_file)
        done_compare_flow = []
        done_compare_torch = []

        cmp_df = pd.DataFrame()
        for mapper in model_map:
            flow_npy = load_flow_npy(flow_path, mapper["flow"])
            flow_momentum_key = mapper["flow"][: -len("/out")] + "-momentum/out"
            flow_momentum_npy = load_flow_npy(flow_path, flow_momentum_key)
            torch_npy = load_torch_npy(py_model_dict, mapper["torch"])

            if (
                flow_npy is not None and torch_npy is not None
            ):  # and mapper["torch"].endswith("bias") == False and mapper["flow"].endswith("scale/out") == False:
                # print("\n")
                cmp = compare_npy(
                    flow_npy.reshape(torch_npy.shape), torch_npy, bn=mapper["flow"][: -len("/out")]
                )
                cmp_df = pd.concat([cmp_df, cmp], axis=0, sort=False)

                done_compare_flow.append(mapper["flow"])
                done_compare_torch.append(mapper["torch"])

                if flow_momentum_npy is not None:
                    if "model_name2state" in optm_state_dict:
                        flow_torch_npy = load_torch_optm_state_npy(optm_state_dict, mapper["torch"])
                        cmp = compare_npy(
                            flow_momentum_npy.reshape(flow_torch_npy.shape),
                            flow_torch_npy,
                            bn=flow_momentum_key[: -len("/out")],
                        )
                        cmp_df = pd.concat([cmp_df, cmp], axis=0, sort=False)
                        done_compare_flow.append(flow_momentum_key)
            else:
                if flow_npy is None:
                    print("{} not found".format(mapper["flow"]))

                if torch_npy is None:
                    print("{} not found".format(mapper["torch"]))

        # cmp_df = cmp_df.drop(["equal", "shape"], axis=1)
        # cmp_df = cmp_df[cmp_df["abs_diff_max"] > 0.01]
        cmp_df = cmp_df[cmp_df["equal"] == False]
        cmp_df = cmp_df.sort_values(by=['abs_diff_max'], ascending=False)
        print(cmp_df.to_string(index=False))

        for k in py_model_dict.keys():
            if not k.startswith("module."):
                k = "module." + k
            if k not in done_compare_torch:
                if (
                    not (k.endswith("running_var") or k.endswith("running_mean"))
                    and "cell_anchors" not in k
                ):
                    print(k)

        for k in os.listdir(flow_path):
            if os.path.join(k, "out") not in done_compare_flow:
                print(k)
            # else:
            # print("skipping: {}".format(mapper["flow"]))

    print(flow_path)


if __name__ == "__main__":
    state_dict = get_pytorch_state_dict(args.py)

    compare_model(
        get_corespondent_flow_path(args.flow, args.py),
        get_pytorch_model_dict(state_dict),
        args.map,
        get_torch_model_name2optm_stat_dict(state_dict),
    )
