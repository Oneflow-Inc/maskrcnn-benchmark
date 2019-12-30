from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle as pkl
import argparse
import torch
import json
parser = argparse.ArgumentParser()
import compare
import glob

parser.add_argument(
    "--py",
    type=str,
    required=True,
)
parser.add_argument(
    "--map", default=None, type=str, required=True, help="model_name_map.txt."
)
parser.add_argument("--flow", type=str, required=True)

args = parser.parse_args()


def get_pytorch_model_dict(path):
    state_dict = torch.load(path)
    iteration = state_dict["iteration"]
    print(iteration)
    model_dict = state_dict["model"]
    model_dict_new = model_dict.copy()
    for old_name, _ in model_dict.items():
        if old_name.startswith("module."):
            new_name = old_name[len("module."):]
            model_dict_new[new_name] = model_dict[old_name]
    return model_dict_new

def load_flow_npy(root, sub_path):
    joined = os.path.join(root, sub_path)
    if os.path.isfile(joined):
        return np.fromfile(joined, dtype=np.float32)
    else:
        # print(joined, "not found")
        return None

def load_torch_npy(dic, key):
    if key in dic:
        return dic[key].cpu().numpy()
    else:
        # print(key, "not found")
        None

latest = glob.glob(os.path.join(args.flow, "model_save-*"))
if len(latest) > 0:
    latest.sort(key=os.path.getmtime)
    iter_num = int(os.path.splitext(os.path.basename(args.py))[0][-5:])
    args.flow = os.path.join(latest[-1], "iter-{}".format(iter_num))
    print(args.flow)

model_dict = get_pytorch_model_dict (args.py)
with open(args.map) as json_file:
    model_map = json.load(json_file)
    for mapper in model_map:
        flow_npy = load_flow_npy(args.flow, mapper["flow"])
        torch_npy = load_torch_npy(model_dict, mapper["torch"])
        if flow_npy is not None and torch_npy is not None and mapper["torch"].endswith("bias") == False and mapper["flow"].endswith("scale/out") == False:
            print("\n")
            compare.main(flow_npy.reshape(torch_npy.shape), torch_npy, bn=mapper["flow"], absolute=0.2, verbose=True, level="error")
