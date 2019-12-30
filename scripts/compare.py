import numpy as np
import os
import glob
import sys
from termcolor import colored
from string import Template
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", type=argparse.FileType("r"), nargs="+")

parser.add_argument(
    "-v", "--verbose", default=False, action="store_true", required=False
)
parser.add_argument(
    "-s", "--slient", default=False, action="store_true", required=False
)
parser.add_argument(
    "-abs", "--absolute", type=float, default=1e-3, required=False
)



def main(valid, data, bn="", threshold=0.0, verbose=False, absolute=1e-3, slient=False, level="info"):
    if not bn != "":
        print("{}".format(bn))

    if valid.size == 1 or data.size == 1:
        print("scalar", valid, data)
    elif valid.shape != data.shape:
        print(
            colored(
                "shape not identical: {} vs {}".format(valid.shape, data.shape),
                "red",
            )
        )
        return
    else:
        if verbose and slient is False:
            print("shape: {} ".format(data.shape))
    if np.count_nonzero(valid) == np.count_nonzero(data) and np.allclose(
        valid, data, rtol=1e-04, atol=absolute
    ):
        if np.array_equal(valid, data):
            if slient is False:
                print(colored("{} identical".format(bn), "green"))
        else:
            if slient is False:
                print(
                    colored(
                        "{} allclose at atol: {}".format(bn, absolute),
                        "blue",
                    )
                )

        return
    else:
        print(
            colored("{} not close at atol: {}".format(bn, absolute), "red")
        )
    if verbose is False:
        return
    diff = valid - data
    relative = diff / (np.abs(valid) + 10e-30)
    argmax = np.argmax(np.abs(diff))
    relative_argmax = np.argmax(relative)

    if np.abs(np.max(relative)) < threshold:
        return

    def fp(metric, v, d=None, c=None):
        if d is None:
            d = ""
        txt = "{:>20} {:<20} {:<20}".format(metric, v, d)
        if c is not None:
            txt = colored(txt, c)
        print(txt)

    fp("metric", "x0", "x1")
    fp("sum", valid.sum(), data.sum())
    fp("abs sum", np.absolute(valid).sum(), np.absolute(data).sum())
    fp("nonzero", np.count_nonzero(valid), np.count_nonzero(data))
    fp("abs diff sum", np.absolute(valid - data).sum())
    fp("diff sum", (diff).sum())
    fp(
        "argmax@{}".format(argmax),
        valid.reshape(-1)[argmax],
        data.reshape(-1)[argmax],
    )
    fp(
        "rltv argmax@{}".format(relative_argmax),
        valid.reshape(-1)[relative_argmax],
        data.reshape(-1)[relative_argmax],
    )

    fp("rltv max", np.abs(np.max(relative)), c="blue")
    fp("rltv sum", np.sum(np.abs(diff / (np.abs(valid) + 10e-20))))
    fp("diff mean", np.mean(np.abs(diff)))
    fp("diff max", np.max(np.abs(diff)))
    fp("diff std", np.std(np.abs(diff)))


if __name__ == "__main__":
    import itertools
    args = parser.parse_args()
    args_verbose = args.verbose
    for f1, f2 in itertools.combinations(args_file, 2):
        import os

        if slient is False:
            print("x" + str(0) + '=np.load("{}")'.format(f1.name))
            print("x" + str(1) + '=np.load("{}")'.format(f2.name))
        main(np.load(f1), np.load(f2), "blob", verbose=args_verbose, absolute=args.absolute)
        if args.slient is False:
            print("")
