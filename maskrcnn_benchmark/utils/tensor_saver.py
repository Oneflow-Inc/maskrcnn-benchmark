import os
import numpy
import pickle as pk

class TensorSaver(object):
    def __init__(self, training, base_dir, iteration, max_iter):
        self.training = training
        self.base_dir = base_dir
        self.iteration = iteration
        if max_iter:
            self.max_iteration = max_iter
        else:
            self.max_iteration = 0

    def step(self, iteration=None):
        if iteration:
            self.iteration = iteration
        else:
            self.iteration += 1

    def save(
        self,
        tensor,
        tensor_name,
        scope=None,
        save_grad=False,
        level=None,
        im_idx=None,
    ):
        if self.iteration > self.max_iteration:
            return

        save_dir = os.path.join(self.base_dir, "iter_{}".format(self.iteration))
        if scope:
            save_dir = os.path.join(save_dir, scope)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        suffix = ""
        if isinstance(im_idx, int):
            suffix = suffix + ".image{}".format(im_idx)
        if isinstance(level, int):
            suffix = suffix + ".layer{}".format(level)
        suffix = suffix + "." + str(tuple(tensor.size()))

        save_path = os.path.join(save_dir, "{}{}".format(tensor_name, suffix))
        numpy.save(save_path, tensor.cpu().detach().numpy())

        if save_grad and self.training:
            grad_save_path = os.path.join(
                save_dir, "{}_grad{}".format(tensor_name, suffix)
            )
            tensor.register_hook(
                lambda grad: numpy.save(
                    grad_save_path, grad.cpu().detach().numpy()
                )
            )


tensor_saver = None


def create_tensor_saver(training, base_dir, iteration=0, max_iter=None):
    global tensor_saver
    tensor_saver = TensorSaver(training, base_dir, iteration, max_iter)


def get_tensor_saver():
    global tensor_saver
    if not tensor_saver:
        raise Exception("Tensor saver not created yet")

    return tensor_saver


def dump_data(iter, images, targets, image_id):
    from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import (
        project_masks_on_boxes,
    )

    data = {}
    data["image_id"] = str(image_id)
    data["images"] = images.tensors.detach().numpy()
    print(type(data["images"]))
    print(data["images"].shape)

    data["gt_bbox"] = []
    data["gt_labels"] = []
    data["gt_segm"] = []
    data["image_size"] = []
    for box_list in targets:
        data["gt_bbox"].append(box_list.bbox.detach().numpy())
        data["gt_labels"].append(
            numpy.array(box_list.get_field("labels"), dtype=numpy.int32)
        )
        segm_mask = project_masks_on_boxes(
            box_list.get_field("masks"), box_list, 28
        )
        data["gt_segm"].append(segm_mask.detach().numpy().astype(numpy.int8))
        data["image_size"].append(numpy.array(box_list.size, dtype=numpy.int32))

    data["image_size"] = numpy.stack(data["image_size"], axis=0)

    dump_dir = os.path.join("train_dump", "iter_{}".format(iter))
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    with open(os.path.join(dump_dir, "data.pkl"), "wb") as f:
        pk.dump(data, f, protocol=2)
