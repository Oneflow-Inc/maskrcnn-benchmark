import os
import numpy
import pickle as pk


class TensorSaver(object):
    def __init__(self, training, base_dir, iteration, max_iter, save_shape=False):
        self.training = training
        self.base_dir = base_dir
        self.iteration = iteration
        if max_iter:
            self.max_iteration = max_iter
        else:
            self.max_iteration = 0
        self.save_shape = save_shape
        self.disable_save = False

    def step(self, iteration=None):
        if iteration:
            self.iteration = iteration
        else:
            self.iteration += 1

    def disable(self):
        self.disable_save = True

    def enable(self):
        self.disable_save = False

    def save(
        self,
        tensor,
        tensor_name,
        scope=None,
        save_grad=False,
        level=None,
        im_idx=None,
    ):
        if self.disable_save:
            return

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
        if self.save_shape:
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


class OfflineTensorSaver(TensorSaver):
    def save(
        self,
        tensor,
        tensor_name,
        scope=None,
        save_grad=False,
        level=None,
        im_idx=None,
    ):
        pass


tensor_saver = None


def create_tensor_saver(
    training, base_dir, iteration=0, max_iter=None, save_shape=False, enable_save=False
):
    global tensor_saver
    if enable_save:
        tensor_saver = TensorSaver(training, base_dir, iteration, max_iter, save_shape)
    else:
        tensor_saver = OfflineTensorSaver(training, base_dir, iteration, max_iter, save_shape)


def get_tensor_saver():
    global tensor_saver
    if not tensor_saver:
        raise Exception("Tensor saver not created yet")

    return tensor_saver


class MockDataMaker:
    def __init__(self, start_iter=1, enabled=True):
        self.iter_ = start_iter
        self.data_ = {}
        self.enabled_ = enabled

    def update_image(self, image_id, images):
        if not self.enabled_: return
        self.data_["image_id"] = str(image_id)
        self.data_["images"] = images.tensors.detach().numpy()

    def update_target(self, targets):
        if not self.enabled_: return
        self.data_["image_size"] = []
        self.data_["gt_bbox"] = []
        self.data_["gt_labels"] = []
        self.data_["gt_segm_poly"] = []
        self.data_["gt_segm_mask"] = []

        for box_list in targets:
            self.data_["gt_bbox"].append(box_list.bbox.detach().numpy())
            self.data_["gt_labels"].append(
                numpy.array(box_list.get_field("labels"), dtype=numpy.int32)
            )
            poly_list = box_list.get_field("masks").convert("poly").instances
            img_polys = []
            for poly_inst in poly_list:
                obj_polys = []
                for polys in poly_inst.polygons:
                    obj_polys.append(numpy.array(polys, dtype=numpy.double))
                img_polys.append(obj_polys)
            self.data_["gt_segm_poly"].append(img_polys)
            self.data_["gt_segm_mask"].append(
                box_list.get_field("masks")
                .get_mask_tensor()
                .detach()
                .numpy()
                .astype(numpy.int8)
            )
            self.data_["image_size"].append(
                numpy.array(box_list.size, dtype=numpy.int32)
            )

        self.data_["image_size"] = numpy.stack(self.data_["image_size"], axis=0)

    def update_mask_targets(self, mask_targets):
        if not self.enabled_: return
        self.data_["segm_mask_targets"] = mask_targets.cpu().detach().numpy()

    def save(self):
        if not self.enabled_: return
        dump_dir = os.path.join("train_dump", "iter_{}".format(self.iter_))
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        with open(os.path.join(dump_dir, "mock_data.pkl"), "wb") as f:
            pk.dump(self.data_, f, protocol=2)

    def step(self):
        self.iter_ += 1


def dump_data(iter, images, targets, image_id):
    get_mock_data_maker().update_image(image_id, images)
    get_mock_data_maker().update_target(targets)


mock_data_maker = None


def create_mock_data_maker(iter, enable=True):
    global mock_data_maker
    mock_data_maker = MockDataMaker(iter, enable)


def get_mock_data_maker():
    global mock_data_maker
    if not mock_data_maker:
        raise Exception("mock_data_maker not created yet")

    return mock_data_maker
