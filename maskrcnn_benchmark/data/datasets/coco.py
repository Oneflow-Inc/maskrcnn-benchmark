# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self,
        ann_file,
        root,
        remove_images_without_annotations,
        transforms=None,
        use_contiguous_category_id=True,
    ):
        super(COCODataset, self).__init__(root, ann_file)

        # remove imgs with category_id > 80
        to_remove = set([])
        for cat_id, _ in self.coco.cats.items():
            if cat_id > 80:
                to_remove |= set(self.coco.catToImgs[cat_id])
        self.ids = list(set(self.ids) - to_remove)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.use_contiguous_category_id = use_contiguous_category_id
        print("coco dataset first 20 image_ids: ", self.ids[0:20])

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        image_id = anno[0]["image_id"]

        if idx < 1:
            print("save image {}, size {} to png".format(image_id, img.size))
            torchvision.utils.save_image(
                torchvision.transforms.functional.to_tensor(img),
                "{:012d}.png".format(image_id),
            )
            np.save("raw_img_{}".format(image_id), torchvision.transforms.functional.to_tensor(img))

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        if self.use_contiguous_category_id:
            classes = [
                self.json_category_id_to_contiguous_id[c] for c in classes
            ]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode="poly")
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        print(
            "coco __getitem__ idx: {}, image_id: {}, anno_len: {}".format(
                idx, image_id, len(anno)
            )
        )
        # print("coco __getitem__ anno example: ", anno[0])
        return img, target, image_id

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
