set -x
python /home/caishenghang/maskrcnn-benchmark/scripts/compare_model.py \
    --py /home/caishenghang/maskrcnn-benchmark/golden_standard/output/model_0000520.pth \
    --flow /home/caishenghang/wksp/ \
    --map /home/caishenghang/maskrcnn-benchmark/scripts/mask_rcnn_R_50_FPN_1x.json
