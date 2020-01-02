set -x
iter=600
# /home/caishenghang/wksp/flow-nan-model-500-520
python /home/caishenghang/maskrcnn-benchmark/scripts/compare_model.py \
    --py /home/caishenghang/maskrcnn-benchmark/golden_standard/output/model_0000$iter.pth \
    --flow /home/caishenghang/wksp/ \
    --map /home/caishenghang/maskrcnn-benchmark/scripts/mask_rcnn_R_50_FPN_1x.json 
