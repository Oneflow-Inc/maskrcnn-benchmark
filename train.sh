rm -r ./dump
python ./tools/train_net.py \
       --config-file "./configs/pytorch_mask_rcnn_benchmark_R_50_FPN_1x.yaml"
