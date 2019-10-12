./clear.sh
CUDA_VISIBLE_DEVICES=2                                                         \
/usr/local/cuda/bin/nvprof -o prof.nvvp python ./tools/train_net.py             \
       --config-file "./configs/customized_e2e_mask_rcnn_R_50_FPN_1x.yaml" 	\
       --skip-test
