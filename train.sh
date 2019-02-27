rm -r ./dump
rm log.txt
# export NGPUS=4
# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#        ./tools/train_net.py\
#        --config-file "./configs/pytorch_mask_rcnn_benchmark_R_50_FPN_1x.yaml" \
#        --skip-test
nohup python ./tools/train_net.py\
       --config-file "./configs/customized_e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       --skip-test &
