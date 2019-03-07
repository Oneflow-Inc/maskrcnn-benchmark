rm -r ./dump
rm last_checkpoint
rm model_final.pth
rm log.txt
# export NGPUS=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#        ./tools/train_net.py\
#        --config-file "./configs/customized_e2e_mask_rcnn_R_50_FPN_1x_all.yaml" \
#        --skip-test
python ./tools/train_net.py\
       --config-file "./configs/customized_e2e_mask_rcnn_R_50_FPN_1x_all.yaml" \
       --skip-test
