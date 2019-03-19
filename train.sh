rm -r ./dump
rm -r ./new_dump
rm -r ./grad_dump
rm -r ./param_grad
rm last_checkpoint
rm model_final.pth
rm log.txt
rm model_0090000.pth
rm e2e_mask_rcnn_R_50_FPN_1x.pth.model_name2momentum_buffer.pkl
# export NGPUS=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#        ./tools/train_net.py\
#        --config-file "./configs/customized_e2e_mask_rcnn_R_50_FPN_1x_all.yaml" \
#        --skip-test
python ./tools/train_net.py\
       --config-file "./configs/customized_e2e_mask_rcnn_R_50_FPN_1x_all.yaml" \
       --skip-test
