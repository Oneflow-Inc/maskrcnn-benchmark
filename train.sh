rm -rf ./dump
rm -rf ./new_dump
rm -rf ./grad_dump
rm -rf ./param_grad
rm -f last_checkpoint
rm -f model_final.pth
rm -f log.txt
rm -f model_0090000.pth
rm -rf model_name2momentum_buffer

CUDA_VISIBLE_DEVICES=1                                                          \
python ./tools/train_net.py                                                     \
       --config-file "./configs/customized_e2e_mask_rcnn_R_50_FPN_1x_all.yaml"  \
       --skip-test
