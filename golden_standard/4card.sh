# export MASKRCNN_BENCHMARK_PATH=/home/zhangwenxiao/repos/maskrcnn-benchmark
# export CONFIG_FILE=./e2e_mask_rcnn_R_50_FPN_16.yaml
set -x
bash ./clear.sh
rm -rf output
rm -rf *.csv
export CONFIG_FILE=/home/caishenghang/maskrcnn-benchmark/golden_standard/mrcn_4x_train.yaml
export NGPUS=4
#export CUDNN_LOGINFO_DBG=1
#export CUDNN_LOGDEST_DBG="cudnn.log"
# /home/caishenghang/docker-mask-nsight-systems/2019.4.2/bin/nsys  profile --force-overwrite true -t cuda -o mask-pytorch-debug-slow 
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    /home/caishenghang/maskrcnn-benchmark/tools/train_net.py \
        --config-file $CONFIG_FILE \
        --skip-test
