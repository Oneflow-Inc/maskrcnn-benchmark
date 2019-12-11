# export MASKRCNN_BENCHMARK_PATH=/home/zhangwenxiao/repos/maskrcnn-benchmark
# export CONFIG_FILE=./e2e_mask_rcnn_R_50_FPN_16.yaml
set -x
bash ./clear.sh
rm -rf output_v2
rm -rf *.csv
export CONFIG_FILE=/home/caishenghang/pytorch_mrcn/4-card.yaml
export NGPUS=4
#export CUDNN_LOGINFO_DBG=1
#export CUDNN_LOGDEST_DBG="cudnn.log"
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    /home/caishenghang/maskrcnn-benchmark/tools/train_net.py \
        --config-file $CONFIG_FILE \
        --skip-test
