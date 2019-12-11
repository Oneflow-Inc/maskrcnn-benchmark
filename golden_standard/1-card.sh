# export CONFIG_FILE=./e2e_mask_rcnn_R_50_FPN_test.yaml
set -x
bash ./clear.sh
rm -rf output_v2
rm -rf *.csv
export CONFIG_FILE=/home/caishenghang/pytorch_mrcn/1-card.yaml
CUDA_VISIBLE_DEVICES=0 \
python /home/caishenghang/maskrcnn-benchmark/tools/train_net.py --config-file $CONFIG_FILE --skip-test
