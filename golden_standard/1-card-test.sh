# export CONFIG_FILE=./e2e_mask_rcnn_R_50_FPN_test.yaml
set -x
bash ./clear.sh
rm -rf output
rm -rf *.csv
export CONFIG_FILE=/home/caishenghang/maskrcnn-benchmark/golden_standard/mrcn_1x_train_init.test.yaml
CUDA_VISIBLE_DEVICES=2 \
python /home/caishenghang/maskrcnn-benchmark/tools/train_net.py --config-file $CONFIG_FILE --skip-test
# bash ./copy_fake_images-1.sh
