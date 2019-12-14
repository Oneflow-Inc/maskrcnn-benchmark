SRC_DIR="/home/zhangwenxiao/wksp/pytorch_maskrcnn_benchmark/train_dump"
DST_DIR="/home/zhangwenxiao/wksp/debug_mrcn/fake_images"
START=90000
END=90009
for i in $(seq $START $END); do 
    index=$(($i - $START))
    src=`realpath $SRC_DIR/iter_$i/image.npy`
    dst="$DST_DIR/image_$index.npy"
    cp "$src" "$dst"
    echo "copy '$src' to '$dst'"
done
