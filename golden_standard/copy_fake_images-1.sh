SRC_DIR="/home/caishenghang/maskrcnn-benchmark/golden_standard/train_dump"
DST_DIR="/home/caishenghang/wksp/fake_images"
START=0
END=10
for i in $(seq $START $END); do 
    index=$(($i - $START))
    src=`realpath $SRC_DIR/iter_$i/image.npy`
    dst="$DST_DIR/image_$index.npy"
    cp "$src" "$dst"
    echo "copy '$src' to '$dst'"
done
