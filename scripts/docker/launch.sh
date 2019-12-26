installed="/workspace/object_detection/maskrcnn_benchmark/"
model_dir=$PWD/model_dir
mkdir $model_dir
extra=""
extra="$extra -v /model_zoo:/model_zoo"
extra="$extra -v /dataset:/dataset"
extra="$extra -v $model_dir:/root/.torch/models"

extra="$extra -v $PWD:/maskrcnn_benchmark"
extra="$extra --workdir /maskrcnn_benchmark"
extra="$extra -v $HOME:$HOME"
# extra="$extra --user $(id -u):$(id -g)"

prelude=""
# prelude="$prelude cp $installed/_C.* /maskrcnn_benchmark;"
# prelude="$prelude python setup.py develop;"

docker run -it --rm --ipc=host \
    $extra \
    py_mask-rcnn \
    bash -c "$prelude bash"
