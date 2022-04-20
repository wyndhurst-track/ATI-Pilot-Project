DATA=darknet-alexeyAB/cfg/detector_augmented.data
NET=darknet-alexeyAB/cfg/yolov3-detector.cfg
WEIGHTS=darknet-alexeyAB/data/darknet53.conv.74

./darknet-alexeyAB/darknet detector train $DATA $NET $WEIGHTS -map
