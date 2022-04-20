DATA=darknet-alexeyAB/cfg/detector_augmented.data
NET=darknet-alexeyAB/cfg/yolov3-detector.cfg
WEIGHTS=darknet-alexeyAB/backup/detector_augmented/yolov3-detector_best.weights

./darknet-alexeyAB/darknet detector demo $DATA $NET $WEIGHTS -c 0