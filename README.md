# YOLO-from-Scratch-

YOLOV2/ YOLOV9000 paper: https://arxiv.org/pdf/1612.08242.pdf 

## All the important points: 

YOLO V1 has relatively low recall compared to region proposal-based methods. Thus YOLOV2  focuses mainly on improving recall and localization while maintaining classification accuracy.

The original YOLO trains the classifier network at 224 × 224 and increases the resolution to 448 for detection. 
For YOLOv2 we first fine tune the classification network at the full 448 × 448 resolution for 10 epochs on ImageNet. This gives the network time to adjust its filters to work better on higher resolution input. We then fine tune the resulting network on detection. This high resolution classification network gives us an increase of almost 4% mAP.

We remove the fully connected layers from YOLO and use anchor boxes to predict bounding boxes. First we eliminate one pooling layer to make the output of the network’s convolutional layers higher resolution. 

We also shrink the network to operate on 416 input images instead of 448×448. We do this because we want an odd number of locations in our feature map so there is a single center cell.

Using anchor boxes we get a small decrease in accuracy. YOLO only predicts 98 boxes per image but with anchor boxes our model predicts more than a thousand. 

Dimension Clusters. We encounter two issues with anchor boxes when using them with YOLO. The first is that the box dimensions are hand picked. The network can learn to adjust the boxes appropriately but if we pick better priors for the network to start with we can make it easier for the network to learn to predict good detections. Instead of choosing priors by hand, we run k-means clustering on the training set bounding boxes to automation find good priors. If we use standard k-means with Euclidean distance larger boxes generate more error than smaller boxes. However, what we really want are priors that lead to good IOU scores, which is independent of the size of the box. 

d(box, centroid) = 1 − IOU(box, centroid) → distance metric formula 

When using anchor boxes with YOLO we encounter a second issue: model instability, especially during early iterations.  

Instead of fixing the input image size we change the network every few iterations. Every 10 batches our network randomly chooses a new image dimension size. Since our model downsamples by a factor of 32, we pull from the following multiples of 32: {320, 352, ..., 608}. Thus the smallest option is 320 × 320 and the largest is 608 × 608. We resize the network to that dimension and continue training.

All the config file which is needed for object detection and other image classification purpose. https://github.com/AlexeyAB/darknet/tree/master/cfg 

Darknet publish a various config which is easy to create a model for every algorithm like Yolov1, YoloV2, YoloV3, .... YoloV7 .

Darknet is used in YOLO for better accuracy and fast prediction which can be used real world scenerios. 
VGG-16 requires 30.69 billion floating point operations for a single pass over a single image at 224 × 224 resolution, Google net requires 8.52 billion operations for a forward pass, while Darknet-19 only requires 5.58 billion operations.

When we analyze YOL0’s performance on ImageNet we see it learns new species of animals well but struggles with learning categories like clothing and equipment.
