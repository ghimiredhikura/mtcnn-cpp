# MTCNN CPP Face Detection and Evaluation

This repository contains MTCNN CPP implementation. Advandages of this implementations is it gives real time runtime performance in CPU.  

### 1. Usage

At first you need to install OpenCV2.0.x+, and CAFFE

```
$git clone https://github.com/ghimiredhikura/mtcnn-cpp.git
$cd mtcnn-cpp
$mkdir build
$cd build/
$cmake ..
$make 
```
#### 1. Test webcam
```
$./mtcnn-cpp -mode=0 -webcam=0
```
#### 2. Test single image
```
$./mtcnn-cpp -mode=1 -path=../image/1.jpg
```
#### 3. Test image lists
```
$./mtcnn-cpp -mode=2 -path=../image/
```
#### 4. Evaluation in benchmark dataset, detection files will be stored in "detections" folder. 
```
a) afw dataset
$./mtcnn-cpp -mode=3 -dataset=AFW -path=/path/to/afw/dataset/

b) PASCAL dataset
$./mtcnn-cpp -mode=3 -dataset=PASCAL -path=/path/to/pascal/dataset/

c) FDDB dataset
$./mtcnn-cpp -mode=3 -dataset=FDDB -path=/path/to/fddb/dataset/

d) WIDER_val dataset
#./mtcnn-cpp -mode=3 -dataset=WIDER_VAL -path=/path/to/wider/validation/dataset/

e) UFDD dataset
#./mtcnn-cpp -mode=3 -dataset=UFDD -path=/path/to/UFDD/validation/dataset/
```

### 2. Evaluation results in benchmark datasets

Dataset download: Please refer to this [git](https://github.com/bonseyes/SFD/blob/master/docs/Test-Instructions.md) for downloading dataset and evaluation tools. 

#### a. [AFW](http://www.ics.uci.edu/~xzhu/face/), PASCAL ([train-val](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), [test](http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/)) and [FDDB](http://vis-www.cs.umass.edu/fddb/index.html)

#### b. [WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

### 3. References:

1. [MTCNN - Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
3. [MTCNN Cpp](https://github.com/golunovas/mtcnn-cpp)
4. [S³FD: Single Shot Scale-invariant Face Detector](https://github.com/bonseyes/SFD)
