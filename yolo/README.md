### Requirements
- python 3.6
- pytorch 0.4.0
### Features
- Include both Yolov2 and Yolov3
- Good performance

|544x544 |VOC2007 Test(mAP)|
| ------------ | ------------ |

| Yolov2  | 77.6% |

| Yolov3  | 79.6% |

- Train as fast as darknet

### Preparation
##### 1) Code
`git clone https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet`

`cd ObjectDetection-OneStageDet/yolo`

`yolo_root=$(pwd)`

`cd ${yolo_root}/utils/test`

`make -j32`

##### 2) Data
`wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`

`wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar`

`wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar`

`tar xf VOCtrainval_11-May-2012.tar`

`tar xf VOCtrainval_06-Nov-2007.tar`

`tar xf VOCtest_06-Nov-2007.tar`

`cd VOCdevkit`

`VOCdevkit_root=$(pwd)`

There will now be a VOCdevkit subdirectory with all the VOC training data in it.

`mkdir ${VOCdevkit_root}/onedet_cache`

`cd ${yolo_root}`

open examples/labels.py, let the variable `ROOT` point to `${VOCdevkit_root}`

`python examples/labels.py` 

open cfgs/yolov2.yml, let the `data_root_dir` point to `${VOCdevkit_root}/onedet_cache`

open cfgs/yolov3.yml, let the `data_root_dir` point to `${VOCdevkit_root}/onedet_cache`

##### 3) weights
Download model weights from [baiduyun](https://pan.baidu.com/s/1a3Z5IUylBs6rI-GYg3RGbw), and move all the model weights to "${yolo_root}/weights" directory.

### Training
##### 1) Yolov2
`cd ${yolo_root}`

1.1) open cfgs/yolov2.yml, let the `weights` of `train` block point to the pretrain weights

1.2) open cfgs/yolov2.yml, let the `gpus` of `train` block point to an available gpu id

1.3) run

`python examples/train.py Yolov2`

##### 2) Yolov3
2.1) open cfgs/yolov3.yml, let the `weights` of `train` block point to the pretrain weights

2.2) open cfgs/yolov3.yml, let the `gpus`  of `train` block point to an available gpu id

2.3) run

`python examples/train.py Yolov3`

#### 3) Results
The logs and weights will be in `${yolo_root}/outputs`.

### Evaluation
##### 1) Yolov2
1.1) open cfgs/yolov2.yml, let the `gpus` of `test` block point to an available gpu id

1.2) run

`python examples/train.py Yolov2`

##### 2) Yolov3
2.1) open cfgs/yolov3.yml, let the `gpus` of `test` block point to an available gpu id

2.2) run

`python examples/train.py Yolov3`

##### 3) Results
The output bbox will be in `${yolo_root}/results`,  every line of the file in   `${yolo_root}/results` has a format like `img_name confidence xmin ymin xmax ymax`

### Credits
I got a lot of code from [lightnet](https://gitlab.com/EAVISE/lightnet), thanks to EAVISE.