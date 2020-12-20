# Machine learning
Facilities for data preparation and training a custom object detection model.

## Training
Runs training and evaluation at the same time. View training metrics and losses in tensorboard.

Possibly due to memory constraints, my machine cannot barely handle this. So this training process is implemented sequentially, so nothing is running in parallel. Loading data into GPU memory is done in a separate thread to clean up GPU memory between each task (doesn't seem to be any easier way to do this).

Repeatedly:
1. train for 1000 steps (default)
2. save the checkpoint
3. evaluate the last checkpoint

### Files
- train_and_eval_tf2.py
- exporter_main_v2.py
- tensorboard
- train_t2.py (used to verify pipeline.config)

###  Commands
```
# train and eval at the same time
python .\train_and_eval_tf2.py --model_dir .\models\d0 --pipeline_config_path .\models\d0\pipeline.config

# view progress
tensorboard --logdir=models/d1

# export model
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\d1\pipeline.config --trained_checkpoint_dir .\models\ssdresnet191 
--output_directory .\models\ssdresnet191\exported
```

## Data preparation
given images organized by labels, prepare for training by
1. generating the label map file
2. converting images to jpg
3. resize the images for memory efficiency
4. generate tf records

inputs
datasetdir
gendir
pretrained coco model
split %
max from each label
resize min dimension

dataset directory structure - files are expected to be organized in folders by their labels
eg.
- dataset
  - label1
  - label2
  - ...

output
- gendir
  - label_map.pbtxt
  - train.tfrecords
  - test.tfrecords
  - jpg
    - label1
    - label2
    - ...

### Files
- prepare_data.py

### Commands
```
python .\prepare_data.py --dataset .\data\ --output out --coco .\models\efficientdet_d1_coco17_tpu-32\saved_model\ --maxperlabel 350 --mindim 640 --equalcounts
```

## Evaluation
Run a saved_model on a tfrecord and generate the confusion matrix

### Files
- confuse_the_matrix.py

### Commands
```
python .\confuse_the_matrix.py --tfrecord .\out640\test.tfrecords --model '.\models\ssdresnet640\exported\saved_model'
```

# Application service
A service to run object detection on a video feed and dispense treats based on a state machine.

## Video feed server
Runs a server that can be connected on a phone to send a camera feed over. The server is the controller to send commands to the phone and dispenser.
(only tested on iphone and safari)

### Files
- dispenserver/server.py

### Commands
```
python .\dispenserver\server.py --cert-file .\cert.pem --key-file .\key.pem --port 8443

# generating cert.pem and key.pem files
openssl req -newkey rsa:2048 -new -nodes -x509 -days 3650 -keyout key.pem -out cert.pem
```

## Dispenser server
Runs a local http server that connects to NXT over bluetooth. On any GET request, dispense a treat by turning the motor in PORT_A.

Turn on NXT: orange button
Turn off NXT: grey button below, then orange button to confirm

### Files
- dispenserver/dispenser.py

### Commands
```
# run the dispenser server
python .\dispenserver\dispenser.py

# calibrate power and tacho params
python .\dispenserver\dispenser.py calibrate

# test server without connecting to bluetooth
python .\dispenserver\dispenser.py test

# kill process using port 8080 (not sure why it hangs sometimes.. ¯\_(ツ)_/¯)
netstat -ano | findstr :8080
taskkill /PID {last column from netstat} /F
```

# Requirements
For machine learning and video feed server:
python==3.8
tensorflow==2.3.1

For dispenser server and NXT:
python==2.7
pybluez==2.3

# Resources
Tensorflow:
- https://www.tensorflow.org/api_docs/python/tf/all_symbols
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

Object detection:
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Data prep:
- https://www.tensorflow.org/tutorials/load_data/tfrecord
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
- https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/

WebRTC:
- https://github.com/aiortc/aiortc
- https://github.com/aiortc/aiortc/tree/main/examples/server

NXT:
- https://github.com/Eelviny/nxt-python/tree/python2
- https://github.com/pybluez/pybluez/issues/180#issuecomment-457071444
  - https://1drv.ms/u/s!AtLn8ELpA_G9frRzQuMKJ1QixKw
