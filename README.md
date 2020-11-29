# Training
train_and_eval_tf2.py
exporter_main_v2.py
tensorboard
train_t2.py (used to verify pipeline.config)

##  Commands
```
# train and eval at the same time
python .\train_and_eval_tf2.py --model_dir .\models\d1 --pipeline_config_path .\models\d1\pipeline.config

# view progress
tensorboard --logdir=models/d1

# export model
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\d1\pipeline.config --trained_checkpoint_dir .\models\ssdresnet191 
--output_directory .\models\ssdresnet191\exported
```

## what do
Runs training and evaluation at the same time. View training metrics and losses in tensorboard.

Possibly due to memory constraints, my machine cannot barely handle this. So this training process is implemented sequentially, so nothing is running in parallel. Loading data into GPU memory is done in a separate thread to clean up GPU memory between each task (doesn't seem to be any easier way to do this).

Repeatedly:
1. train for 1000 steps (default)
2. save the checkpoint
3. evaluate the last checkpoint

# Data preparation
prepare_data.py

## commands
```
python .\prepare_data.py --dataset .\data\ --output out --coco .\models\efficientdet_d1_coco17_tpu-32\saved_model\ --maxperlabel 350 --mindim 640 --equalcounts --shards=10
```

## what do
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

# Requirements
python==3.8
tensorflow==2.3.1

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
