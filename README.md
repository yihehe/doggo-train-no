# commands
```
python .\prepare_data.py --dataset ..\data\ --output out --coco ..\training\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8\saved_model\\ --maxperlabel 10
```

# what do
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

# requirements
python3.8

# resources
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md