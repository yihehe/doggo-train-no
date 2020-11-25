import tensorflow as tf
import numpy as np
import glob
from tensorflow.keras.preprocessing.image import *
import matplotlib.pyplot as plt
import IPython.display as display
from object_detection.utils import dataset_util
import os
import io
from PIL import Image
import random
from object_detection.utils import label_map_util
from pathlib import Path

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from IPython.display import display


pretrained_model = tf.saved_model.load(
    'training/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model')
label_map = label_map_util.load_labelmap(
    'training/annotations/pose_label_map.pbtxt')
label_map_dict = label_map_util.get_label_map_dict(label_map)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    return output_dict


def convertJpg(image_path, label):
    filename = os.path.basename(image_path)

    new_file = Path(gen_dir + label + '/' + filename)
    dest_path = new_file.with_suffix('.jpg')

    if Path.exists(dest_path):
        return str(dest_path)

    Path.mkdir(dest_path.parent, parents=True, exist_ok=True)

    im = Image.open(image_path)

    resize = 300
    width, height = im.size
    if width < height:
        wpercent = (resize/float(width))
        hsize = int((float(height)*float(wpercent)))
        im = im.resize((resize, hsize), Image.ANTIALIAS)
    else:
        hpercent = (resize/float(height))
        wsize = int((float(width)*float(hpercent)))
        im = im.resize((wsize, resize), Image.ANTIALIAS)

#     im = im.resize((im.size[0]//2, im.size[1]//2))
    rgb_im = im.convert('RGB')
    rgb_im.save(dest_path, 'jpeg')

    return str(dest_path)


def toExampleTf(image_path, label):
    path = convertJpg(image_path, label)
    print(path)

    with tf.io.gfile.GFile(path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    filename = os.path.basename(path).encode('utf8')
    image_format = b'jpg'

    image_np = np.array(image)
    output_dict = run_inference_for_single_image(pretrained_model, image_np)
    if int(output_dict['detection_classes'][0][0]) != 18:
        print('NOT A DOG')
        print(output_dict['detection_classes'][0][:4])
        index = label_map_util.create_category_index_from_labelmap(
            'models/research/object_detection/data/mscoco_label_map.pbtxt', use_display_name=True)

        num_detections = int(output_dict.pop('num_detections'))
        num2 = int(100)
        #   print(num_detections == num2)
        output_dict = {key: value[0, :num2].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num2

        #   print(output_dict['detection_boxes'][:2])

        #   b0=output_dict['detection_boxes'][0]
        #   b1= output_dict['detection_boxes'][1]
        #   print(output_dict['detection_boxes'][:2])
        #   output_dict['detection_boxes'][0] = output_dict['detection_boxes'][3]
        #   output_dict['detection_boxes'][0,0] = 0

        #   print('output!', output_dict['detection_boxes'][:1])

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(
            np.int64)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            index,
            instance_masks=None,
            use_normalized_coordinates=True,
            line_thickness=8)

        display(Image.fromarray(image_np))
        return None
    ymin, xmin, ymax, xmax = output_dict['detection_boxes'][0][0]
    xmins = [xmin.numpy()]
    xmaxs = [xmax.numpy()]
    ymins = [ymin.numpy()]
    ymaxs = [ymax.numpy()]

    classes_text = [label.encode('utf8')]
    classes = [label_map_dict[label]]

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))


def partition(paths, num, ratio, label):
    ids = list(range(0, len(paths)))
    random.shuffle(ids)

    split = int(num*ratio)
    train_ids = ids[:split]
    test_ids = ids[split:num]

    print(train_ids)
    print(test_ids)

    train_tfs = []
    test_tfs = []
    for idx in train_ids:
        ex = toExampleTf(paths[idx], label)
        if ex != None:
            train_tfs.append(ex)
        else:
            print('skipping')

    for idx in test_ids:
        ex = toExampleTf(paths[idx], label)
        if ex != None:
            test_tfs.append(ex)
        else:
            print('skipping')

    return train_tfs, test_tfs


data_dir = "Dog photos for Yi/"
stand_photos = glob.glob(data_dir + 'Stand/' + '*')
down_photos = glob.glob(data_dir + 'Down/' + '*')
paw_photos = glob.glob(data_dir + 'Paw/' + '*')
# sit_photos = glob.glob(data_dir + 'Sit/' + '*')

gen_dir = 'training/annotations/gen2/'
tfrecordtrain = gen_dir + 'train.tfrecords'
tfrecordtest = gen_dir + 'test.tfrecords'

train_tfs_stand, test_tfs_stand = partition(stand_photos, 100, 0.85, 'stand')
train_tfs_down, test_tfs_down = partition(down_photos, 100, 0.85, 'down')
train_tfs_paw, test_tfs_paw = partition(paw_photos, 100, 0.85, 'paw')

train_tfs = []
train_tfs.extend(train_tfs_stand)
train_tfs.extend(train_tfs_down)
train_tfs.extend(train_tfs_paw)

# num_shards=10
# with contextlib2.ExitStack() as tf_record_close_stack:
#     output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, tfrecordtrain, num_shards)
#     for idx, ex in enumerate(train_tfs):
#         output_shard_index = idx % num_shards
#         output_tfrecords[output_shard_index].write(ex.SerializeToString())

with tf.io.TFRecordWriter(tfrecordtrain) as writer:
    for ex in train_tfs:
        writer.write(ex.SerializeToString())

test_tfs = []
test_tfs.extend(test_tfs_stand)
test_tfs.extend(test_tfs_down)
test_tfs.extend(test_tfs_paw)

with tf.io.TFRecordWriter(tfrecordtest) as writer:
    for ex in test_tfs:
        writer.write(ex.SerializeToString())

print(len(train_tfs), len(test_tfs))
