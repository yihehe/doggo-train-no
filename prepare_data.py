from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
import sys
from functools import cached_property
import argparse
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

import PIL
import json
import cv2
import timeit

# tf2.4 on ampere OOM
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class CocoClassNotFound(Exception):
    pass


def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def dedupeFiles(files):
    print('[DOGGO] deduping files', len(files))
    unique_files = []

    hashes = {}
    for f in files:
        try:
            image = Image.open(f)
            h = dhash(np.array(image.convert('RGB')))
            if h not in hashes.keys():
                unique_files.append(f)
                hashes[h] = [f]
            else:
                hashes[h].append(f)
        except PIL.UnidentifiedImageError as e:
            print('ERROR:', e)

    return unique_files


def generate_and_persist_labels(labels, outdir):
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    for i, l in enumerate(labels):
        label_map_item = string_int_label_map_pb2.StringIntLabelMapItem(
            id=i+1, name=l)
        label_map.item.append(label_map_item)

    outfile = os.path.join(outdir, 'label_map.pbtxt')
    Path.mkdir(Path(outfile).parent, parents=True, exist_ok=True)

    with open(outfile, 'w') as f:
        f.write(text_format.MessageToString(label_map))

    return label_map_util.get_label_map_dict(label_map)


def writeTfRecords(examples, out, num_shards=None):
    if num_shards:
        num_shards = int(num_shards)
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, out, num_shards)
            for idx, ex in enumerate(examples):
                output_shard_index = idx % num_shards
                output_tfrecords[output_shard_index].write(
                    ex.SerializeToString())
    else:
        with tf.io.TFRecordWriter(out) as writer:
            for ex in examples:
                writer.write(ex.SerializeToString())


class JpgCache:
    def __init__(self, outdir, mindim=None):
        self.outdir = outdir
        self.mindim = int(mindim) if mindim else mindim

    def toJpg(self, image_path, label):
        # get jpg destination path
        filename = os.path.basename(image_path)
        jpg_path = Path(os.path.join(self.outdir, label,
                                     filename)).with_suffix('.jpg')

        # if already converted, skip
        if Path.exists(jpg_path):
            return str(jpg_path)

        Path.mkdir(jpg_path.parent, parents=True, exist_ok=True)

        im = Image.open(image_path)

        # resize if neccessary
        if self.mindim != None:
            width, height = im.size
            if width < height:
                wpercent = (self.mindim/float(width))
                hsize = int((float(height)*float(wpercent)))
                im = im.resize((self.mindim, hsize), Image.ANTIALIAS)
            else:
                hpercent = (self.mindim/float(height))
                wsize = int((float(width)*float(hpercent)))
                im = im.resize((wsize, self.mindim), Image.ANTIALIAS)

        # save and return path
        rgb_im = im.convert('RGB')
        rgb_im.save(jpg_path, 'jpeg')
        return str(jpg_path)


class TfGen:
    def __init__(self, label_map_dict, cocomodel, target_coco_class):
        self.label_map_dict = label_map_dict
        self.cocomodel = cocomodel
        self.target_coco_class = target_coco_class

    @cached_property
    def detection_model(self):
        print('[DOGGO] loading model...', self.cocomodel)
        return tf.saved_model.load(self.cocomodel)

    def run_inference_for_single_image(self, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        model_fn = self.detection_model.signatures['serving_default']
        output_dict = model_fn(input_tensor)
        return output_dict

    def find_coco_class(self, image):
        image_np = np.array(image)
        output_dict = self.run_inference_for_single_image(image_np)
        if int(output_dict['detection_classes'][0][0]) != self.target_coco_class:
            raise CocoClassNotFound('looking for {} but got {}'.format(
                self.target_coco_class, int(output_dict['detection_classes'][0][0])))

        # ymin, xmin, ymax, xmax
        return output_dict['detection_boxes'][0][0]

    def toExampleTf(self, jpg_path, label):
        print('[DOGGO] label {} example {}'.format(label, jpg_path))

        with tf.io.gfile.GFile(jpg_path, 'rb') as fid:
            encoded_image_data = fid.read()
        image = Image.open(io.BytesIO(encoded_image_data))

        ymin, xmin, ymax, xmax = self.find_coco_class(image)
        xmins = [xmin.numpy()]
        xmaxs = [xmax.numpy()]
        ymins = [ymin.numpy()]
        ymaxs = [ymax.numpy()]

        width, height = image.size

        filename = os.path.basename(jpg_path).encode('utf8')
        image_format = b'jpg'

        classes_text = [label.encode('utf8')]
        classes = [self.label_map_dict[label]]

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


class LabelGen:
    def __init__(self, label, files, tf_gen, jpg_cache):
        self.label = label
        self.files = files
        self.unique_files = dedupeFiles(files)
        self.tf_gen = tf_gen
        self.jpg_cache = jpg_cache

        self.ids = list(range(0, len(self.unique_files)))
        random.shuffle(self.ids)
        self.i = 0

    # can't use yield because we need to handle exceptions
    def next(self):
        idx = self.ids[self.i]
        self.i += 1

        jpg_path = jpg_cache.toJpg(self.unique_files[idx], self.label)
        return self.tf_gen.toExampleTf(jpg_path, self.label)

    def count(self):
        return len(self.ids)

    def total(self):
        return len(self.files)

    def skipCount(self):
        return len(self.files) - len(self.unique_files)

    def getLabel(self):
        return self.label


def partition(paths, num, ratio, label):
    ids = list(range(0, len(paths)))
    random.shuffle(ids)

    split = int(num*ratio)
    train_ids = ids[:split]
    test_ids = ids[split:num]

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


# parse args
parser = argparse.ArgumentParser(description='what is my purpose?')

required = parser.add_argument_group('required arguments')
required.add_argument('--dataset', required=True,
                      help='directory that contains the dataset')
required.add_argument('--output', required=True,
                      help='directory where the generated files will go')
required.add_argument('--coco', required=True,
                      help='directory of the pretrained coco model (eg ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model')

parser.add_argument('--split', default=0.8,
                    help='what percentage of dataset should be training data (default 0.8 split)')
parser.add_argument(
    '--maxperlabel', help='limit the max number of examples per label')
parser.add_argument(
    '--mindim', help='minimum dimension to resize to (if provided)')
parser.add_argument('--equalcounts', action='store_true',
                    help='if we should have the same number of counts per label')
parser.add_argument('--shards',
                    help='number of shards to write the training TfRecords')

args = parser.parse_args()

# business logic

start = timeit.default_timer()

target_coco_class = 18  # dog

stats = {
    'labels': {}
}

labels = os.listdir(args.dataset)
tf_gen = TfGen(generate_and_persist_labels(
    labels, args.output), args.coco, target_coco_class)
jpg_cache = JpgCache(args.output, args.mindim)

label_gens = {}
for l in labels:
    files = [os.path.join(args.dataset, l, f)
             for f in os.listdir(os.path.join(args.dataset, l))]
    label_gens[l] = LabelGen(l, files, tf_gen, jpg_cache)


label_data = {}
for l, gen in label_gens.items():
    print('generating for', l)
    total = gen.total()
    not_detected_count = 0

    # collect as many examples as needed
    ex = []
    for i in range(gen.count()):
        try:
            next_tf = gen.next()
            ex.append(next_tf)
        except CocoClassNotFound as e:
            print('ERROR:', e)
            not_detected_count += 1
        except PIL.UnidentifiedImageError as e:
            print('ERROR:', e)
            # ignore me, this is just because this is not possible since we cover this case during dedupe
            not_detected_count += 1

        if args.maxperlabel and len(ex) >= int(args.maxperlabel):
            break

    label_data[l] = ex

    stats['labels'][l] = {
        'total': total,
        'not_detected_count': not_detected_count,
        'count': len(ex),
        'skip_count': gen.skipCount(),
    }

all_train = []
all_test = []
for l, data in label_data.items():
    print('collecting for', l)
    print(l, len(data))
    usable_len = len(min(label_data.values(), key=len)
                     ) if args.equalcounts else len(data)
    usable = data[:usable_len]

    # split into train and test
    split = int(usable_len*args.split)
    train = usable[:split]
    test = usable[split:]

    all_train.extend(train)
    all_test.extend(test)

    # record stats
    stats['labels'][l].update({
        'train': len(train),
        'test': len(test),
        'count_usable': usable_len,
    })

writeTfRecords(all_train, os.path.join(
    args.output, 'train.tfrecords'), args.shards)
writeTfRecords(all_test, os.path.join(args.output, 'test.tfrecords'))

stats['final_train_count'] = len(all_train)
stats['final_test_count'] = len(all_test)

stats['runtime_ms'] = int((timeit.default_timer() - start) * 1000)

print('stats:', json.dumps(stats, indent=2))

with open(os.path.join(args.output, 'stats.json'), "w") as f:
    f.write(json.dumps(stats, indent=2))
