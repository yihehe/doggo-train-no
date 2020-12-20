import argparse
import tensorflow as tf
import numpy as np
import timeit

# tf2.4 on ampere OOM
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


features={
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64)
}


def getClass(model, ex):
    if model == None: # for testing
        return 1

    image = np.asarray(tf.image.decode_jpeg(ex['image/encoded']))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    return int(output_dict['detection_classes'][0][0].numpy())

# parse args
parser = argparse.ArgumentParser(description='what is my purpose?')

required = parser.add_argument_group('required arguments')
required.add_argument('--tfrecord', required=True, help='the dataset')
required.add_argument('--model', required=True,
                      help='directory of the (eg ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model')

args = parser.parse_args()

# business logic

start = timeit.default_timer()

label_cache = {} # this is just so that we don't have to import the label map
confusion_matrix = np.zeros((5,5))

model = tf.saved_model.load(args.model)
dataset = tf.data.TFRecordDataset(args.tfrecord)
for record in dataset:
    ex = tf.io.parse_single_example(record, features)

    filename = ex['image/filename'].numpy()
    text = ex['image/object/class/text'].values.numpy()[0]
    label = ex['image/object/class/label'].values.numpy()[0]

    pred = getClass(model, ex)

    confusion_matrix[pred][label] += 1
    label_cache[label] = text

# print(label_cache)
# print(confusion_matrix)

print('Confusion Matrix:')
for row in range(len(confusion_matrix)):
    for col in range(len(confusion_matrix[row])):
        if row == 0 and col == 0:
            print('\t', end='')
        elif row == 0 or col == 0:
            print(label_cache[max(row, col)].decode(), '\t', end = '')
        else:
            print(int(confusion_matrix[row, col]), '\t', end='')
    print()

print()
print('done in %sms' % int((timeit.default_timer() - start) * 1000))

# Confusion Matrix:
#         Down    Paw     Sit     Stand 
# Down    61      1       6       1
# Paw     2       56      6       2
# Sit     0       6       50      2
# Stand   0       0       1       58

# done in 197201ms

# Confusion Matrix:
#         Down    Paw     Sit     Stand 
# Down    61      1       6       1
# Paw     2       56      6       2
# Sit     0       6       50      2
# Stand   0       0       1       58

# done in 161207ms