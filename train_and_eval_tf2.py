# Modified from model_main_tf2.py

from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

from object_detection.utils import config_util
import multiprocessing
import ctypes

import winsound

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_integer(
    'checkpoint_every_n', 1000, 'Integer defining how often we checkpoint.')
flags.DEFINE_bool(
    'alert', False, 'Whether to play a sound when done (windows only)')

FLAGS = flags.FLAGS

# tf2.4 on ampere OOM
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def getCurrentStep(ret_value, model_dir):
    global_step = tf.Variable(0, dtype=tf.compat.v2.dtypes.int64)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    tf.compat.v2.train.Checkpoint(step=global_step).restore(latest_checkpoint)
    ret_value.value = int(global_step.numpy())


def runTrainLoop(ret_value, pipeline_config_path, model_dir, checkpoint_every_n, train_steps=None):
    try:
        if train_steps:
            model_lib_v2.train_loop(
                pipeline_config_path=pipeline_config_path,
                model_dir=model_dir,
                train_steps=train_steps,
                checkpoint_every_n=checkpoint_every_n,
                checkpoint_max_to_keep=100,
            )
        else:
            model_lib_v2.train_loop(
                pipeline_config_path=pipeline_config_path,
                model_dir=model_dir,
                checkpoint_every_n=checkpoint_every_n,
                checkpoint_max_to_keep=100,
            )
    except Exception as e:
        ret_value.value = True
        print(e)


def runEval(ret_value, pipeline_config_path, model_dir, sample_1_of_n_eval_examples, sample_1_of_n_eval_on_train_examples):
    try:
        wait_interval = timeout = 0  # don't wait
        model_lib_v2.eval_continuously(
            pipeline_config_path=pipeline_config_path,
            model_dir=model_dir,
            sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=sample_1_of_n_eval_on_train_examples,
            checkpoint_dir=model_dir,  # checkpoints are put into model_dir
            wait_interval=wait_interval, timeout=timeout)
    except Exception as e:
        ret_value.value = True
        print(e)


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    tf.config.set_soft_device_placement(True)

    # get total steps from config
    total_steps = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)['train_config'].num_steps

    while True:
        current_step_value = multiprocessing.Value(ctypes.c_int)
        p = multiprocessing.Process(target=getCurrentStep, args=[
                                    current_step_value, FLAGS.model_dir])
        p.start()
        p.join()

        current_step = current_step_value.value

        if current_step >= total_steps:
            print("done training!!!")
            break

        up_to_step = current_step + FLAGS.checkpoint_every_n

        print("training steps {} to {}".format(current_step, up_to_step))

        error_value = multiprocessing.Value(ctypes.c_bool)
        p = multiprocessing.Process(target=runTrainLoop, args=[
            error_value,
            FLAGS.pipeline_config_path,
            FLAGS.model_dir,
            FLAGS.checkpoint_every_n,
            up_to_step,
        ])
        p.start()
        p.join()

        if error_value.value:
            print("there was an error running training loop, stopping")
            break

        print("evaluate")

        error_value = multiprocessing.Value(ctypes.c_bool)
        p = multiprocessing.Process(target=runEval, args=[
            error_value,
            FLAGS.pipeline_config_path,
            FLAGS.model_dir,
            FLAGS.sample_1_of_n_eval_examples,
            FLAGS.sample_1_of_n_eval_on_train_examples,
        ])
        p.start()
        p.join()

        if error_value.value:
            print("there was an error running evaluation, stopping")
            break

        print("done! and again")

    if FLAGS.alert:
        # windows only
        while True:
            winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

if __name__ == '__main__':
    tf.compat.v1.app.run()
