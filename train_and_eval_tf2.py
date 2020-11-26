# Modified from model_main_tf2.py

from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

from object_detection.utils import config_util
import multiprocessing

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

FLAGS = flags.FLAGS


def getCurrentStep(model_dir, ret_value):
    global_step = tf.Variable(0, dtype=tf.compat.v2.dtypes.int64)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    tf.compat.v2.train.Checkpoint(step=global_step).restore(latest_checkpoint)
    ret_value.value = int(global_step.numpy())


def runTrainLoop(pipeline_config_path, model_dir, train_steps, checkpoint_every_n):
    model_lib_v2.train_loop(
        pipeline_config_path=pipeline_config_path,
        model_dir=model_dir,
        train_steps=train_steps,
        checkpoint_every_n=checkpoint_every_n)


def runEval(pipeline_config_path, model_dir, train_steps, sample_1_of_n_eval_examples, sample_1_of_n_eval_on_train_examples):
    wait_interval = timeout = 0  # don't wait
    model_lib_v2.eval_continuously(
        pipeline_config_path=pipeline_config_path,
        model_dir=model_dir,
        sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=sample_1_of_n_eval_on_train_examples,
        checkpoint_dir=model_dir,  # checkpoints are put into model_dir
        wait_interval=wait_interval, timeout=timeout)


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    tf.config.set_soft_device_placement(True)

    # get total steps from config
    total_steps = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)['train_config'].num_steps

    while True:
        current_step_value = multiprocessing.Value('i')
        p = multiprocessing.Process(target=getCurrentStep, args=[
                                    FLAGS.model_dir, current_step_value])
        p.start()
        p.join()

        current_step = current_step_value.value

        if current_step >= total_steps:
            print("done training!!!")
            break

        up_to_step = current_step + FLAGS.checkpoint_every_n

        print("training steps {} to {}".format(current_step, up_to_step))

        p = multiprocessing.Process(target=runTrainLoop, args=[
            FLAGS.pipeline_config_path,
            FLAGS.model_dir,
            up_to_step,
            FLAGS.checkpoint_every_n])
        p.start()
        p.join()

        print("evaluate")

        p = multiprocessing.Process(target=runEval, args=[
            FLAGS.pipeline_config_path,
            FLAGS.model_dir,
            FLAGS.sample_1_of_n_eval_examples,
            FLAGS.checkpoint_every_n,
            FLAGS.sample_1_of_n_eval_on_train_examples])
        p.start()
        p.join()

        print("done! and again")


if __name__ == '__main__':
    tf.compat.v1.app.run()
