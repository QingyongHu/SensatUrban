from os import makedirs, system
from os.path import exists, join, dirname, abspath
import tensorflow as tf
import numpy as np
import time


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_' + dataset.name + '.txt', 'a')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['test']]

    def test(self, model, dataset, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.test_init_op)

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'test_preds')) if not exists(join(test_path, 'test_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'])

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1

            except tf.errors.OutOfRangeError:

                new_min = np.min(dataset.min_possibility['test'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    log_out('\nConfusion on sub clouds', self.Log_file)
                    num_test = len(dataset.input_labels['test'])

                    # Project predictions
                    log_out('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), self.Log_file)
                    proj_probs_list = []

                    for i_test in range(num_test):
                        # Reproject probs back to the evaluations points
                        proj_idx = dataset.test_proj[i_test]
                        probs = self.test_probs[i_test][proj_idx, :]
                        proj_probs_list += [probs]

                    # Show vote results
                    log_out('Confusion on full clouds', self.Log_file)
                    for i_test in range(num_test):
                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                        save_name = join(test_path, 'test_preds', dataset.input_names['test'][i_test] + '.label')
                        preds = preds.astype(np.uint8)
                        preds.tofile(save_name)

                    # creat submission files
                    base_dir = dirname(abspath(__file__))
                    results_path = join(base_dir, test_path, 'test_preds')
                    system('cd %s && zip -r %s/submission.zip *.label' % (results_path, results_path))
                    return

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
