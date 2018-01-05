from __future__ import absolute_import
from __future__ import division
from __future__ import  print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from process_data import cbow_process_data
import utils

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
WEIGHTS_FLD = 'cbow_processed/'
SKIP_STEP = 100

class CBOWModel:
    """ Build the graph for word2vec model """
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learing_rate, skip_window):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learing_rate
        self.skip_window = skip_window
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("data"):
            self.content_words = tf.placeholder(tf.int32, shape=[self.batch_size, self.skip_window * 2], name='contex_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        with tf.device('/cpu:0'):
            with tf.name_scope("embed"):
                self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                                   self.embed_size], -1.0, 1.0), name='embed_matrix')

    def _create_loss(self):
        """ Step 3 + 4: define the model + the loss function """
        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                # Step 3: define the inference
                embed = None
                for i in range(2 * self.skip_window):
                    embedding_i = tf.nn.embedding_lookup(self.embed_matrix, self.content_words[:, i], name="embedding_i")
                    embed_x, embed_y = embedding_i.get_shape().as_list()
                    if embed is None:
                        embed = tf.reshape(embedding_i, [embed_x, embed_y, 1])
                    else:
                        embed = tf.concat(values=[embed, tf.reshape(embedding_i, [embed_x, embed_y, 1])], axis=2)
                avg_embed = tf.reduce_mean(embed, 2, False, "avg_embed")
                # Step 4: define loss function
                # construct variables for NCE loss
                nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                            stddev=1.0 / (self.embed_size ** 0.5)),
                                                            name='nce_weight')
                nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
                #define loss function to be NCE loss function
                """
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                          biases=nce_bias,
                                                          labels=self.target_words,
                                                          inputs=avg_embed,
                                                          num_sampled=self.num_sampled,
                                                          num_classes=self.vocab_size), name='loss')
                """
                self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weight,
                                                                      biases=nce_bias,
                                                                      inputs=avg_embed,
                                                                      labels=self.target_words,
                                                                      num_sampled=self.num_sampled,
                                                                      num_classes=self.vocab_size), name='loss')


    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

def train_model(model, batch_gen, num_train_steps, weights_fld):
    saver = tf.train.Saver() # defaults to saving all variables
    initial_step = 0
    utils.make_dir('cbow_checkpoints')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('cbow_checkpoints/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('cbow_improved_graph/lr'+str(LEARNING_RATE), sess.graph)
        initial_step = model.global_step.eval()
        for index in range(initial_step, initial_step + num_train_steps):
            contexts, targets = next(batch_gen)
            feed_dict = {model.content_words: contexts, model.target_words: targets}
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'cbow_checkpoints/skip-gram', index)


        ########
        # code to visualize the embeddings.
        final_embed_matrix = sess.run(model.embed_matrix)

        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('cbow_processed')

        # # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        embedding.metadata_path = 'cbow_processed/vocab_1000.tsv'
        # # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'cbow_processed/model3.ckpt', 1)


def main():
    model = CBOWModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE, SKIP_WINDOW)
    model.build_graph()
    batch_gen = cbow_process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)

if __name__ == '__main__':
    main()
