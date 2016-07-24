#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from init import *

class LSTM_MODEL(object):

	def __init__(self, is_training, word_embeddings, config):

		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size
		num_classes = config.num_classes

		self.input_x = tf.placeholder(tf.int32, [batch_size, num_steps])
		self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes])

		# Slightly better results can be obtained with forget gate biases
		# initialized to 1 but the hyperparameters of the model would need to be
		# different than reported in the paper.
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

		self._initial_state = cell.zero_state(batch_size, tf.float32)

		with tf.device("/cpu:0"), tf.name_scope("embedding"):
			embedding = tf.get_variable("embedding", [vocab_size, size])
			inputs = tf.nn.embedding_lookup(embedding, self.input_x)

		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		output = tf.expand_dims(tf.reshape(tf.concat(1,  outputs), [batch_size, -1, size]), -1)

		with tf.name_scope("maxpool"):
			output_pooling = tf.nn.max_pool(output,
					ksize=[1, num_steps, 1, 1],
					strides=[1, 1, 1,1],
					padding='VALID',
					name="pool")
			self.output = tf.reshape(output_pooling, [-1, size])

		with tf.name_scope("output"):
			softmax_w = tf.get_variable("softmax_w", [size, num_classes])
			softmax_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="softmax_b")
			self.scores = tf.nn.xw_plus_b(self.output, softmax_w, softmax_b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss = tf.reduce_mean(losses)

		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
		# if not is_training:
		# 	return
		# optimizer = tf.train.AdamOptimizer(0.001)
		# grads_and_vars = optimizer.compute_gradients(cnn.loss)
		# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

class Config(object):

	def __init__(self):
		self.num_layers = 2
		self.batch_size = 64
		self.keep_prob = 0.5
		self.num_epochs = 20
		self.num_steps = 50
		self.hidden_size = 50
		self.vocab_size = 10000
		self.num_classes = 50

def main(_):
	w, x_train, y_train, x_dev, y_dev = load_data_and_labels()
	config = Config()
	config.num_steps = len(x_train[0])
	config.hidden_size = len(w[0])
	config.vocab_size = len(w)
	config.num_classes = len(y_train[0])
	
	eval_config = Config()
	eval_config.num_steps = len(x_train[0])
	eval_config.hidden_size = len(w[0])
	eval_config.vocab_size = len(w)
	eval_config.num_classes = len(y_train[0])
	eval_config.keep_prob = 1.0
	# eval_config.batch_size = 64



	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():

			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				m = LSTM_MODEL(is_training=True, word_embeddings = w, config = config)
			with tf.variable_scope("model", reuse=True, initializer = initializer):
				mtest = LSTM_MODEL(is_training=False, word_embeddings = w, config = eval_config)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(0.001)
			grads_and_vars = optimizer.compute_gradients(m.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			sess.run(tf.initialize_all_variables())

			def train_step(x_batch, y_batch):
				"""
				A single training step
				"""
				feed_dict = {
					m.input_x: x_batch,
					m.input_y: y_batch,
				}
				_, step, loss, accuracy = sess.run(
					[train_op, global_step, m.loss, m.accuracy], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				if step % 100 == 0:
	 				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

			def dev_step(x_batch, y_batch, writer=None):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
					mtest.input_x: x_batch,
					mtest.input_y: y_batch,
				}
				step, loss, accuracy = sess.run(
					[global_step, mtest.loss, mtest.accuracy], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				return accuracy

			batches = batch_iter(list(zip(x_train, y_train)), config.batch_size, config.num_epochs)

			for batch in batches:
				x_batch, y_batch = zip(*batch)
				train_step(x_batch, y_batch)
				current_step = tf.train.global_step(sess, global_step)
				if current_step % 500 == 0:
					print("\nEvaluation:")
					res = 0.0
					tot = 0.0
					num = (int)(len(y_dev) / (float)(eval_config.batch_size))
					print num
					for i in range(num):
						res = res + dev_step(x_dev[i * eval_config.batch_size:(i+1)*eval_config.batch_size], y_dev[i * eval_config.batch_size:(i+1)*eval_config.batch_size])
						tot = tot + 1.0
					print res/tot

if __name__ == "__main__":
	tf.app.run()