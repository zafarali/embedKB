import tensorflow as tf
from embedKB.training import losses
import numpy as np
import os
import logging
l2_loss = tf.nn.l2_loss

class TrainWrapper(object):
    """
    Provides background utilities for training and testing methods
    implemented in the general framework.
    """

    _initialized = False
    _objective_created = False
    _optimizer_created = False
    _summaries_created = False
    saver = None

    def create_objective(self,
                         loss=losses.margin_loss,
                         margin=1,
                         regularize=None,
                         regularization_weight=0.0):
        """
        Creates the objective function for training
        :param loss: the loss function to use (by default contrastive max-margin)
        :param margin: margin to use
        :regularize: a list of matrices to regularize.
        """
        with tf.name_scope('objective'):
            self.loss_individual = loss(self.score, self.score_false, margin)
            self.score_loss = tf.reduce_sum(self.loss_individual, name='score_loss')
            self.regularized_matrices = 0
            if regularize:
                for tensor in regularize:
                    self.regularized_matrices += regularization_weight*l2_loss(tensor)
            self.objective = self.score_loss + self.regularized_matrices

            # normalize the embeddings
            self.normalize_embeddings = tf.assign(self.W_entity_embedding,
                                                  tf.nn.l2_normalize(self.W_entity_embedding,
                                                                     dim = 1))
        self._objective_created = True


    def create_optimizer(self,
            optimizer=tf.train.GradientDescentOptimizer,
            optimizer_args={'learning_rate':0.01}):
        assert self._objective_created, 'You must create the objective first'

        self.train_step = optimizer(**optimizer_args).minimize(self.objective)
        self._optimizer_created = True

    def create_summaries(self):
        # this will create summaries
        # @TODO
        tf.summary.scalar('objective', self.objective)
        tf.summary.scalar('score_loss', self.score_loss)
        tf.summary.histogram('individual_loss', self.loss_individual)
        tf.summary.histogram('pos_score_dist', self.score)
        tf.summary.histogram('neg_score_dist', self.score_false)
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self._summaries_created = True
        pass

    def score_triple(self, head_entity_id, relationship_id, tail_entity_id):
        """
        Scores a triple.
        :param head_entity_id: must be an int representing the head entity
        :param relationship_id: must be an int representing the relationship
        :param tail_entity_id: must be an int representing the tail entity
        """

        return self.batch_score_triples(np.array([[head_entity_id]]),
                                        np.array([[relationship_id]]),
                                        np.array([[tail_entity_id]]))[0][0]


    def batch_score_triples(self, head_entity_ids, relationship_ids, tail_entity_ids):
        session = self.session()
        score = session.run(self.score, feed_dict={
                self.head_entity_id: np.array(head_entity_ids).reshape(-1, 1),
                self.tail_entity_id: np.array(tail_entity_ids).reshape(-1, 1),
                self.relationship_id: np.array(relationship_ids).reshape(-1, 1),
            })

        return score

    def train_batch(self, positive_data, negative_data, summary_writer=None, epoch=None):
        """
        Trains on a batch of data.
        :param positive_data: a tuple containing 3 numpy arrays corresponding to
                             head entity ids, relationship ids, tail entity ids
        :param negative_data: a tuple containing 3 numpy arrays corresponding to
                             head entity ids, relationship ids, tail entity ids

        Notes: Each array must be of size (Bx1)
        """
        assert self._optimizer_created and \
               self._objective_created and \
               self._summaries_created, \
               'You must create the optimizer and the objective first.'
        assert positive_data[0].shape == negative_data[0].shape
        assert positive_data[1].shape == negative_data[1].shape
        assert positive_data[2].shape == negative_data[2].shape
        assert np.all(negative_data[1] == positive_data[1])
        session = self.session()
        _, obj, score, summaries = session.run([self.train_step,
                                                self.objective,
                                                self.score_loss,
                                                self.summaries],
                                                feed_dict={
                                                    self.head_entity_id: positive_data[0],
                                                    self.tail_entity_id: positive_data[2],
                                                    self.relationship_id: positive_data[1],
                                                    self.head_entity_id_false: negative_data[0],
                                                    self.tail_entity_id_false: negative_data[2],
                                                    self.relationship_id_false: negative_data[1]
                                            })
        if summary_writer and epoch is not None:
            summary_writer.add_summary(summaries, epoch)
        
        # normalize embeddings after every gradient step
        session.run(self.normalize_embeddings) # normalize the embeddings

        return obj, score, summaries

    def session(self, session=None):
        """
        Instantiates or returns a session if one is already created.
        :param session: a session that will be used if not instantiated 
                        otherwise will create a new one. 
        """
        if not self._initialized:
            session = tf.Session() if not session else session
            session.run(tf.global_variables_initializer())
            self._initialized = session
            return self._initialized
        else:
            return self._initialized
    
    def _session_end(self):
        self._initialized.close()   

    def load_model(self, path):
        session = self.session()
        self.saver = tf.train.Saver()
        self.saver.restore(session, path)

    def save_model(self, path):
        session = self.session()
        if not self.saver:
            self.saver = tf.train.Saver()
        self.saver.save(session, path)

    def train(self,
              data,
              epochs=1,
              val_data=None,
              batch_log_frequency=100,
              logging_directory='./checkpoints'):
        """
        Will train the model
        :param data: a embedKB.datatools.Dataset object or any generator that
                     returns a finite number of minibatches of (positive_data, negative_data)
                     each *_data variable must be a tuple of 3 numpy arrays:
                     head_ids (Bx1), 
                     relationship_ids (Bx1)
                     tail_ids (Bx1)
        :param epochs: number of training epochs
        :param val_data: the validation embedKB.datatools.Dataset object to validate on
        :param batch_log_frequency: number of batches to log the scores at
        :param logging_directoryt: where to save the checkpoints and summary files.
        """
        session = self.session()
        train_summary_writer = tf.summary.FileWriter(os.path.join(logging_directory,
                                                                './train__summaries_'+self.name),
                                                    session.graph)

        batch_summary_writer = tf.summary.FileWriter(os.path.join(logging_directory,
                                                                './batch__summaries_'+self.name))

        if val_data:
            if self._summaries_created:
                val_summary_writer = tf.summary.FileWriter(os.path.join(logging_directory,
                                                        './val__summaries_'+self.name))

        total_batches = 0
        try:
            for epoch in range(epochs):
                for i, (positive_data, negative_data) in enumerate(data.get_generator()):
                    total_batches += 1
                    obj, score, summaries = self.train_batch(positive_data,
                                                             negative_data)

                    if i % batch_log_frequency == 0:
                        print('\t Epoch {:d}, batch {:d}: score: {:.4f}, objective: {:.4f}'.format(
                            epoch, i, score, obj))
                        batch_summary_writer.add_summary(summaries, total_batches)

                if self._summaries_created:
                    # train_summary_writer.add_summary(summaries, epoch)
                    self.saver.save(session, os.path.join(logging_directory, self.name+".ckpt"), epoch)


                print('Epoch {:d}/{:d}: train_scores: {:.4f}, train_objective: {:.4f}'.format(
                       epoch, epochs, score, obj))


                self.evaluate(data, 'training', train_summary_writer, epoch)
                if val_data:
                    val_obj, val_score, val_summaries = self.evaluate(val_data,
                                                                      'validation',
                                                                      val_summary_writer,
                                                                      epoch)
        except KeyboardInterrupt as e:
            print('Training stopped early.')

    def top_entities(self, entity, relationship, k=5):
        """
        Returns the top k entities that fit as either subject or object.
        :param entity: the entity in the triple (int)
        :param relationship: the relationship in the triple (int)
        :return: the top k heads that best fit in (?, relationship, entity)
        :return: the top k tails that best fit in (entity, relationship, ?)
        """
        #tf.nn.top_k...#
        pass

    def top_relationships(self, entity_head, entity_tail, k=5):
        """
        Returns the top k relationships that fit the triple
        :param entity_head: the head entity in the triple (int)
        :param entity_tail: the tail entity in the triple (int)
        :return: the top k relationships that best fit in (?, relationship, entity)        """
        #tf.nn.top_k...#
        pass

    def evaluate(self, data, name='', summary_writer=None, epoch=None):
        objs = 0
        scores = 0
        session = self.session()

        for i, (positive_data, negative_data) in enumerate(data.get_generator()):
            assert np.all(negative_data[1] == positive_data[1])
            obj, score, summaries = session.run([self.objective,
                                                 self.score_loss,
                                                 self.summaries],
                                                 feed_dict={
                                                    self.head_entity_id: positive_data[0],
                                                    self.tail_entity_id: positive_data[2],
                                                    self.relationship_id: positive_data[1],
                                                    self.head_entity_id_false: negative_data[0],
                                                    self.tail_entity_id_false: negative_data[2],
                                                    self.relationship_id_false: negative_data[1]
                                                })
            objs += obj
            scores += score

        score = scores / (i+1)
        obj = objs / (i+1)
        print('\t Evaluating {}: score: {:.4f}, objective: {:.4f}'.format(name, score, obj))
        
        if summary_writer and epoch is not None:
            summary_writer.add_summary(summaries, epoch)
            
        return score, obj, summaries

    def assign_metadata_to_embedding(self,
                                     embedding_tensor,
                                     metadata_path,
                                     logging_directory):
        logging.warn('UNTESTED FUNCTIONALITY. DO NOT USE.')
        summary_writer = tf.summary.FileWriter(os.path.join(logging_directory,
                                                            './embedding_metadata__'+self.name))
        from tensorflow.contrib.tensorboard.plugins import projector
        # Use the same LOG_DIR where you stored your checkpoint.

        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_tensor.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(metadata_path)

        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
    
    def export_embedding(self,
                         embedding_to_export,
                         file_to_save):
        """
        Exports an embedding to a numpy file
        :param embedding_to_export: embedding to export
        :param file_to_save: file to save into
        """
        session = self.session()
        embedding_matrix = session.run(embedding_to_export)
        np.save(file_to_save, embedding_matrix)
