    if summary_path:
        # setup summary saver
        self.writer = tf.summary.FileWriter(summary_path)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('score', self.score)
        tf.summary.scalar('mean_q', self.mean_q)
        tf.summary.scalar('max_q', self.max_q)
        self.write_op = tf.summary.merge_all()

    if self.save_path:
        # setup model saver
        self.saver = tf.train.Saver()
    
  
def save_model(self, sess):
	''' Write TF checkpoint '''
	self.saver.save(sess, self.save_path)
    
    
def load(self, sess):
    ''' Load from TF checkpoint '''
    self.saver.restore(sess, self.save_path)
    

def write_summary(self, sess, states, actions, targets, score):
    ''' Write session summary to TensorBoard '''
    global_episode = self.global_episode.eval(sess)  # what is global_episode at runtime?
    summary = sess.run([self.loss, self.optimizer],
                       feed_dict = {self.inputs: states,
                                    self.actions: actions,
                                    self.targets: targets})
    
    
def increment_global_episode(self, sess):
    ''' Increment the global episode tracker '''
    sess.run(self.increment_global_episode)
    
    
def optimizer_op(self, sess, states, actions, targets):
    ''' Perform one iteration of gradient updates '''
    loss, _ = sess.run([self.loss, self.optimizer],
                       feed_dict={self.inputs: states,
                                  self.actions: actions,
                                  self.targets: targets})

