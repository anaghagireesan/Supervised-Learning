import tensorflow as tf
from pip._vendor.requests.sessions import session
from tensorflow.python.framework.dtypes import int16, float32
import BeamSearchDecoder_testing
import Decoder_testing
from termcolor import colored
#from tflearn.layers.embedding_ops import embedding

START_ID=0
PAD_ID=1
END_ID=2

class PointerWrapper(tf.contrib.seq2seq.AttentionWrapper):
  """Customized AttentionWrapper for PointerNet."""

  def __init__(self,cell,attention_size,memory,initial_cell_state=None,name=None):
    # In the paper, Bahdanau Attention Mechanism is used
    # We want the scores rather than the probabilities of alignments
    # Hence, we customize the probability_fn to return scores directly
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, memory, probability_fn=lambda x: x )
    # According to the paper, no need to concatenate the input and attention
    # Therefore, we make cell_input_fn to return input only
    cell_input_fn=lambda input, attention: input
    # Call super __init__
    super(PointerWrapper, self).__init__(cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=None,
                                         alignment_history=False,
                                         cell_input_fn=cell_input_fn,
                                         output_attention=True,
                                         initial_cell_state=initial_cell_state,
                                         name=name)
  @property
  def output_size(self):
    return self.state_size.alignments

  def call(self, inputs, state):
    _, next_state = super(PointerWrapper, self).call(inputs, state)
    return next_state.alignments, next_state
 

class PointerNet(object):
  """ Pointer Net Model
  
  This class implements a multi-layer Pointer Network 
  aimed to solve the Convex Hull problem. It is almost 
  the same as the model described in this paper: 
  https://arxiv.org/abs/1506.03134.
  """
  # 8 users, 11 max input, 9 max output

  def __init__(self, batch_size=50, max_input_sequence_len=18, max_output_sequence_len=16, no_of_users=15,
              rnn_size=128, attention_size=128, num_layers=2, beam_width=1,
              learning_rate=0.001, max_gradient_norm=5, forward_only=False):
    '''def __init__(self, batch_size=128, max_input_sequence_len=5, max_output_sequence_len=7, no_of_users=6,
              rnn_size=128, attention_size=128, num_layers=2, beam_width=2,
              learning_rate=0.001, max_gradient_norm=5, forward_only=False):'''
    """Create the model.

    Args:
      batch_size: the size of batch during training
      max_input_sequence_len: the maximum input length 
      max_output_sequence_len: the maximum output length
      rnn_size: the size of each RNN hidden units
      attention_size: the size of dimensions in attention mechanism
      num_layers: the number of stacked RNN layers
      beam_width: the width of beam search 
      learning_rate: the initial learning rate during training
      max_gradient_norm: gradients will be clipped to maximally this norm.
      forward_only: whether the model is forwarding only
    """
    self.batch_size = batch_size
    self.max_input_sequence_len = max_input_sequence_len
    self.max_output_sequence_len = max_output_sequence_len
    self.no_of_users = no_of_users
    self.no_of_users_to_decoder = no_of_users+1
    self.forward_only = forward_only
    self.init_learning_rate = learning_rate
    self.num_layers = num_layers
    self.rnn_size = rnn_size
    self.beam_width=beam_width
    self.attention_size=attention_size
    # Note we have three special tokens namely 'START', 'PAD' and 'END'
    # Here the size of vocab need be added by 3
    self.vocab_size = max_input_sequence_len+3
    self.num_attr = 2 #excluding coordinates
    #self.user_attr = 4
    self.user_attr = 4
    # Global step
    self.global_step = tf.Variable(0, trainable=False,name="global_step")
    

    # Choose LSTM Cell
    self.targets,self.shifted_targets,self.dec_input_weights, self.predicted_ids_with_logits,self.logits,self.predicted_ids,self.decoder_inputs,self.temp,self.time,self.temp3,self.mec_attr = self.create_network()
    if forward_only == False:
        # Losses
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.shifted_targets, logits=self.logits)
        # Total loss
        self.loss = tf.reduce_sum(self.losses*tf.cast(self.dec_input_weights,tf.float32))/self.batch_size
        print(colored(self.loss,"magenta"))
        # Get all trainable variables
        parameters = tf.trainable_variables()
        # Calculate gradients
        gradients = tf.gradients(self.loss, parameters)
        # Clip gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        # Optimization
        #optimizer = tf.train.GradientDescentOptimizer(self.init_learning_rate)
        optimizer = tf.train.AdamOptimizer(self.init_learning_rate)
        # Update operator
        self.update = optimizer.apply_gradients(zip(clipped_gradients, parameters),global_step=self.global_step)
        # Summarize
        tf.summary.scalar('loss',self.loss)
        for p in parameters:
          tf.summary.histogram(p.op.name,p)
        for p in gradients:
          tf.summary.histogram(p.op.name,p)
        # Summarize operator
        self.summary_op = tf.summary.merge_all()
        #DEBUG PART
        self.debug_var = self.logits
        #/DEBUG PART
    # Saver
    self.saver = tf.train.Saver(tf.global_variables())
    for i, var in enumerate(self.saver._var_list):
      print('Var {}: {}'.format(i, var))

    

  def create_network(self):
        
      self.user = tf.Variable(0,trainable=False, expected_shape=[self.batch_size,2],name="user")
      cell = tf.contrib.rnn.LSTMCell
      # Create placeholders
      self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size,self.max_input_sequence_len,self.num_attr+2], name="inputs")
      self.user_inputs = tf.placeholder(tf.float32, shape=[self.batch_size,self.no_of_users+1,self.user_attr+2], name="user_inputs")
      self.outputs = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_output_sequence_len+1], name="outputs")
      self.enc_input_weights = tf.placeholder(tf.int32,shape=[self.batch_size,self.max_input_sequence_len], name="enc_input_weights")
      self.dec_input_weights = tf.placeholder(tf.int32,shape=[self.batch_size,self.max_output_sequence_len], name="dec_input_weights")
      # Calculate the lengths
      self.enc_input_lens=tf.reduce_sum(self.enc_input_weights,axis=1)
      #print("=========================")
      dec_input_lens=tf.reduce_sum(self.dec_input_weights,axis=1)
      #print(dec_input_lens)
      # Special token embedding creating a variable of shape (3,2)
      self.special_token_embedding = tf.get_variable("special_token_embedding", [3,self.num_attr+2], tf.float32, tf.contrib.layers.xavier_initializer())
      #print(self.special_token_embedding)
      #init = tf.constant(100.,shape=[3,self.num_attr+2],dtype=tf.float32)
      #self.special_token_embedding=tf.get_variable('special_token_embedding', initializer=init)

      # Embedding_table
      # Shape: [batch_size,vocab_size,features_size]
      #tf.expand_dims(special_token_embedding,0) = (1,[3,2])
      #tf.tile(tf.expand_dims(special_token_embedding,0),[self.batch_size,1,1]) = (1,[3,2]) * [128,1,1] = (128,[3,2])
      #tf.concat = (128,[3,2]) concat with (128,[5,2]) along axis 1 = (128,[3+5,2]) = (128,[8,3])
      #https://docs.w3cub.com/tensorflow~python/tf/concat/
      self.embedding_table = tf.concat([tf.tile(tf.expand_dims(self.special_token_embedding,0),[self.batch_size,1,1]), self.inputs],axis=1) 
      self.mec_attr = tf.cast(tf.slice(self.embedding_table,[0,0,2],[self.batch_size,self.vocab_size,self.num_attr]),dtype=tf.int32)
      self.user_attr = tf.cast(tf.slice(self.user_inputs,[0,0,2],[self.batch_size,self.no_of_users,self.num_attr]),dtype=tf.int32)
      
      
      # Unstack embedding_table
      # Shape: batch_size*[vocab_size,features_size]
      # tf.unstack along axis 0, so we will have output tensor of shape(8,2) 128 times
      embedding_table_list = tf.unstack(self.embedding_table, axis=0)
      self.embed = embedding_table_list
      embedding_table_list_users = tf.unstack(self.user_inputs,axis=0)

      # Unstack outputs
      # Shape: (max_output_sequence_len+1)*[batch_size (8)*128
      #  here tf.unstack is along dimension 1, so (8)*128 becomes output with shape 128, that is 8 times.
      #https://programming.vip/docs/tensorflow-understand-tf.unstack-value-num-none-axis-0-name-unstack.html
      outputs_list = tf.unstack(self.outputs, axis=1)
      #print(outputs_list)
      # targets
      # Shape: [batch_size,max_output_sequence_len] (128,7)
      #outputs_list[1:] so it begins from 2 row, first  row is removed i.e the first value of output is removed which is 1 here.
      self.targets = tf.stack(outputs_list[1:],axis=1)
      #print(self.targets)
      
      #this will remove last row, i.e the last element of output from all 128 because of :-1
      #shape now becomes(128,7,1) since we expand on 2nd axis. 0 is row, 1 column and then 2
      # decoder input ids 
      # Shape: batch_size*[max_output_sequence_len,1] 128*(7,1)
     
      dec_input_ids = tf.unstack(tf.expand_dims(tf.stack(outputs_list[:-1],axis=1),2),axis=0)
      
      # encoder input ids 
      # Shape: batch_size*[max_input_sequence_len+1,1]
      self.enc_input_ids = [tf.expand_dims(tf.range(2,self.vocab_size),1)]*self.batch_size
      # Look up encoder and decoder inputs
      dec_user_input_ids = [tf.expand_dims(tf.range(0,self.no_of_users_to_decoder),1)] * self.batch_size
      ##shape of user_input_ids = 128*8*1 to get the index to access embedding table
      ##user_input_ids =  [tf.expand_dims(tf.range(8,self.vocab_size+self.users_size),1)]*self.batch_size
      self.encoder_inputs = []
      self.decoder_inputs = []
      decoder_user_inputs = []
      ##user_inputs = []
      for i in range(self.batch_size):
        self.encoder_inputs.append(tf.gather_nd(embedding_table_list[i], self.enc_input_ids[i]))
        self.decoder_inputs.append(tf.gather_nd(embedding_table_list[i], dec_input_ids[i]))
        decoder_user_inputs.append(tf.gather_nd(embedding_table_list_users[i], dec_user_input_ids[i]))
        ##user_inputs.append(tf.gather_nd(embedding_table_list[i], user_input_ids[i]))
        
      # Shape: [batch_size,max_input_sequence_len+1,2]
      self.encoder_inputs = tf.stack(self.encoder_inputs,axis=0)
      # Shape: [batch_size,max_output_sequence_len,2]
      self.decoder_inputs = tf.stack(self.decoder_inputs,axis=0)
      decoder_user_inputs = tf.stack(decoder_user_inputs,axis=0)
      #shape = batch_size,max_output,2+2
      self.decoder_inputs = tf.concat([self.decoder_inputs,decoder_user_inputs], axis=2)
      
      
      ##shape [batch_size,users_size,2]
      ##user_inputs = tf.stack(user_inputs,axis=0)
      
      #final_decoder_input = tf.concat([decoder_inputs,user_inputs],axis=2)
      
      # Stack encoder cells if needed
      if self.num_layers > 1:
        fw_enc_cell = tf.contrib.rnn.MultiRNNCell([cell(self.rnn_size) for _ in range(self.num_layers)]) 
        bw_enc_cell = tf.contrib.rnn.MultiRNNCell([cell(self.rnn_size) for _ in range(self.num_layers)]) 
      else:
        fw_enc_cell = cell(self.rnn_size)
        bw_enc_cell = cell(self.rnn_size)    
      # Tile inputs if forward only
      if self.forward_only:
        # Tile encoder_inputs and enc_input_lens
        self.encoder_inputs = tf.contrib.seq2seq.tile_batch(self.encoder_inputs,self.beam_width)
        self.enc_input_lens = tf.contrib.seq2seq.tile_batch(self.enc_input_lens,self.beam_width)
      # Encode input to obtain memory for later queries
      memory,_ = tf.nn.bidirectional_dynamic_rnn(fw_enc_cell, bw_enc_cell, self.encoder_inputs, self.enc_input_lens, dtype=tf.float32)
      # Shape: [batch_size(*beam_width), max_input_sequence_len+1, 2*rnn_size]
      memory = tf.concat(memory, 2) 
      # PointerWrapper
      pointer_cell = PointerWrapper(cell(self.rnn_size), self.attention_size, memory)
      # Stack decoder cells if needed
      if self.num_layers > 1:
        dec_cell = tf.contrib.rnn.MultiRNNCell([cell(self.rnn_size) for _ in range(self.num_layers-1)]+[pointer_cell])
      else:
        dec_cell = pointer_cell
     
      # Different decoding scenario
      if self.forward_only:
        # Tile embedding_table
        #expand dim shape becomes 128,1,8,2  and tiling it becomes 128,2,8,2
        tile_embedding_table = tf.tile(tf.expand_dims(self.embedding_table,1),[1,self.beam_width,1,1])
        tile_embedding_table_user = tf.tile(tf.expand_dims(self.user_inputs,1),[1,self.beam_width,1,1])
        
        # Customize embedding_lookup_fn from beam_search_encoder class
        def embedding_lookup(ids,time):
          # Note the output value of the decoder only ranges 0 to max_input_sequence_len
          # while embedding_table contains two more tokens' values 
          # To get around this, shift ids
          # Shape: [batch_size,beam_width] 
          #ids_user = ids + (user+2)
          ids = ids+2
          #self.user = self.user + 1
          # Shape: [batch_size,beam_width,vocab_size] 128,2,8
          one_hot_ids_base = tf.cast(tf.one_hot(ids,self.vocab_size), dtype=tf.float32)
          # Shape: [batch_size,beam_width,vocab_size] 128,2,7
          one_hot_ids_user = tf.cast(tf.one_hot(time,self.no_of_users+1), dtype=tf.float32)
          # Shape: [batch_size,beam_width,vocab_size,1] 128,2,8,1
          one_hot_ids_base = tf.expand_dims(one_hot_ids_base,-1)
          # Shape: [batch_size,beam_width,vocab_size,1] 128,2,7,1
          one_hot_ids_user = tf.expand_dims(one_hot_ids_user,-1)
          tf.Print(self.user,[self.user])
          # Shape: [batch_size,beam_width,features_size]
          next_inputs_base = tf.reduce_sum(one_hot_ids_base*tile_embedding_table, axis=2)  
          next_inputs_user = tf.reduce_sum(one_hot_ids_user*tile_embedding_table_user, axis=2)
          next_inputs = tf.concat([next_inputs_base,next_inputs_user], axis=2)
          return next_inputs
        # Do a little trick so that we can use 'BeamSearchDecoder'
        shifted_START_ID = START_ID - 2
        shifted_END_ID = END_ID - 2
        # Beam Search Decoder
        '''decoder = tf.contrib.seq2seq.BeamSearchDecoder(dec_cell, embedding_lookup, 
                                            tf.tile([shifted_START_ID],[self.batch_size]), shifted_END_ID, 
                                            dec_cell.zero_state(self.batch_size*beam_width,tf.float32), beam_width)'''
        decoder = BeamSearchDecoder_testing.BeamSearchDecoder(dec_cell, embedding_lookup, 
                                            tf.tile([shifted_START_ID],[self.batch_size]), shifted_END_ID, 
                                            dec_cell.zero_state(self.batch_size*self.beam_width,tf.float32),
                                            self.beam_width,self.batch_size,self.no_of_users,self.num_attr, self.mec_attr,self.user_attr,self.vocab_size)
        '''decoder = BeamSDecoder.BeamSearchDecoder(dec_cell, embedding_lookup, 
                                            tf.tile([shifted_START_ID],[self.batch_size]), shifted_END_ID, 
                                            dec_cell.zero_state(self.batch_size*self.beam_width,tf.float32),self.beam_width,self.block,self.no_of_users,self.batch_size)'''
        # Decode
        # Decode
        #outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        self.outputs, _, _,self.temp,self.temp2,self.temp3,self.time =  Decoder_testing.dynamic_decode(decoder,self.batch_size,self.vocab_size,self.num_attr)
        #self.outputs, _, _ =  Decoder_testing.dynamic_decode(decoder)
        # predicted_ids
        # Shape: [batch_size, max_output_sequence_len,  beam_width]
        predicted_ids = self.outputs.predicted_ids
        # Transpose predicted_ids
        # Shape: [batch_size, beam_width, max_output_sequence_len]
        self.predicted_ids = tf.transpose(predicted_ids,[0,2,1])
        return None,None,None,None,None,self.predicted_ids,None,self.temp,self.time,self.temp3,self.mec_attr
        #return None,None,None,None,None,self.predicted_ids,None
      else:
        # Get the maximum sequence length in current batch
        cur_batch_max_len = tf.reduce_max(dec_input_lens)  #7
        # Training Helper
        helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs, dec_input_lens) 
       
        # Basic Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, dec_cell.zero_state(self.batch_size,tf.float32)) 
        # Decode
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)
        # logits
        self.logits = outputs.rnn_output
        # predicted_ids_with_logits
        self.predicted_ids_with_logits=tf.nn.top_k(self.logits)
        # Pad logits to the same shape as targets

        self.logits = tf.concat([self.logits,tf.ones([self.batch_size,self.max_output_sequence_len-cur_batch_max_len,self.max_input_sequence_len+1])],axis=1)
        # Subtract target values by 2
        # because prediction output ranges from 0 to max_input_sequence_len+1
        # while target values are from 0 to max_input_sequence_len + 3 
        self.shifted_targets = (self.targets - 2)*self.dec_input_weights
        return self.targets,self.shifted_targets,self.dec_input_weights, self.predicted_ids_with_logits,self.logits,None,self.decoder_inputs,None,None,None,self.mec_attr
      
    
  

  def step(self, session, inputs, user_inputs, enc_input_weights, outputs=None, dec_input_weights=None):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      inputs: the point positions in 2D coordinate. shape: [batch_size,max_input_sequence_len,2]
      enc_input_weights: the weights of encoder input points. shape: [batch_size,max_input_sequence_len]              
      outputs: the point indexes in inputs. shape: [batch_size,max_output_sequence_len+1] 
      dec_input_weights: the weights of decoder input points. shape: [batch_size,max_output_sequence_len] 

    Returns:
      (training)
      The summary      
      The total loss
      The predicted ids with logits
      The targets
      The variable for debugging

      (evaluation)
      The predicted ids
    """
    #Fill up inputs 
    input_feed = {}
    input_feed[self.inputs] = inputs
    input_feed[self.user_inputs] = user_inputs
    input_feed[self.enc_input_weights] = enc_input_weights
    #print(inputs)
    if self.forward_only==False:
      input_feed[self.outputs] = outputs
      input_feed[self.dec_input_weights] = dec_input_weights
    if self.forward_only:
      output_feed = [self.predicted_ids,self.temp,self.temp2,self.temp3,self.time,self.mec_attr,self.user_attr,self.embedding_table]
    else:
      output_feed = [self.update, self.summary_op, self.loss, self.predicted_ids_with_logits, self.shifted_targets, self.debug_var, self.targets,self.logits,self.embedding_table, self.losses]
      #print("output feed", output_feed)
    
    #Run step
    outputs = session.run(output_feed, input_feed)

    if self.forward_only:
      return outputs[0]
    else:
      return outputs[1],outputs[2],outputs[3],outputs[4],outputs[5]
