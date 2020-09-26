#FOR SHUFFLED USERS MEC_envv7, pointer_ddpg

import tensorflow as tf
import numpy as np
import pointer_ddpg
import time
import os
from termcolor import colored
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import gym
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import mecpt_supervised


tf.app.flags.DEFINE_integer("batch_size", 50,"Batch size.") #30
tf.app.flags.DEFINE_integer("no_of_base_stations", 34,"Base stations.")
tf.app.flags.DEFINE_integer("no_of_users", 15, "Users.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 18, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 16, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 1, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
tf.app.flags.DEFINE_string("not_log", "./temp", "Log directory")
tf.app.flags.DEFINE_string("data_path", "shuffled_train.txt", "Data path.")
tf.app.flags.DEFINE_string("user_data_path", "shuffled_user.txt", "User path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",80 , "frequence to do per checkpoint.") #60

FLAGS = tf.app.flags.FLAGS

class ConvexHull(object):
    def __init__(self, forward_only):
        custom_MEC = mecpt_supervised.MEC_env(FLAGS.batch_size,FLAGS.no_of_users)
        self.hop_to_hop, self.pos_mec = custom_MEC.latency()
        self.forward_only = forward_only
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
        self.build_model()
        self.replay_buffer = ReplayBuffer(50000, 123)
        self.read_data(custom_MEC)

    '''def read_data(self):
        custom_MEC = gym.make('custom_gym:MEC-v0')
        for i in range(20):
            custom_MEC.reset()
            for j in range(5):
                action = custom_MEC.action_space.sample()
                action = list(action)
                for k in range(10):
                    if action[k] == 0:
                        action[k] = 1
                pob,a,r,done,info,ob = custom_MEC.step(action)
                self.replay_buffer.add(pob,a,r,done,ob)
        print("exited read")'''

    def read_data(self, custom_MEC):
        for i in range(1000):
            if i%20 == 0:
                print(i)
            custom_MEC.reset()
            for j in range(1):
                action = [1]
                pob,a,ob,mec_map,b = custom_MEC.calc_reward(action)
                self.replay_buffer.add(pob,a,ob,mec_map,b)
        print("exited read")

    def plot_learning_curve(self,filename, value_dict, xlabel='step'):
    # Plot step vs the mean(last 50 episodes' rewards)
        fig = plt.figure(figsize=(12, 4 * len(value_dict)))

        for i, (key, values) in enumerate(value_dict.items()):
            ax = fig.add_subplot(len(value_dict), 1, i + 1)
            ax.plot(range(len(values)), values)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(key)
            ax.grid('k--', alpha=0.6)

        plt.tight_layout()
        os.makedirs(os.path.join(FLAGS.not_log, 'figs_sp'), exist_ok=True)
        plt.savefig(os.path.join(FLAGS.not_log, 'figs_sp', filename))

    def prepare_inputs(self,inputs):
        data_inputs= []
        user_inputs = []
        enc_input_weights = []
        input_feed = {}
        observation = []
        bandwidth = []
        users = []
        for index in range(len(inputs)):
            s = inputs[index]
            observation = s[0]
            bandwidth = s[1]
            mecservice = s[2]
            users_band = s[3]
            users = s[4]
            base_station = s[5]
            u_inp = []
            u_band = []
            #mig = []
            base = []
            for t in users:
                u_inp.append(float(t))
            for t in users_band:
                u_band.append(float(t))
           
            for t in base_station:
                base.append(float(t))
            current_users_len = int(len(u_inp)/2)
            #append -1.0 for satisfying the length
            u_inp += [-1.0]*(((FLAGS.no_of_users)-current_users_len)*2)
            u_band += [-1.0]*(((FLAGS.no_of_users)-current_users_len)*1)
            #append -10.0 for the last user to predict end token
            u_inp += [-10.0]*(2)
            u_band += [-10.0]*(1)
            u_inp = np.array(u_inp).reshape([-1,2])
            u_band = np.array(u_band).reshape([-1,1])
            val = np.concatenate((u_inp,u_band),axis=1)
            #u_ser = np.full((FLAGS.no_of_users+1,1),1)#for additional appended user
            #u_ser = np.full((current_users_len,1),1)
            u_ser = [1 for i in range(current_users_len)]
            u_ser += [-1.0]*(((FLAGS.no_of_users)-current_users_len)*1)
            u_ser += [-10.0]*1
            u_ser = np.reshape(u_ser,(FLAGS.no_of_users+1,1))
            #u_ser[current_users_len:]=0
            val = np.concatenate((val,u_ser),axis=1)
            #mig += [0.0]*(((FLAGS.no_of_users+1)-current_users_len)*1)
            base += [-1.0]*(((FLAGS.no_of_users)-current_users_len)*2)
            base += [-10.0]*2
            #mig = np.array(mig).reshape([-1,1])
            base = np.array(base).reshape([-1,2])
            #val = np.concatenate((val,mig),axis=1)
            val = np.concatenate((val,base),axis=1)
            #val = val.reshape([-1,6])
            val = val.reshape([-1,6])

            user_inputs.append(val)

            '''ADD OBSERVATION (mec coordinates) AND BANDWIDTH, PAD 10 IF DOES NOT SATISFY MAX INPUT LENGTH''' 

            '''observation = np.append(observation,10)
            observation = np.append(observation,10)
            bandwidth = np.append(bandwidth,60)
            mecservice = np.append(mecservice,6)'''
            len_ob = int(len(observation)/2)
            pad_ob = [0] * ((FLAGS.max_input_sequence_len-(len_ob))*2)
            pad_bw = [0] * ((FLAGS.max_input_sequence_len-(len_ob))*1)
            pad_ser = [0] * ((FLAGS.max_input_sequence_len-(len_ob))*1)
            observation = np.append(observation,pad_ob)
            bandwidth = np.append(bandwidth,pad_bw)
            mecservice = np.append(mecservice,pad_ser)
            observation = observation.reshape([-1,2])
            bandwidth = bandwidth.reshape([-1,1])
            mecservice = mecservice.reshape([-1,1])
            val = np.concatenate((observation,bandwidth),axis=1)
            val = np.concatenate((val,mecservice),axis=1)
            val = val.reshape([-1,4]) 
            data_inputs.append(val)
            '''ENC INPUT WEIGHTS'''
            weight = np.zeros(FLAGS.max_input_sequence_len)
            weight[:len_ob+1]=1
            enc_input_weights.append(weight)

        data_inputs = np.stack(data_inputs)
        user_inputs = np.stack(user_inputs)
        enc_input_weights = np.stack(enc_input_weights)
        return data_inputs,user_inputs,enc_input_weights

    def mec_mapping(self,m,b):
        mec_map = []
        base = []
        for i in range(len(m)):
            mec_map.append(m[i])
            base.append(b[i])
        mec_map = np.stack(mec_map)
        return mec_map,base

    def prepare_outputs(self,outputs):

        output = []
        dec_input_weights = []
        for i in range(len(outputs)):
            out = [] 
            weight = []
            out.append(0)
            act = outputs[i]
            for val in act:
                out.append(val+2)
            out.append(2)
            dec_input_len = len(out)-1
            out += [1]*(FLAGS.max_output_sequence_len-dec_input_len)
            out = np.array(out)
            output.append(out)
            weight = np.zeros(FLAGS.max_output_sequence_len)
            weight[:dec_input_len]=1
            dec_input_weights.append(weight)
        output = np.stack(output)
        dec_input_weights = np.stack(dec_input_weights)
        return output,dec_input_weights


    def get_batch(self):
        data_size = self.inputs.shape[0]
        sample = np.random.choice(data_size,FLAGS.batch_size,replace=True)
        return self.inputs[sample],self.user_inputs[sample],self.enc_input_weights[sample],\
        self.outputs[sample], self.dec_input_weights[sample]

    def build_model(self):
        with self.graph.as_default():

            self.model = pointer_ddpg.PointerNet(batch_size=FLAGS.batch_size, 
                        max_input_sequence_len=FLAGS.max_input_sequence_len, 
                        max_output_sequence_len=FLAGS.max_output_sequence_len, 
                        no_of_users=FLAGS.no_of_users,
                        rnn_size=FLAGS.rnn_size, 
                        attention_size=FLAGS.attention_size, 
                        num_layers=FLAGS.num_layers,
                        beam_width=FLAGS.beam_width, 
                        learning_rate=FLAGS.learning_rate, 
                        max_gradient_norm=FLAGS.max_gradient_norm, 
                        forward_only=self.forward_only)
            # Prepare Summary writer
            self.writer = tf.summary.FileWriter(FLAGS.not_log + '/train',self.sess.graph)
            # Try to get checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.not_log)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Load model parameters from %s" % ckpt.model_checkpoint_path)
                self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                self.sess.run(tf.global_variables_initializer())


    def train(self):
        step_time = 0.0
        loss = 0.0
        current_step = 0
        finish = True
        loss_v = []
        latency = []
        i=1
        while i in range(50):
            start_time = time.time()
            s_batch, a_batch, s2_batch, m_batch, b_batch = \
                self.replay_buffer.sample_batch(FLAGS.batch_size)
            self.inputs,self.user_inputs,self.enc_input_weights = self.prepare_inputs(s_batch)
            self.mec_map, self.base = self.mec_mapping(m_batch, b_batch)
            self.outputs,self.dec_input_weights = self.prepare_outputs(a_batch)
            #print(self.user_inputs)
            '''summary, step_loss, predicted_ids_with_logits, targets, debug_var = \
                      self.model.step(self.sess, inputs, enc_input_weights, outputs, dec_input_weights)'''
            summary, step_loss, predicted_ids_with_logits, targets, debug_var = \
                      self.model.step(self.sess, self.inputs, self.user_inputs, self.enc_input_weights, self.outputs, self.dec_input_weights)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            x = step_loss / FLAGS.steps_per_checkpoint
            loss_v.append(x)
            current_step += 1
            #Time to print statistic and save model
            if current_step % FLAGS.steps_per_checkpoint == 0:
                with self.sess.as_default():
                    gstep = self.model.global_step.eval()
                print ("global step %d step-time %.2f loss %.2f" % (gstep, step_time, loss))
                #Write summary
                self.writer.add_summary(summary, gstep)
                #Randomly choose one to check
                sample = np.random.choice(FLAGS.batch_size,1)[0]
                #print(np.array(predicted_ids_with_logits[1]))
                print("Predict: "+str(np.array(predicted_ids_with_logits[1][sample]).reshape(-1)))
                print("Target : "+str(targets[sample]))
                print("="*20)  
                checkpoint_path = os.path.join(FLAGS.not_log, "convex_hull.ckpt")
                self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)

                step_time, loss = 0.0, 0.0
                finish = False
                i = i+1
                l =0
                
                #c_action = targets[sample]
                action = np.array(predicted_ids_with_logits[1][sample]).reshape(-1)
                action = action[action!=0]
                action = action.tolist()
                mec_map = self.mec_map[sample]
                base = self.base[sample]
                r_action = [mec_map[i] for i in action if i in mec_map]
                #c_action = [mec_map[i] for i in c_action if i in mec_map]
                '''print(c_action)
                print(r_action)
                print(base)
                print(self.user_inputs[sample])
                print(self.inputs[sample])'''
                for k in range(len(r_action)):
                    if r_action[k] == FLAGS.max_input_sequence_len:
                        l = l + 5
                    else:
                        mec_in_bs = list(self.pos_mec.keys())[list(self.pos_mec.values()).index(r_action[k])]
                        l = l + self.hop_to_hop[base[k]][mec_in_bs]+1
                l = l/len(r_action)
                print(l)
                latency.append(l)
                
        data_dict_loss = {
        'loss': loss_v}
        data_dict_latency = {
        'latency': latency}
        self.plot_learning_curve('loss', data_dict_loss, xlabel='steps')
        self.plot_learning_curve('latency', data_dict_latency, xlabel='steps')

    def eval(self):
        
        step_time = 0.0
        s_batch, a_batch, s2_batch, m_batch, b_batch = \
                self.replay_buffer.sample_batch(FLAGS.batch_size)
        self.inputs,self.user_inputs,self.enc_input_weights = self.prepare_inputs(s_batch)
        self.outputs,self.dec_input_weights = self.prepare_outputs(a_batch)
        start_time = time.time()
        '''predicted_ids = self.model.step(self.sess, inputs, enc_input_weights)'''
        print("before step")
        print(self.inputs, self.user_inputs, self.enc_input_weights)  
        predicted_ids = self.model.step(self.sess, self.inputs, self.user_inputs, self.enc_input_weights)  
        print("after step")
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        print("="*20)
        success = 0
        for i in range(FLAGS.batch_size):
            reward = 0
            tar = self.outputs[i,1:]-2
            pred = predicted_ids[i][0]
            tar = [x for x in tar if x !=-1]
            print("* %dth sample target: %s" % (i,str(self.outputs[i,1:]-2)))
            print("predict is ", predicted_ids[i][0])
            for j in range(len(tar)):
                if pred[j] == tar[j]:
                    reward = reward + 1
            print(reward)
            print(len(tar))
            reward = float(reward/len(tar))
            print(reward) 
            success = success + reward
            print(success)
                 
        print("step-time %.2f" % (step_time))
        print(pred)
        print("Accuracy ", success/FLAGS.batch_size * 100, "%")

        print("="*20)

    def run(self):
        if self.forward_only:
            self.eval()
        else:
            self.train()

def main(_):
    convexHull = ConvexHull(FLAGS.forward_only)
    convexHull.run()

if __name__ == "__main__":
    tf.app.run()
