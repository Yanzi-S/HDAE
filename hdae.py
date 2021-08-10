import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from library.Autoencoder import Autoencoder
import argparse
from data_loader_1d import Data
import time

parser = argparse.ArgumentParser(description='Denoising for HSI')
# path
parser.add_argument('--result',dest='result',default='')
parser.add_argument('--log',dest='log',default='')
parser.add_argument('--model',dest='model',default='')
parser.add_argument('--tfrecords',dest='tfrecords',default='')
# data name and path
parser.add_argument('--data_name',dest='data_name',default='')
parser.add_argument('--data_path',dest='data_path',\
                    default="")
parser.add_argument('--locx',dest='locx',default=)
parser.add_argument('--locy',dest='locy',default=)
# noise added to HSI
parser.add_argument('--SNR',dest='SNR',default=)
# network parameter setup
parser.add_argument('--training_epochs',dest='training_epochs',default=)
parser.add_argument('--batch_size',dest='batch_size',default=)
parser.add_argument('--corruption_level',dest='corruption_level',default=)
#parser.add_argument('--sparse_reg',dest='sparse_reg',default=)
parser.add_argument('--balance_wei',dest='balance_wei',default=)
parser.add_argument('--n_inputs',dest='n_inputs',default=)
parser.add_argument('--n_hidden',dest='n_hidden',default=)
parser.add_argument('--lr',dest='lr',default=)
parser.add_argument('--thre',dest='thre',default=)

# model loading
parser.add_argument('--model_name',dest='model_name',default='ae.ckpt')
parser.add_argument('--load_model',dest='load_model',default="")

args = parser.parse_args()
args.filename = '-'.join([str(args.data_name),str(args.thre),str(args.training_epochs),str(args.lr),\
                    str(args.batch_size),str(args.corruption_level),str(args.balance_wei),\
                    str(args.sparse_reg),str(args.n_inputs),str(args.n_hidden)])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = Data(args)
dataset.read_data()
train_dataset = dataset.data_parse(os.path.join(args.filename,args.tfrecords, 'train_data.tfrecords'), type='train')
test_dataset = dataset.data_parse(os.path.join(args.filename,args.tfrecords, 'test_data.tfrecords'), type='test')
val_dataset = dataset.data_parse(os.path.join(args.filename,args.tfrecords, 'val_data.tfrecords'), type='val')

tgt=sio.loadmat(os.path.join(args.filename,args.result,'d.mat'))
prior = tgt['d']
ae = Autoencoder(n_layers=[args.n_inputs, args.n_hidden], prior = prior,
                          transfer_function = tf.nn.sigmoid,
                          optimizer = tf.train.AdamOptimizer(learning_rate = args.lr,beta1=0.8),ae_para = [args.corruption_level, args.sparse_reg, args.balance_wei])
var_list=tf.trainable_variables()
saver = tf.train.Saver(var_list=var_list, max_to_keep=100)
init = tf.global_variables_initializer()
sess = tf.Session()
summary_write = tf.summary.FileWriter(os.path.join(args.filename,args.log),graph=sess.graph)
sess.run(init)

if not args.load_model:
    train_data_iter = sess.run(train_dataset)
    test_data_iter = sess.run(test_dataset)
    val_data_iter = sess.run(val_dataset)
    ftxt = open(os.path.join(args.filename,args.result,'time.txt'),'a')
    for itera in range(10):
        print("Iteration:", '%d,' % (itera))
        n_samples = train_data_iter.shape[0]
        train_start = time.clock()
        for epoch in range(args.training_epochs):
            avg_cost = 0
            index = list(range(train_data_iter.shape[0]))
            np.random.shuffle(index)
            total_batch = int(n_samples / args.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs_train=train_data_iter[index[i*args.batch_size:(i+1)*args.batch_size]]
        
                # Fit training using batch data
                temp = ae.partial_fit()
                cost, opt = sess.run(temp, feed_dict={ae.x: batch_xs_train, ae.keep_prob : ae.in_keep_prob})
                summery = sess.run(ae.cost_summary(), feed_dict={ae.x: batch_xs_train, ae.keep_prob : ae.in_keep_prob})
                sigma1,sigma2 = sess.run(ae.sigma(), feed_dict={ae.x: batch_xs_train, ae.keep_prob : ae.in_keep_prob})
                if np.max(sigma1)>0.98:
                    print("maximum sigma1 is:", '%f,' % (np.max(sigma1)))
                if np.max(sigma2)>0.99:
                    print("maximum sigma2 is:", '%f,' % (np.max(sigma2)))
                # Compute average loss
                avg_cost += cost / n_samples * args.batch_size
                
            # Display logs per epoch step
            if epoch % 1 == 0:
                print("Epoch:", '%d,' % (epoch + 1),
                      "Cost:", "{:.9f}".format(avg_cost))
            if epoch == 0 or (epoch+1) % 10 == 0:
                if not os.path.exists(os.path.join(args.filename,args.model,'_'.join(['iteration',str(itera)]))):
                    os.makedirs(os.path.join(args.filename,args.model,'_'.join(['iteration',str(itera)])))
                saver.save(sess,os.path.join(args.filename,args.model,'_'.join(['iteration',str(itera)]),args.model_name),global_step=epoch)
                print('model saved')
            summary_write.add_summary(summery,epoch)
            if epoch % 2 == 0:
                noisy_x_train = sess.run(ae.show_noisy_x(), feed_dict={ae.x: batch_xs_train, ae.keep_prob : ae.in_keep_prob})        
                recover_xs_train = sess.run(ae.reconstruct(), feed_dict={ae.x: batch_xs_train, ae.keep_prob : ae.in_keep_prob})        
                for j in range(0):
                    p1, = plt.plot(batch_xs_train[j], 'r')
                    p2, = plt.plot(noisy_x_train[j], 'g')
                    p3, = plt.plot(recover_xs_train[j], 'b')
                    plt.legend([p1, p2, p3], ['input', 'noisy', 'recons'], loc = 'upper left')
                    plt.show()
        train_end = time.clock()
        # validating
#        val_data0 = np.copy(val_data_iter)
        #val_data += np.sqrt(1/(10**(args.SNR/10.0))) * np.random.normal(loc=0, scale=1, size=val_data.shape)
        ae_val_cost = sess.run(ae.calc_total_cost(), feed_dict={ae.x: val_data_iter, ae.keep_prob : 1.0})
        recons_val_data = sess.run(ae.reconstruct(), feed_dict={ae.x: val_data_iter, ae.keep_prob : 1.0})        
        for j in range(2):
            plt.figure()
#            p1, = plt.plot(val_data0[j], 'r')
            p2, = plt.plot(val_data_iter[j], 'r')
            p3, = plt.plot(recons_val_data[j], 'b')
            plt.legend([p2, p3], ['input', 'recons'], loc = 'upper left')
            plt.show()
        print("Total cost: " + str(ae_val_cost))
        print("Avg cost: " + str(ae_val_cost/val_data_iter.shape[0]))
        # denoising data generation
        batch_xs_test = test_data_iter
        test_start = time.clock()
        recover_xs_test = sess.run(ae.reconstruct(), feed_dict={ae.x: batch_xs_test, ae.keep_prob : 1.0})
        test_end = time.clock()
        noisy_x_test = np.copy(batch_xs_test)
        noisy_x_test += np.sqrt(1/(10**(args.SNR/10.0))) * np.random.normal(loc=0, scale=1, size=noisy_x_test.shape)
        recover_noisy_xs_test = sess.run(ae.reconstruct(), feed_dict={ae.x: noisy_x_test, ae.keep_prob : 1.0})
        for j in range(10):
            plt.figure()
            p1, = plt.plot(batch_xs_test[j*500], 'r')
            p2, = plt.plot(recover_xs_test[j*500], 'b')
            p3, = plt.plot(noisy_x_test[j*500], 'r', linestyle="--")
            p4, = plt.plot(recover_noisy_xs_test[j*500], 'b', linestyle="--")
            plt.legend([p1, p2, p3, p4], ['input', 'recons', 'noisy', 'recons_noisy'], loc = 'upper left')
            plt.show()        
        sio.savemat(os.path.join(args.filename,args.result,''.join(['recover_hsi_',str(itera),'.mat'])),{
            'recover_hsi':recover_xs_test,
        })    
        sio.savemat(os.path.join(args.filename,args.result,''.join(['input_hsi_',str(itera),'.mat'])),{
            'input_hsi':batch_xs_test,
        })
        sio.savemat(os.path.join(args.filename,args.result,''.join(['recover_noisy_hsi_',str(itera),'.mat'])),{
            'recover_noisy_hsi':recover_noisy_xs_test,
        })
        sio.savemat(os.path.join(args.filename,args.result,''.join(['noisy_hsi_',str(itera),'.mat'])),{
            'noisy_hsi':noisy_x_test,
        })
        res = np.mean(np.square(recover_xs_test - batch_xs_test))
        print("Residual value between neighboring denoising HSI: " + str(res))
        idtrain = list(range(recover_xs_test.shape[0]))
        np.random.shuffle(idtrain)
        train_data_iter = recover_xs_test[idtrain[0: np.ceil(recover_xs_test.shape[0]*0.5).astype(np.int)]]
        val_data_iter = recover_xs_test[idtrain[np.ceil(recover_xs_test.shape[0]*0.5).astype(np.int):recover_xs_test.shape[0]]]
        test_data_iter = recover_xs_test
        ftxt.write("\nIteration {}: ".format(itera))
        ftxt.write("\nTrain time: {}, test time: {}".format(train_end - train_start, test_end - test_start))
        if  res < args.thre:
            break        
    ftxt.close()

# loading model
else:
    ftxt = open(os.path.join(args.filename,args.result,'time.txt'),'a')
    number = len([lists for lists in os.listdir(os.path.join(args.filename,args.model)) if os.path.isdir(os.path.join(os.path.join(args.filename,args.model), lists))])
    test_data_iter = sess.run(test_dataset)
    for itera in range(number):
        print("Loading model in iteration {}".format(itera))
        model_name = os.path.join(args.filename,args.model,'_'.join(['iteration',str(itera)]))
        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            try:
                saver.restore(sess, os.path.join(model_name, ckpt_name))
                print("Load successful.")
            except ValueError:
                sess.run(tf.global_variables_initializer())
        else:
            print("Load fail!!!")
        # testing
        noisy_x = np.copy(test_data_iter)
        noisy_x += np.sqrt(1/(10**(args.SNR/10.0))) * np.random.normal(loc=0, scale=1, size=noisy_x.shape)
        #noisy_x = sess.run(ae.show_noisy_x(), feed_dict={ae.x: batch_xs, ae.keep_prob : ae.in_keep_prob})        
        test_start = time.clock()
        recover_xs = sess.run(ae.reconstruct(), feed_dict={ae.x: test_data_iter, ae.keep_prob : 1.0})        
        test_end = time.clock()
        recover_noisy_xs = sess.run(ae.reconstruct(), feed_dict={ae.x: noisy_x, ae.keep_prob : 1.0})        
        for j in range(10):
            plt.figure()
            p1, = plt.plot(test_data_iter[j*500], 'r')
            p2, = plt.plot(recover_xs[j*500], 'b')
            p3, = plt.plot(noisy_x[j*500], 'r', linestyle="--")
            p4, = plt.plot(recover_noisy_xs[j*500], 'b', linestyle="--")
            plt.legend([p1, p2, p3, p4], ['input', 'recons', 'noisy', 'recons_noisy'], loc = 'upper left')
            plt.show()        
        
        sio.savemat(os.path.join(args.filename,args.result,''.join(['recover_hsi_',str(itera),'.mat'])),{
            'recover_hsi':recover_xs,
        })
        sio.savemat(os.path.join(args.filename,args.result,''.join(['input_hsi_',str(itera),'.mat'])),{
            'input_hsi':test_data_iter,
        })
        sio.savemat(os.path.join(args.filename,args.result,''.join(['recover_noisy_hsi_',str(itera),'.mat'])),{
            'recover_noisy_hsi':recover_noisy_xs,
        })
        sio.savemat(os.path.join(args.filename,args.result,''.join(['noisy_hsi_',str(itera),'.mat'])),{
            'noisy_hsi':noisy_x,
        })
        test_data_iter = recover_xs
        ftxt.write("\nIteration {}: ".format(itera))
        ftxt.write("\ntest time: {}".format(test_end - test_start))
    ftxt.close()

