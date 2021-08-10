import pathlib
import scipy.io as sio
import numpy as np
import tensorflow as tf
import os

def max_min(x):
    m=np.min(x,axis=0)
    res=np.max(x,axis=0)
    return (x-np.tile(m[:,:,np.newaxis],[1,1,x.shape[2]]))/(np.tile(res[:,:,np.newaxis],[1,1,x.shape[2]]))-np.tile(m[:,:,np.newaxis],[1,1,x.shape[2]]))
#    return (x-np.min(x))/(np.max(x)-np.min(x))

class Data():

    def __init__(self,args):
        self.args = args
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.locx = args.locx
        self.locy = args.locy
        self.result = args.result
        self.tfrecords = args.tfrecords
        self.filename = args.filename
        self.snr=args.SNR
        self.data_dict=sio.loadmat(str(pathlib.Path(self.data_path, self.data_name + '.mat')))
        data_name = [t for t in list(self.data_dict.keys()) if not t.startswith('__')][0]
        data_gt_name = [t for t in list(self.data_dict.keys()) if not t.startswith('__')][1]
        self.data = self.data_dict[data_name]
        self.data = max_min(self.data).astype(np.float32)
        self.data_gt = self.data_dict[data_gt_name].astype(np.int64)
        self.height=self.data.shape[0]
        self.width=self.data.shape[1]
        self.dim = self.data.shape[2]

            
    def read_data(self):
        data = self.data
        data_gt = self.data_gt
        # add noise with SNR to HSI
#        if self.snr:
#            data=self.add_noise(data,self.snr)
        if not os.path.exists(os.path.join(self.filename,self.result)):
            os.makedirs(os.path.join(self.filename,self.result))
        sio.savemat(os.path.join(self.filename,self.result,''.join(['_'.join(['data','SNR']),str(self.snr),'.mat'])),{
        'data':data,
        })
        sio.savemat(os.path.join(self.filename,self.result,'groundtruth.mat'),{
            'groundtruth':self.data_gt,
        })
        sio.savemat(os.path.join(self.filename,self.result,'d.mat'),{
            'd':data[self.locx,self.locy,:],
        })

        data_pos = list()
        for i in range(data_gt.shape[0]):
            for j in range(data_gt.shape[1]):
                data_pos.append([i, j])
        self.data_pos = data_pos
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # training patches
        if not os.path.exists(os.path.join(self.filename,self.tfrecords)):
            os.makedirs(os.path.join(self.filename,self.tfrecords))
        train_data_name = os.path.join(self.filename,self.tfrecords, 'train_data.tfrecords')
        writer_train = tf.python_io.TFRecordWriter(train_data_name)
        val_data_name = os.path.join(self.filename,self.tfrecords, 'val_data.tfrecords')
        writer_val = tf.python_io.TFRecordWriter(val_data_name)
        test_data_name = os.path.join(self.filename,self.tfrecords, 'test_data.tfrecords')
        writer_test = tf.python_io.TFRecordWriter(test_data_name)

#        if not os.path.exists(os.path.abspath(test_data_name)):
        for j in data_pos:
            [r,c]=j
            pixel_t = self.data[r,c].astype(np.float32).tostring()
            example_test = tf.train.Example(features=(tf.train.Features(
                feature={
                    'testdata': _bytes_feature(pixel_t),
                }
            )))
            writer_test.write(example_test.SerializeToString())                
        writer_test.close()

#        if not (os.path.exists(os.path.abspath(train_data_name)) and os.path.exists(os.path.abspath(val_data_name))):
        np.random.shuffle(data_pos)
        kk = 0
        for i in data_pos:
            [r,c]=i
            pixel_t = self.data[r,c].astype(np.float32).tostring()
            if kk < np.ceil(self.data.shape[0]*self.data.shape[1]*0.5):
                example_train = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'traindata': _bytes_feature(pixel_t),
                    }
                )))
                writer_train.write(example_train.SerializeToString())
            else:
                example_val = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'valdata': _bytes_feature(pixel_t),
                    }
                )))
                writer_val.write(example_val.SerializeToString())
            kk = kk + 1                              
        writer_train.close()
        writer_val.close()

    def data_parse(self,filename,type='train'):
        dataset = tf.data.TFRecordDataset([filename])
        def parser_train(record):
            keys_to_features = {
                'traindata': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            train_data = tf.decode_raw(features['traindata'], tf.float32)
            return train_data
        def parser_val(record):
            keys_to_features = {
                'valdata': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            val_data = tf.decode_raw(features['valdata'], tf.float32)
            return val_data
        def parser_test(record):
            keys_to_features = {
                'testdata': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            test_data = tf.decode_raw(features['testdata'], tf.float32)
            return test_data

        if type == 'train':
            dataset = dataset.map(parser_train)
            dataset = dataset.batch(np.ceil(self.data.shape[0]*self.data.shape[1]*0.5))
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'val':
            dataset = dataset.map(parser_val)
            dataset = dataset.batch(self.data.shape[0]*self.data.shape[1]-np.ceil(self.data.shape[0]*self.data.shape[1]*0.5))
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'test':
            dataset = dataset.map(parser_test)
            dataset = dataset.batch(self.data.shape[0]*self.data.shape[1])
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()