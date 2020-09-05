import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
import paddle
# from model import TSN1
# from reader import KineticsReader


from TSN1 import TSNResNet

from ECO_paddle import ECOfull


from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import numpy as np

import random
def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group

def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
        "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images

def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
            "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs
def imgs_transform(imgs, label, mode='train', seg_num=32, seglen=1, short_size=224,
                           target_size=224, img_mean=0.456, img_std=0.225):
#             imgs =  list(np.array(img) for img in imgs)
            imgs = group_scale(imgs, short_size)

            if mode == 'train':
                
                imgs = group_random_crop(imgs, target_size)
                imgs = group_random_flip(imgs)

                #添加数据增强部分，提升分类精度
            else:
                imgs = group_center_crop(imgs, target_size)

            np_imgs = (np.array(imgs[0]).astype('float32').transpose(
                (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
            for i in range(len(imgs) - 1):
                img = (np.array(imgs[i + 1]).astype('float32').transpose(
                    (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
                np_imgs = np.concatenate((np_imgs, img))
            imgs = np_imgs
            imgs -= img_mean
            imgs /= img_std
            imgs = np.reshape(imgs,
                              (seg_num, seglen * 3, target_size, target_size))

            return imgs, label


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class TSNDataSet():
    def __init__(self, root_path, list_file,
                 num_segments=32, new_length=1, modality='RGB',
                 image_tmpl='{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        
        if test_mode:
            self.mode = 'test'
        else:
            self.mode = 'train'

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        # try:
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        imgs,label = self.get(record, segment_indices)
        label_onehot = np.zeros([101],dtype=np.int32)
        label_onehot[label] = 1
        return imgs_transform(imgs,label,self.mode)

        # except Exception as e:
        #     print('1'*100)
        #     # input()
        #     print(record.num_frames)
        #     # print(self.video_list)
        #     # print(index)
        #     print(e)
        #     print(self.list_file)
        #     print('2'*100)
        #     raise Exception

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                # print(record.num_frames)
                # try:
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
                # except:
                #     break

        process_data = images#self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)



logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    # parser.add_argument(
    #     '--config',
    #     type=str,
    #     default='configs/tsn.txt',
    #     help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # config = parse_config(args.config)
        # train_config = merge_configs(config, 'train', vars(args))
        # print_configs(train_config, 'Train')

        #根据自己定义的网络，声明train_model
        train_model = ECOfull()
        # train_model.fc_final = nn.Linear


        opt = fluid.optimizer.Momentum(0.001, 0.9, parameter_list=train_model.parameters())

        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/tsn_model')
            train_model.load_dict(model)

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        # train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        dataset = TSNDataSet("", 'list/ucf101_train_split1.txt')

        def reader():
            dataset = TSNDataSet("", 'list/ucf101_train_split1.txt')
            for i in range(dataset.__len__()):
                yield dataset.__getitem__(i)
        # dataset = TSNDataSet()
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader=reader,buf_size=200
            ), batch_size=args.batch_size
)





        epochs = args.epoch or train_model.epoch_num()
        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')

                # y_data = np.reshape(y_data,[y_data.shape[0],y_data.shape[2]])
                
                # print('*'*40)
                # print('x:',dy_x_data.shape)
                # print('y:',y_data.shape)

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                
                # print('*label*'*10)
                # print(label)
                
                out, acc = train_model(img, label)

                # print('*out*'*10)
                # print(out)

                
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()
                
                
                if batch_id % 1 == 0:
                    logger.info("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/tsn_model')
        logger.info("Final loss: {}".format(avg_loss.numpy()))
        print("Final loss: {}".format(avg_loss.numpy()))
                


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
