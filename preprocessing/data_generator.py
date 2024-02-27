import os
import re
import glob as glob
import numpy as np
import cv2 as cv
import tensorflow as tf
import random

def sort_files(file):
    # Trích xuất số từ tên tệp
    number = int(re.search(r'\d+', file).group())
    return number

def video2array(data, det): 
        
    for i in glob.glob(data + '/*'):
        _class = str.split(i, '/')[-1]
        print('class: ', _class)
        os.makedirs(det + '/np_data/' + _class, exist_ok=True)
        sorted_li = [[sorted(os.listdir(list_videos), key=lambda x: int(x[:-4])), list_videos] for list_videos in glob.glob(i + '/*')]
        try:
            block = [
            np.stack([np.stack([cv.imread(list_videos[1] + '/' + list_videos[0][i], 0) for i in range(0, len(list_videos[0]), int(len(list_videos[0]) / 20) if len(list_videos[0]) > 20 else 1)])
            for k in range(int(len(list_videos[0]) / 20 if len(list_videos[0]) > 20 else 1))])
            for list_videos in sorted_li
            ]
        except:
            block = [
                [[cv.imread(list_videos[1] + '/' + list_videos[0][i], 0) for i in range(0, len(list_videos[0]), int(len(list_videos[0]) / 20) if len(list_videos[0]) > 20 else 1)]
                for k in range(int(len(list_videos[0]) / 20 if len(list_videos[0]) > 20 else 1))]
            for list_videos in sorted_li
            ]
        count = 0
        while block:
            tmp = block.pop()
            try:
                print(np.stack(tmp).shape)
            except:
                print('0')
            try:
                if tmp.shape[0] > max_frames:
                    max_frames = tmp.shape[0]
                    print('max frame changed: ', max_frames)
            except:
                if len(tmp) > max_frames:
                    max_frames = len(tmp)
                    print('max frame changed: ', max_frames)
            try:
                np.save(det + '/np_data/' + _class + '/' + str(count), tmp)
            except:
                try:
                    np.save(det + '/np_data/' + _class + '/' + str(count), np.stack(tmp))
                except:
                    continue
            count += 1
    
def padding(data, det, max_len, img_size):
    _list = glob.glob(data + '/*')
    for k in glob.glob(data + '/*'):
        link = det + '/' + str.split(k, '/')[-1]
        os.makedirs(link, exist_ok=True)
        count = 0
        for i in glob.glob(k +'/*'):
            video = np.load(i)
            pad = np.empty((video.shape[0], 0, img_size[0], img_size[1]), dtype=np.uint8)
            if video.shape[1] > max_len:
                imgs = []
                for i in range(max_len):
                    imgs.append(video[:, i + int(video.shape[1] / (max_len - 1) / 2)][:, np.newaxis])
                imgs = np.concatenate(imgs, axis=1)
                np.save(link + '/' + str(count), imgs)
                print(f'shape of {count}.npy - {imgs.shape}')
                count += 1
                continue
            for j in range(20 - video.shape[1]):
                t = random.choice(_list)
                t = np.load(random.choice(glob.glob(t + '/*')))
                idx = random.randint(0, t.shape[1] - 1)
                t = t[:, idx:idx+1, ...]
                pad = np.concatenate((pad, t[:pad.shape[0]]), axis = 1)
            video = np.concatenate((pad, video), axis = 1)
            np.save(link + '/' + str(count), video)
            print(f'shape of {count}.npy - {video.shape}')
            count += 1

def train_test_split(path):
    _class = os.listdir(path)
    list_data = [data + '/' + c + '.npy' for c in _class]
    list_label = [data + '/' + c + '_label.npy' for c in _class]

    data = [np.load(i) for i in list_data]
    label = [np.load(i) for i in list_label]

    train_datas = np.concatenate([i[:int(len(i) * 0.6)] for i in data], axis = 0)
    train_labels =  np.concatenate([i[:int(len(i) * 0.6)] for i in label], axis = 0)

    val_datas = np.concatenate([i[int(len(i) * 0.6):] for i in data], axis = 0)
    val_labels =  np.concatenate([i[int(len(i) * 0.6):] for i in label], axis = 0)

    return (train_datas, train_labels), (val_datas, val_labels)
