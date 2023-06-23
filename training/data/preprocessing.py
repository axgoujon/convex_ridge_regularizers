import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob


def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    print(img.shape)
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(patch_size, stride, aug_times=1, image_type='BSD'):
    '''
       - image_type: 'BSD' or 'CT' or 'MRI'
    '''
    # train
    print(patch_size)
    print('process training data')
    if image_type == 'BSD':
        scales = [1, 0.9, 0.8, 0.7]
    else:
        scales = [1]

    files = glob.glob(f'./images/{image_type}/train/*.png')

    
    
    files.sort()

    output_directory_train = f'preprocessed/{image_type}'
    if not os.path.exists(output_directory_train):
        os.makedirs(output_directory_train)
    h5f = h5py.File(f'{output_directory_train}/train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()

    # val
    print('\nprocess validation data')
    files.clear()
    if image_type == 'BSD':
        files = glob.glob(f'./images/{image_type}/Set12/*.png')
    else:
        files = glob.glob(f'./images/{image_type}/validation/*.png')
    
    files.sort()
    output_directory_val = f'preprocessed/{image_type}'
    if not os.path.exists(output_directory_val):
        os.makedirs(output_directory_val)
    h5f = h5py.File(f'{output_directory_val}/validation.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()

    # test
    if image_type == 'BSD':
        print('\nprocess test data')
        files.clear()
        files = glob.glob(os.path.join('./images/BSD/Set68', '*.png'))
        files.sort()
        output_directory_test = f'preprocessed/{image_type}'
        if not os.path.exists(output_directory_test):
            os.makedirs(output_directory_test)
        h5f = h5py.File(f'{output_directory_test}/test.h5', 'w')
        test_num = 0
        for i in range(len(files)):
            print("file: %s" % files[i])
            img = cv2.imread(files[i])
            img = np.expand_dims(img[:,:,0], 0)
            img = np.float32(normalize(img))
            h5f.create_dataset(str(test_num), data=img)
            test_num += 1
        h5f.close()
    else:
        test_num = 0

    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)
    print('test set, # samples %d\n' % test_num)


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


if __name__ == "__main__":
    image_type = "MRI"
    if image_type == 'MRI':
        stride = 5
        aug_times = 1
    else:
        stride = 10
        aug_times = 1
    prepare_data(patch_size=40, stride=stride, aug_times=aug_times, image_type="MRI")