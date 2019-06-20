#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time,glob,re
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G
from config import config
import cv2
from PSNR import psnr

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128
range_step=2*4096/65535 #this is the quatilization step which is normalized to 1
in_weight=0 #in_weight and out weight is the punishment factor for the diff=gen-zp in the range of [0,4096] and out of that range
out_weight=65535

ni = int(np.sqrt(batch_size))

def train():
    # create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    # load dataset
    train_hr_img_list=glob.glob(os.path.join(config.TRAIN.hr_img_path,"*"))
    train_hr_img_list=[f for f in train_hr_img_list if os.path.splitext(f)[-1]=='.png']

    ## If your machine have enough memory, please pre-load the whole train set.
    #train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_hr_imgs=[]
    #use tensorflow to load image in form of RGB
#    for f in train_hr_img_list:
#        img=tf.io.read_file(f)
#        img=tf.image.decode_png(img,channels=3,dtype=tf.uint16)
#        max_value=np.max(img)
#        print(max_value)
#        train_hr_imgs.append(img)
    #use opencv to load image in form of BGR (Blue,Green,Red), we decide to use this way because we can use cv2.imshow() which support 16bit image
    for f in train_hr_img_list:
        img=cv2.imread(f,cv2.IMREAD_UNCHANGED)
        max_value=np.max(img)
        if max_value==0:
            continue
        #print(max_value)
        train_hr_imgs.append(img)

    # dataset API and augmentation
    def generator_train():
        for img in train_hr_imgs:
            yield img
    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [96,96,3])
        # Randomly flip the image horizontally.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        hr_patch = tf.image.random_flip_up_down(hr_patch)
        # Randomly adjust hue, contrast and saturation.
        hr_patch = tf.image.random_hue(hr_patch, max_delta=0.05)
        hr_patch = tf.image.random_contrast(hr_patch, lower=0.3, upper=1.0)
        hr_patch = tf.image.random_brightness(hr_patch, max_delta=0.2)
        hr_patch = tf.image.random_saturation(hr_patch, lower=0.0, upper=2.0)
        hr_patch = tf.image.rot90(hr_patch, np.random.randint(1,4))
        
        lr_patch = tf.floor(hr_patch/4096.0) #compress 16bit image to 4bit. caution!!! do not divide a int number,or you will get all zero result
        lr_patch = lr_patch*4096.0  #padding 4bit with zeros to 8bit
        #make the value of pixel lies between [-1,1]
        hr_patch = hr_patch / (65535. / 2.)
        hr_patch = hr_patch - 1.  
        
        lr_patch = lr_patch / (65535. / 2.)
        lr_patch = lr_patch - 1.  
        return lr_patch, hr_patch
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    #train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    #train_ds = train_ds.repeat(2)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=4096) 
    train_ds = train_ds.batch(batch_size)
    # value = train_ds.make_one_shot_iterator().get_next()

    # obtain models
    G = get_G((batch_size, None, None, 3)) # (None, 96, 96, 3)
    print('load VGG')
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    '''print(G)
    print(VGG)'''

    # G.load_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode'])) # in case you want to restore a training
    # D.load_weights(checkpoint_dir + '/d_{}.h5'.format(tl.global_flag['mode']))

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(mse_loss, var_list=g_vars)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(g_loss, var_list=g_vars)

    G.train()
    VGG.train()
    
    n_step_epoch = round(n_epoch_init // batch_size)
    for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
        step_time = time.time()
        with tf.GradientTape() as tape:
            fake_hr_patchs = G(lr_patchs)
            mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
        grad = tape.gradient(mse_loss, G.trainable_weights)
        g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
        step += 1
        epoch = step//n_step_epoch
        print("Epoch: [{}/{}] step: [{}/{}] time: {}s, mse: {} ".format(
            epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        #if (epoch != 0) and (epoch % 10 == 0):
            #tl.vis.save_images(fake_hr_patchs.numpy(), [ni, ni], save_dir_gan + '/train_g_init_{}.png'.format(epoch))

    # initialize learning (G)
    for epoch in range(120):
        train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
        train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        #train_ds = train_ds.repeat(n_epoch_init + n_epoch)
        #train_ds = train_ds.repeat(1)
        train_ds = train_ds.shuffle(shuffle_buffer_size)
        train_ds = train_ds.prefetch(buffer_size=4096) 
        train_ds = train_ds.batch(batch_size)
        n_step_epoch = round(n_epoch_init // batch_size)
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
                feature_fake = VGG((fake_hr_patchs+1)/2.)
                feature_real = VGG((hr_patchs+1)/2.)
                vgg_loss = 0.6* tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                diff=fake_hr_patchs-lr_patchs
                range_loss=tf.reduce_sum(tf.where((tf.greater(diff,range_step) | tf.less(diff,0)),tf.abs(diff)*out_weight,diff*in_weight))
                #range_loss=tf.reduce_sum(tf.where((tf.greater(diff,range_step) | tf.less(diff,0) ),out_weight,in_weight))
                g_loss=0.000005*range_loss+vgg_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            
            #epoch = step//n_step_epoch
            if step % 100 ==0:
                print("Epoch: [{}/{}] step: [{}/{}] time: {}s, mse: {}  vgg_loss: {}".format(
                    epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss,vgg_loss))
            step += 1
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay**(epoch // decay_every)
                lr_v.assign(lr_init * new_lr_decay)
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
    
            if (epoch != 0) and ((epoch+1) % 5 == 0):
                #tl.vis.save_images(fake_hr_patchs.numpy(), [ni, ni], save_dir_gan + '/train_g_{}.png'.format(epoch))
                G.save_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode']))


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###

    valid_hr_img_list=glob.glob(os.path.join(config.VALID.hr_img_path,"*"))
    valid_hr_img_list=[f for f in valid_hr_img_list if os.path.splitext(f)[-1]=='.png']
    #valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    valid_hr_imgs=[]
    for f in valid_hr_img_list:
        #print(f)
        img=cv2.imread(f,cv2.IMREAD_UNCHANGED)
        #print(img[1:10,1:10,0])
        max_value=np.max(img)
        print('max_value:{}'.format(max_value))
        valid_hr_imgs.append(img)
    
    ###========================== DEFINE MODEL ============================###
    G = get_G([1, None, None, 3])
    #G.load_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode']))
    G.load_weights(checkpoint_dir + '/g_{}.h5'.format('srgan'))
    G.eval()
    for imid in range(8):
        #imid =7
        
        valid_hr_img = valid_hr_imgs[imid]
        valid_hr_img=cv2.resize(valid_hr_img,(500,218),interpolation=cv2.INTER_CUBIC)
        valid_lr_img = tf.floor(valid_hr_img/4096.0)
        valid_lr_img= valid_lr_img*4096.0
    
        valid_lr_img = (valid_lr_img / 32767.5) - 1  # rescale to ［－1, 1]
        valid_lr_img=tf.cast(valid_lr_img,dtype=tf.float32)
        valid_lr_img_input=tf.reshape(valid_lr_img,[1,valid_lr_img.shape[0],valid_lr_img.shape[1],valid_lr_img.shape[2]])
        # print(valid_lr_img.min(), valid_lr_img.max())
#        print(valid_lr_img.shape)
    
#        G = get_G([1, None, None, 3])
#        #G.load_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode']))
#        G.load_weights(checkpoint_dir + '/g_{}.h5'.format('srgan'))
#        G.eval()
    
        out = G(valid_lr_img_input).numpy()
        print(np.min(out), np.max(out))
#        cv2.imshow('out',out[0])
#        cv2.waitKey(0)
    
#        print("[*] save images")
#        tl.vis.save_image(out[0], save_dir + '/valid_gen_{}.png'.format(str(imid)))
#        tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr_{}.png'.format(str(imid)))
#        tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr_{}.png'.format(str(imid)))
        '''do not use tl.vis.save_image,for which do not support 16bit mode'''
        out=tf.cast((out[0]+1)*32767.5,dtype=tf.uint16).numpy()
        #out=out.eval()
        valid_lr_img=tf.cast((valid_lr_img+1)*32767.5,dtype=tf.uint16).numpy()
        valid_hr_img=tf.cast(valid_hr_img,dtype=tf.uint16).numpy()
#        cv2.imshow('out',out)
#        cv2.waitKey(0)
#        print(np.max(out))
#        print(np.min(out))
        psnr_gen=psnr(np.float32(out),np.float32(valid_hr_img),2)
        psnr_zp=psnr(np.float32(valid_lr_img),np.float32(valid_hr_img),2)
        print('psnr_gen:{}   psnr_zp:{}'.format(psnr_gen,psnr_zp))
        cv2.imwrite(save_dir + '/valid_gen_{}.png'.format(str(imid)),out,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        cv2.imwrite(save_dir + '/valid_lr_{}.png'.format(str(imid)),valid_lr_img,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        cv2.imwrite(save_dir + '/valid_hr_{}.png'.format(str(imid)),valid_hr_img,[int(cv2.IMWRITE_PNG_COMPRESSION),0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        print('trainning')
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
