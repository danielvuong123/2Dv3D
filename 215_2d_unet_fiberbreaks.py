# -*- coding: utf-8 -*-
"""215_2D_Unet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b7JSJZ-PnRxvXpWlyxi01wo-GnNCRqXv
"""

#This code uses 3D Unet to train a network on 3D subvolumes (64x64x64).
#It also segments a large volume and outputs a multidimensional OMETIFF file
#Custom dataset is used for this code but it should  work on any dataset, including BRATS.

# Commented out IPython magic to ensure Python compatibility.
#Latest Tensorflow (2.4) is giving error for some of the libraries we will be using,
# especially segmentation models 3D.
#Therefore, I am defining TF version 1.x.
#If you have your own 3D unet model, you can try the latest TF version.
# %tensorflow_version 1.x

#Install all dependencies for sgementation-models-3D library.
#We will use this library to call 3D unet.
#Alternative, you can define your own Unet, if you have skills!
# !pip install classification-models
# !pip install efficientnet
# !pip install segmentation-models
#
# #Use patchify to break large volumes into smaller for training
# #and also to put patches back together after prediction.
# !pip install patchify
#
# !pip install --upgrade tensorflow-gpu==1.8.0
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
import os

import tifftools





def merge_directory_tif(directory,name):
    image = None
    count =0
    for filename in os.listdir(directory):
        count+=1
        print(count)
        if image is None:
            image = tifftools.read_tiff(os.path.join(directory, filename))
        else:
            next_image = tifftools.read_tiff(os.path.join(directory, filename))
            image['ifds'].extend(next_image['ifds'])
    tifftools.write_tiff(image,name)

def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def train_2d(images_path,labels_path,model_name,num_epochs,save_graphs=True):
    """
    images_path (str):
        Path to the raw images of fibre breaks
    labels_path (str):
        Path to the masks/labels of the images
    image_resolution (tuple):
        Tuple containing (width,height) in order to patch image correctly
    """
    images = io.imread(images_path)[:64,:704,:1056]
    num_images, height, width = images.shape
    img_patches = patchify(images,(1,height,width),step=1)
    del images
    mask=io.imread(labels_path)[:64,:704,:1056]
    mask_patches = patchify(mask,(1,height,width),step=1)#[num_images,1,1,depth,height,width]
    del mask

    input_img = np.reshape(img_patches,(num_images,height,width))
    input_mask = np.reshape(mask_patches,(num_images,height,width))
    del mask_patches
    del img_patches



    n_classes = 2

    train_img = np.stack((input_img,)*3,axis =-1)
    train_mask = np.expand_dims(input_mask,axis=3)
    del input_img
    del input_mask

    #Current mask has values 0 and 255, we wish to have class labels, thus we
    #change the values to 0 and 1
    train_mask = np.where(train_mask==255,1,train_mask)

    train_mask_cat = to_categorical(train_mask,num_classes=n_classes)

    X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)

    del train_mask
    del train_mask_cat
    del train_img

    encoder_weights = 'imagenet'
    BACKBONE = 'vgg16'  #Try vgg16, efficientnetb7, inceptionv3, resnet50
    activation = 'softmax'
    patch_size = 256
    n_classes = 2
    channels=3

    #Set learning rate and optimizer
    LR = 0.0001
    optim = keras.optimizers.Adam(LR)
    # optim = tf.keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([.5, .5])) #We want to set a very high weight for the fibre and low for the NOT break
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    preprocess_input = sm.get_preprocessing(BACKBONE)

    #Preprocess input data - otherwise you end up with garbage resutls
    # and potentially model that does not converge.
    X_train_prep = preprocess_input(X_train)
    X_test_prep = preprocess_input(X_test)

    del X_train
    del X_test

    #Define the model. Here we use Unet but we can also use other model architectures from the library.
    model = sm.Unet(BACKBONE, classes=n_classes,
                    input_shape=(704, 1056, channels), #Change shape to match 2d shape
                    encoder_weights=encoder_weights,
                    activation=activation)

    model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
    print(model.summary())

    #Fit the model
    history=model.fit(X_train_prep,
              y_train,
              batch_size=1,
              epochs=num_epochs,
              verbose=1,
              validation_data=(X_test_prep, y_test))

    model.save('models/' + model_name)
    # model.save('models/2D_model_vgg16_100epochs.h5')

    if save_graphs:
        #plot the training and validation IoU and loss at each epoch
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plots/' + model_name + '_loss.png')
        # plt.show()

        acc = history.history['iou_score']
        val_acc = history.history['val_iou_score']

        plt.clf()

        plt.plot(epochs, acc, 'y', label='Training IOU')
        plt.plot(epochs, val_acc, 'r', label='Validation IOU')
        plt.title('Training and validation IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.legend()
        plt.savefig('plots/' + model_name + '_IoU.png')



if __name__ == "__main__":
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
    gpus = tf.config.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    print('Found GPU at: {}'.format(device_name))


    sm.set_framework('tf.keras')

    sm.framework()
    train_2d("UoS_SR_A1_1339N_1080_730_1400/full_stack.tif",'UoS_SR_A1_1339N_1080_730_1400_fiberdmg_labels_GV/full_stack_labels.tif','2d_fibre_break_40_epochs.h5',40)
