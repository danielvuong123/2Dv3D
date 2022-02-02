from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import segmentation_models_3D as sm3
import segmentation_models as sm2
import tensorflow as tf
import keras.losses
from tensorflow import keras
# from keras import load_model

#Declare preprocessing functions, both 2d and 3d
preprocess_input_2D = sm2.get_preprocessing('vgg16')
preprocess_inplut_3D = sm3.get_preprocessing('vgg16')


#Import both models, define custom objects
model_2d = keras.models.load_model('models/2D_model_vgg16_100epochs.h5',custom_objects={"dice_loss_plus_1focal_loss":sm2.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])),
                                                                                        "iou_score":sm2.metrics.IOUScore(threshold=.5),
                                                                                        "f1-score":sm2.losses.CategoricalFocalLoss()})
model_3d = keras.models.load_model('models/3D_model_vgg16_100epochs.h5',custom_objects={"dice_loss_plus_1focal_loss":sm3.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])),
                                                                                        "iou_score":sm3.metrics.IOUScore(threshold=.5),
                                                                                        "f1-score":sm3.losses.CategoricalFocalLoss()})

#Grab entired training block, and mask for IoU computation later
images_training_block = io.imread('sandstone_data_for_ML/data_for_3D_Unet/train_images_256_256_256.tif')
masks_training_block = io.imread('sandstone_data_for_ML/data_for_3D_Unet/train_masks_256_256_256.tif')

#Patchify both images, one in 2d, one in 3d
image_patches_3D = patchify(images_training_block, (64,64,64), step=64)
image_patches_2D = patchify(images_training_block, (1,256,256), step=1)

shape_3D = image_patches_3D.shape
shape_2D = image_patches_2D.shape

#Run the 3d model on all the 64x64x64 patches
predicted_patches_3D = []
for i in range(shape_3D[0]):
    for j in range(shape_3D[1]):
        for k in range(shape_3D[2]):
            single_patch = image_patches_3D[i,j,k,:,:,:]
            single_patch_3ch = np.stack((single_patch,)*3,axis=-1)#adds 3 channels for pixel info
            single_patch_3ch_input = preprocess_input_2D(np.expand_dims(single_patch_3ch,axis=0))
            single_patch_prediction = model_3d.predict(single_patch_3ch_input)
            # print(single_patch_prediction.shape)
            single_patch_prediction_argmax = np.argmax(single_patch_prediction, axis=4)[0,:,:,:]
            predicted_patches_3D.append(single_patch_prediction_argmax)

#Convert the predictions back into the original 256x256x256 shape
predicted_patches_3D = np.array(predicted_patches_3D)
predicted_patches_3D_reshaped = np.reshape(predicted_patches_3D,image_patches_3D.shape)
reconstructed_image_3D = unpatchify(predicted_patches_3D_reshaped,images_training_block.shape)

#Run the 2d model on all the 256x256 images
predicted_patches_2D = []
for i in range(shape_2D[0]):
    single_patch = image_patches_2D[i,0,0,0,:,:]
    single_patch_3ch = np.stack((single_patch,)*3,axis=-1)
    single_patch_3ch_input = preprocess_input_2D(np.expand_dims(single_patch_3ch,axis=0))
    single_patch_prediction = model_2d.predict(single_patch_3ch_input)
    single_patch_prediction_argmax = np.argmax(single_patch_prediction,axis = 3)[0,:,:]
    predicted_patches_2D.append(single_patch_prediction_argmax)

#Convert the predictions back into the 256x256x256 shape
predicted_patches_2D = np.array(predicted_patches_2D)
predicted_patches_2D_reshaped = np.reshape(predicted_patches_3D,image_patches_2D.shape)
reconstructed_image_2D = unpatchify(predicted_patches_2D_reshaped,images_training_block.shape)

#We define the union as (true_positives + false_positives + false_negatives).
#This is equal to (correct predictions + incorrect predictions) given the way
#we define these labels
#We then define the intersection as the true true_positives, then place these
#values over each other to compute the final IoU score.
union = 0
intersection_2D = 0
intersection_3D = 0
for i in range(masks_training_block.shape[0]):
    for j in range(masks_training_block.shape[1]):
        for k in range(masks_training_block.shape[2]):
            if reconstructed_image_2D[i,j,k] == masks_training_block[i,j,k]:
                intersection_2D += 1
            if reconstructed_image_3D[i,j,k] == masks_training_block[i,j,k]:
                intersection_3D += 1
            union += 1

print("Computed 2D IoU score: " + str(intersection_2D/union))
print("Computed 3D IoU score: " + str(intersection_3D/union))
