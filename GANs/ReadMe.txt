

SimpleGAN_tf:
It has two NN:
 1) Generator : It generates images from noise as input.
 2) Discriminator : It classifies as images into real or fake. It can be trained using real images as "real" and noise images generated from Generator as Fake images. It is a simple classifier. It is trained using both real and noise (fake) images as part of training. During this training, weights are modified such that the loss is low.
 3) We define a combination of Generator and Discriminator which takes input as noise, passes this through entire combination of generator and discriminator and finally classifies as real or fake images. The combined model is a classifier.
e.i. noise is converted into image by generator; the image is classified as real or fake by discriminator.
 
  During the training, we pass the noise to Generator which producess the fake images. These images are fed into discriminator. the combined model is designed as a classifier. The input fake images were told as real to combined model during the training. Also, we freeze the discriminator. (The wieghts will not get udpated while combined model is  trained). Only Generator weights are updated.

The whole training is done such the both training of Discrinimator and Generators are done alternatively(seuentially in the same loop) with a small batch of images and noise.

Intitution:
D will improve its classification knowledge  after every batch of training. i.e. its wieghts will change.
G will learn to generate better images with the knoweldge of classification of D. 
Note that D weights still will be working during the classification. However, during the backprop, we only modify the generator weights.

Question: Can we generate a specific class of images ? Currently it  is generated as randomly for given input of noise.



