
1. Simple mnist_classification_tf mnist digit classification using tf 2.0
2. Simple RNN used for predicting next data poitns for given last 10 points.
  This is used for regression value for given Sin wave data.
3. TwoSinWaves is the input for TxD (D=2) input data to predict one value. 
   You can combine different operations like +,- and *

4. Linear_Classification_Breast_Cancer:
  This is linearClassfier : 
  The data set maps the multiple features of Breast Cancer attributes to classifies whether a person has breast cancer or not.
  Simple ANN can be used. No need to have CNN. It is also fine to use CNN in this case. 

5. TF_CNN_FMNIST_Image_classification.ipynb - Simple multi layered cov layers with softmax. there are dropout layers at the classification layer.

6. TF_CIFAR_image_classification - simple CNN with conv layers, dropout and softmax. Note that CIFAR is a color images. The accuracy is low.

7. TF_CIFAR_CNN_along_with_Image_Generation_and_NormalizationLayers - 
   this is got cov along with Max pool , along with normalization is included.
   this machine's accuracy is improved.
   ImageDataGenerator is used to generate multiple images from CIFAR imagedata set (not from directory). The model is trained using fit_generator() function.

8. TF_Transfer_VGG16_extending_to_Food_Non_Food_database

   uses tranfer model. VGG16 is used for tranformation layer (pre-trained model) and final classifier is added for classifing food and non-food images with different sizes.
   ImageDataGenerators are used. Here the images are stored in specific directory structure. 
   In addition to fit_generation, one can do Fit_evaluation for checking the accuracy. In this case, we have used training data itself is used.
   However, one can use evaluation images for checking the final accuracy.

9. ImageClassification_Keras
   This is simple image classifications for cats and dogs. 
   it uses pure Keras API. It will look like Tensorflow. 
   Note that you need to have Keras in the evironment.
   it has some environment ....
   model.save_weights is the function to pick it from this file.
   Also, not that data to be downloaded and set up in specific directory structure
   so that ImageDataGenerators can be used.


