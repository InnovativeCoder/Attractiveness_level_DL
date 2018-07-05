Need to make a prediction model for prediction the attractiveness of the person, this project uses SCUT FBP 500 dataset.

APPROACH OF MODEL---->>
----------------

1. Take the dataset, preprocess it and data augmentation, to increase the small dataset.
2. Make neural network for prediction, improve the accuracy to maximum.
3. See how to apply model in production using weights and pickle files or anything, work on cloud(Collab).



Done with all the data augmentation work, made a directory with all the 5000 images, i.e. 500 girls with 10 images each, also loaded them to notebook in correct format, made the labels corresponding to them, i.e. 10 replica of each attractiveness label, in a sequence. 

Now starting with model, not applying model directly on the augmented images, instead first trying the models on original 500 images, cropping them into 224X224, using GPU is much easy on 500 images as compared to 5000 images.

Using transfer learning for the purpose, picking up pre trained models directly without any changes in the convolution structure, only adding a dense layer with one unit at the output layer, to solve the purpose of regression, else no hidden layer, using very simple network for first go.
Only 1001(i.e. the last layer) parameters are trainable as of now.

500 images are divided into training and testing , i.e. training is 400 and testing is 100, without any shuffle. Further training set is divided into 360 train and 40 validation images.

For all the models used, batch size = 20, and epochs = 25, learn rate = 0.01
Performance of different models :-
VGG16 = 50 %
VGG19 = 52 %
ResNet50 = 52.8 %
InceptionResNetV2 = 40 %
Inception = 43.6 %
Xception = 68 %


Using Augmented data now, for better training of the model, 500 images converted to 5000 images.
 After changing a few parameters, epochs = 35, batch size = 20, learning rate = 0.001
Xception model performing, 70 % on augmented data.

