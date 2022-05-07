# SML_310_Project
Towards Developing a Robust Deep Learning Based Sign Language Recognition System




In this project, I propose a robust deep learning-based sign language recognition system on two public American Sign Language (ASL) datasets. I built my own deep 71 layers CNN model, Sign_languageNet, which is robust and able to classify statics signs from the ASL alphabet. In addition, for both datasets with this as the baseline model, the state-of-the-art CNN architectures were tested using pretrained weights on large dataset such as ImageNet through transfer learning to see how they affect the performance. Thus, such a system is important as being able to recognize fingerspelling-based hand gestures leads to whole words being recognized through combining signs, allowing for easier communication between ASL and non-ASL speakers. Several techniques were implemented to improve the performance of Sign_languageNet and transfer learning models such as hyperparameter tuning, regularization, data augmentation, and test-time augmentations. 

#### Direction to set up the system for an experiment

The folder titled "Sign_Language" has two folders - Dataset_I, and Dataset_II. Dataset_I has small images in grayscale format saved in CSV format for both training and testing. In addition, it has 24 classe from A to Z(excluding J and Z). However, Dataset II is more complex, and has colored and high resolution images saved as jpg format. I created custom folders for training and testing as I used subset of the data. In addition, it has 27 classes with the addition of SPACE, DELETE, NOTHING.

In addition, The folder titled "Sign_Language" has 4 jupyter notebook files named named as "Best_pretrained_model.ipynb", "SML_310_Project_1.ipynb", "SML_310_Project_3.ipynb", and "SML_310_Project_4.ipynb".


"SML_310_Project_1.ipynb" is the jupyter notebook that has the code for dataset I. This notebook has Sign_LanguageNet. In addition, it has the transfer learning methods tested on Dataset I.


"Best_pretrained_model.ipynb"- This has the code for dataset II. This code is used to get the best pretrained model among the seven known pretrained models such as MobileNet, DenseNet, ResNet, EfficientNet, InceptionNet, XceptionNet, and VGG.

"SML_310_Project_3.ipynb"- This has the complete code that trains and evaluates Dataset II using MobileNet, which has 27 classes.

 "SML_310_Project_4.ipynb"- This implements/tests Sign_LanguageNet on Dataset II.

Make sure to put the datasets in a directory that you can acess, so that training the CNNs will be easier.
