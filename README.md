# SML_310_Project
Towards Developing a Robust Deep Learning Based Sign Language Recognition System




In this project, I propose a robust deep learning-based sign language recognition system on two public American Sign Language (ASL) datasets. I built my own deep 71 layers CNN model, Sign_languageNet, which is robust and able to classify statics signs from the ASL alphabet. In addition, for both datasets with this as the baseline model, the state-of-the-art CNN architectures were tested using pretrained weights on large dataset such as ImageNet through transfer learning to see how they affect the performance. Thus, such a system is important as being able to recognize fingerspelling-based hand gestures leads to whole words being recognized through combining signs, allowing for easier communication between ASL and non-ASL speakers. Several techniques were implemented to improve the performance of Sign_languageNet and transfer learning models such as hyperparameter tuning, regularization, data augmentation, and test-time augmentations. For the two public ASL datasets, it is proven that data augmentation and regularization not only helped to build a robust model, but also improved the per class performance of letter. Confusion matrices, accuracies, and F1-scores are used to evaluate and analyze the performance of the classification models and the per class performance of the letters. The results of the models trained on both datasets are promising, and they can be deployed for real time application which would bring a huge impact on ASL translation by facilitating the communication between hearing and hearing-impaired people.



The file has 4 jupyter notebook files named named as "Best_pretrained_model.ipynb", "SML_310_Project.ipynb", "SML_310_Project.ipynb"


"ENEE_436_Project_1", and it has several function to perform PCA, LDA, KNN, BAYES.

Make sure the data.mat is saved in the same folder as the notebook file

PCA:

. def Perform_PCA- is the function that performs PCA . def project_data_PCA- project the dataset using PCA . def recover_X_PCA-recovers the orginal data set to make sure that X is recovered

-------run the script to get the resiult of PCA and to check the recovered dataset

LDA:

. def Perform_LDA- is the function that performs LDA . def project_data_LDA-project the dataset using LDA . def recover_X_LDA- recovers the orginal data set to make sure that X is recovered

-------run the script to get the resiult of PCA and to check the recovered dataset

Dataset:

. def split_data - split the data into training and testing using 0.2 as the test size

KNN:

class KNN()- this define a K-nerest Neighbour class

. Then run the two scripts two show the minimum error rate and optimal K value for accuracy for both PCA and LDA . There is also a script that calclates the precision, recall, and the confusion matrices for both PCA and LDA data for my bayes implementation and sklearn's bayes

. It also calculates the predicted lables for the test data of both PCA and LDA data for my bayes implementation and sklearn's bayes

Bayes:

. class Bayes- This define a K-nerest Neighbour class

. def ML_Estimation- this method performs maximum likelihood estimation to find the mean and covariance

. run the script that performs a 5 fold cross validtion for both PCA data and LDA data. It calculates the predicted lables for the test data and calculates the maximum accuracy for the 5 fold cross validation for my bayes implementation and sklearn's bayes class
