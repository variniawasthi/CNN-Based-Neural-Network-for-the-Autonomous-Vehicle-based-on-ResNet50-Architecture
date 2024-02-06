# CNN-Based-Neural-Network-for-the-Autonomous-Vehicle
Abstract: An in-depth assessment of the emergence of autonomous vehicle technology, following its trajectory from traditional rule-based techniques, highlighted by ALVINN, to the revolutionary potential of Deep Neural Networks (DNNs) and Convolutional Neural Networks (CNNs). The study also presents a CNN-based optimized model, illustrating the effect of DNN and CNN on perception, decision-making, and control in autonomous driving. Furthermore, it assesses the challenges faced in their implementation, solutions devised to address them, and outlines the potential future directions and advancements in the realm of autonomous vehicles. This comprehensive analysis aims to provide a profound understanding of the paradigm shift in autonomous driving technology and its significance in shaping the future of transportation.

END-TO-END DEEP LEARNING ALGORITHM: The recent implementation of Pomerleau's original idea by Nvidia, using deep neural networks, represents a paradigm shift in autonomous driving algorithms. The adoption of deep learning and end-to-end learning approaches allowed NVIDIA to train a self-driving model on an unprecedented scale. While Pomerleau initially trained on approximately 5 minutes of driving data, NVIDIA leveraged an extensive dataset of 3000 hours. This corresponds to a staggering 36,000-fold increase in data, showcasing the exponential growth in available training samples.

PROPOSED MODEL: Following is one of the outcomes of the study that illustrates a proposition, a leading-edge Convolutional Neural Network (CNN) that is developed specifically for autonomous driving. This CNN delivers increased perception, decision-making, and oversight for better and more effective autonomous vehicles by employing novel computer vision technologies.

![image](https://github.com/swapnilgupta14/CNN-Based-Neural-Network-for-the-Autonomous-Vehicle-based-on-ResNet50-Architecture/assets/85231522/c876537e-fe7c-4161-9e38-f449d7b2716f)


DATASET: The dataset utilized in the proposed model is collected through the Udacity driving simulator. This simulator provides a virtual environment for driving scenarios, offering data from three front-facing cameras mounted on a car. Alongside camera images, the dataset also includes essential driving statistics such as throttle, speed, and steering angle. These camera images serve as the primary input data for training the model. The Table 1. showcases a portion of dataset used in proposition.

![Untitled-1](https://github.com/swapnilgupta14/CNN-Based-Neural-Network-for-the-Autonomous-Vehicle-based-on-ResNet50-Architecture/assets/85231522/27499f45-ee57-41ea-9c43-f66611794b76)


Table 1. A structure of the driving dataset, encompassing image file paths from the center, left, and right cameras, along with associated steering angles, throttle, reverse status, and speed measurements.
![image](https://github.com/swapnilgupta14/CNN-Based-Neural-Network-for-the-Autonomous-Vehicle-based-on-ResNet50-Architecture/assets/85231522/e5c25985-4b68-4fc1-9225-8ad0b53ea8aa)


DATA PREPROCESSING. Prior to inputting the images into the neural network, a series of essential preprocessing steps are undertaken. This process significantly improved the quality of the input data and enhanced the learning process, such as Cropping, Colour Space Conversion, Gaussian Blur, Pixel Value Normalization. The Fig. 5. display the processed image converted from input road image.

![Untitled](https://github.com/swapnilgupta14/CNN-Based-Neural-Network-for-the-Autonomous-Vehicle-based-on-ResNet50-Architecture/assets/85231522/e17b7f05-2bfb-4e3a-b6a4-c2aac57350a2)


ARCHITECTURE: The proposed architecture is inspired from the NVIDIA's implementation using CNNs. The ResNet50 architecture utilising pre-trained weights as the foundation for feature extraction. The ResNet50 architecture is known for its depth and performance in image-related tasks. The majority of the layers in the ResNet50 base model are made frozen (non-trainable), except for the last four layers, which were fine-tuned to adapt the model to the specific task. On top of the base model, several layers are added, including dropout layers for regularization and fully connected dense layers for prediction. Building upon the ResNet50 base, a set of additional layers is introduced to create the proposed architecture. These additional layers include dropout layers and fully connected dense layers. The inclusion of dropout layers serves as a regularization technique, helping to prevent overfitting during training. The fully connected dense layers are responsible for making the final prediction of the steering angle.

TRAINING PROCEDURE: The training of proposed model involves, Dataset Splitting, the dataset is divided into training and validation sets (80% for training, 20% for validation). Loss Function, Mean Squared Error (MSE) is used as the loss function, which measures the discrepancy between predicted and ground truth steering angles. Optimization, The Adam optimizer is employed to minimize the loss, Parameters: the proposed model epoch and batch size is 25 and 128 respectively. Data Augmentation, Data augmentation techniques, such as horizontal flipping, are applied to increase the dataset's diversity. The Fig.6. illustrate the distribution of steering angles in the divided dataset. 

![Uploading Untitled.png…]()


MODEL EVALUATION:

•	Loss Values. The final training loss was 0.3048, and the validation loss was 0.2398. These values represent the MSE between predicted and actual steering angles. The Fig. 7. display the distribution of the Loss Function over Epochs performed. 
 
![image](https://github.com/swapnilgupta14/CNN-Based-Neural-Network-for-the-Autonomous-Vehicle-based-on-ResNet50-Architecture/assets/85231522/3f4e611f-fbc1-4d48-a121-f20f29a842b6)


•	Accuracy Values. Accuracy while training went up from 0.4159 to 0.5736 with time. Validation accuracy started off at 0.5698 and was the same thereafter.
•	Performance Evaluation. The model demonstrated an ability to learn and make steering angle predictions. The decreasing loss values indicate that it learned from the data. However, the constant validation accuracy suggests that the model may not significantly improve its performance on the validation data

![image](https://github.com/swapnilgupta14/CNN-Based-Neural-Network-for-the-Autonomous-Vehicle-based-on-ResNet50-Architecture/assets/85231522/25deefa0-34fa-4c1a-ba75-7213305476b5)

