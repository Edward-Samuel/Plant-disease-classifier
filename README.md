# Plant_disease_classifier
Classify plant diseases with a custom CNN (ResNet) model built from scratch. A lightweight and efficient solution for detectingplant diseases using images

Day 1 - initial model build
![image](https://github.com/user-attachments/assets/9307abe4-7a72-4de8-964f-b562f90a4fc3)
Trained on 200 images on each disease images and normal image

![image](https://github.com/user-attachments/assets/1560805e-4778-4358-a224-eb815a4c8baf)
Test accuracy between 55% - 65% on average

Day 2 - changes made to optimizers and parameters to reduce model overfit

Day 3 - Tried to validate the model with images from internet ,but able to fifnd limited images only
  Split the datset into (75/25)csv format for easier access , previously(80/20)
  Testing accuracy on unseen data(given) reached a average of 90%
  
  ![image](https://github.com/user-attachments/assets/3c9c214e-9d32-4f7f-a4bf-b776329601ce)

  ![image](https://github.com/user-attachments/assets/1f16d1b2-a2c9-4b18-9b85-f91d43077fb4)

 
Day 4 - https://github.com/B3-SOFTWARE/Paddy-Disease-Classifier

Day 5 - Tested the old ResNet CNN model and pretrained VisionTransformer based deit model on  new unseen test images 3000+ taken from kaggle "https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image"


CNN model performed well with an accuracy around 78% on testing data
![image](https://github.com/user-attachments/assets/343cdef0-76b5-49ea-800f-fcab4030e31b)

new Vi-Transformer model reached training accuracy 97% and 60% on testing data
![image](https://github.com/user-attachments/assets/b2b585dd-a2bd-4c0c-8d71-f2b7e8d6c011)
![image](https://github.com/user-attachments/assets/e23eafb5-d1a0-4f5e-9804-1d29b92eaa46)

### Day 6 - Optimization and Results
  Focused on optimizing the model through adjustments to learning rates, adaptive optimizers (e.g., AdamW), and enhanced data augmentation (rotations, flips, zooms). Regularization techniques like dropout and L2 were applied to reduce overfitting. The ViT model consistently achieved **95%-96% testing accuracy**, showing improved robustness and stability across diverse datasets. This phase solidified the model's reliability for plant disease detection.
  
![image](https://github.com/user-attachments/assets/65feaa04-115c-44f8-bcb2-608fc17c74b7)
![image](https://github.com/user-attachments/assets/dd50cbff-4e88-4561-80ca-f886ebd8d2f3)
