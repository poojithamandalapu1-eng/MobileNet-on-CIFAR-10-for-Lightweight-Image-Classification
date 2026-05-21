# MobileNet on CIFAR-10 for lightweight Image Classification
## GOAL
Compare MobileNet accuracy vs model size for edge devices using CIFAR-10 dataset
## Work done by three
Poojitha - Data Loading & Preprocessing 
Vishwa - Model Building & Trainig
Nitya - Evaluation and Comparision

## Project Structure
data_loader.py #Data loading, preprocessing, augmentation
model.py #MobileNet architecture
train.py #Training script (2-phase)
evaluate.py #Accuracy, size, inference time, plots
requirements.txt #Dependencies
README.md #Project Documentation

## How to Run?
### 1.Install dependencies
pip install -r requirements.txt
### 2.Train the model
python train.py
### 3.Evaluate the model
python evaluate.py
## Results
Model        Accuracy      Size      Inference
Simple CNN     ~70%        ~5MB        ~5 ms
MobileNet      ~85-90%     ~16MB       ~10 ms
## Dataset
CIFAR-10:https://www.cs.toronto.edu/~kriz/cifar.html