# The Over Simplified Classifier
## What is Classification?
Classification involves predicting which class an item belongs to. For example, classifying dogs and cats or defective and okay components. 
## When do we use Classifcation?
When we want to segregate objects based on complex features into two or more classes or groups, we use the classification module. This is a black box classifier and the system learns important features on it's own. 
For example: 
#### A cat vs dog classifier
![alt text](catVdog.jpg "Cat vs Dog")

#### A fruit classifier
![alt text](fruit.jpg "Cat vs Dog")
## Choice of models:
For simple images with simple features, use shallow models - C0, C1, C2. For images with complex features and patterns, use deeper models - C3, C4, C5. 
A shallow model will be faster during training as well as deployment. Best practice is to start with shallow models and move to deeper models only when accuracy is low. 

Image sizes corresponding to each of the models: 
* C0 - 224*224
* C1 - 240*240
* C2 - 260*260
* C3 - 300*300
* C4 - 380*380
* C5 - 456*456

## Preprocessing: 
To ensure higher accuracy, the system carries out basic preprocessing and image augmentation. If required, the user can take control and set the following parameters:
```python
shear_range - 0-1, Default = 0
zoom_range- 0-1, Default = 0
horizontal_flip - True/False, Default = True
vertical_flip - True/False, Default = True
rotation_range - 0-360 degrees, Default = 10
width_shift_range - 0-1, Default = 0
height_shift_range - 0-1, Default = 0
```
## Example:
Common examples are provided in [Example.py](./Example.py)

## Function Definitions:
_* required_
### 1. _classifier()_ constructor 
Takes 2 or more arguments. 
#### Creating and training a new model:
```python
classifier(Model_Name*, Number_of_Classes*, 
            Path_to_Data*, Model_Save_Path="./Models/") 
```
#### Loading an existing model: 
```python
classifier(Path_to_Model*, List_of_Class_Labels=[1,2,3...n])
```
### 2. _data_loader()_ function
This function attempts to automatically load as much data as possible. 
If the function finds separate Train and Validation folders, it will use them as is. 
If the function finds only a Train folder, the user can define the Validation split to ensure the model does not overfit. 
It also contains augmentation options. 
```python
data_loader(Path_to_Data*, batch_size=16,
        rescale_val=1./255, shear_range=0, zoom_range=0,
        rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None,
        horizontal_flip=True, vertical_flip=True, 
        validation_split=0.0)
```

### 3. _train()_ function
This function is used to train the model. For most cases, start with 10 epochs. Based on the loss and accuracy curves, decide on whether to increase or decrease number of epochs. 
```python
train(Number_of_Epochs*, plot=True)
```

### 4. _evaluate()_ function
This method is used to check the real world performance of the model. It uses data that is in the same structure as the Train data but has never been seen by the data. It is essentaially data that has already been labeled by the user but has never been shown to the model. This function provides a loss and accuracy score for this test set. 
_evaluate()_ will not dispaly individual sample outputs. It will only provide overall accuracy and loss. 
```python
evaluate(Path_to_Data*)
```

### 5. _predict()_ function 
This function can be used to test both a folder containing images as well as single images. In the case of folder testing, a _csv_ file is saved with the file name, the predicted class and the confidence score. 
#### Folder testing
```python
predict(Path_to_Folder*)
```
#### File testing
```python
predict(Path_to_File*)
```

## Folder structure to be followed:
Only the _Train_ folder is compulsary. _Validation_ and _Test_ folders are optional but recommended. 
Number and names of folders within _Train_ and _Validation_ must be the same.
``` 
Root
    |---- Train
        |---- Class One 
            |---- img1.jpg
            |---- img2.jpg
            |---- img3.jpg
            |---- ...
        |---- Class Two
            |---- img1.jpg
            |---- img2.jpg
            |---- img3.jpg
            |---- ...
        |---- Class Three
            |---- img1.jpg
            |---- img2.jpg
            |---- img3.jpg
            |---- ...
        |---- ....
    |---- Validation
        |---- Class One
            |---- img1.jpg
            |---- img2.jpg
            |---- img3.jpg
            |---- ...
        |---- Class Two
            |---- img1.jpg
            |---- img2.jpg
            |---- img3.jpg
            |---- ...
        |---- Class Three
            |---- img1.jpg
            |---- img2.jpg
            |---- img3.jpg
            |---- ...
        |---- ...
    |---- Test
        |---- img1.jpg
        |---- img2.jpg
        |---- img3.jpg
        |---- ...
```