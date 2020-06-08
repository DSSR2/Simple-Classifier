from classification import classifier
'''
Trainng Example: 
Data stored in ./Data
Model to be used is base model
Number of classes is 3
Save new models in ./Models/
'''
classify = classifier("C0", 2, "Data/", "Models/")

# Add a few extra augmentations
classify.data_loader(shear_range=0.2, rotation_range=90, horizontal_flip=True)

# Let's train for 10 epochs
classify.train(10)

'''
Load Trained Model
'''
classify = classifier("./Models/", ["Nok", "Ok"])

'''
Evaluation Example
'''
# During evaluation, the folder structure is the same as that of the train folder. 
classify.evaluate("./Data/Test/")

'''
Testing Example
'''
# The same function can be used for running tests on single images as well as folders.
# Test folder
classify.predict("./Data/Test/")

# Test single image
classify.predict("./Data/Test/1.jpg")

