import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import imutils
import time
import cv2
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn
import pandas as pd
from glob import glob
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

'''
Allow GPU Growth
'''
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

class classifier():
    def __init__(self, *args, **kwargs):
        if(len(args) == 1 or len(args)==2):
            oneArg = True
        else:
            oneArg = False
        
        if(oneArg):
            try:
                self.load(args[0], args[1])
                self.labels = args[1]
                shape = self.model.layers[0].get_output_at(0).get_shape().as_list()
                self.img_size = (shape[1], shape[2])
            except:
                print("No model named ", self.model_name, ". Default C0 model loaded.")
                self.model_name = "C0"
                self.img_size = (224,224)
        else:
            self.init_from_args(*args)


    def init_from_args(self, *args):
        self.model_name = args[0]
        self.num_classes = args[1]
        self.data_path = args[2]
        try:
            self.save_path = args[3]
        except:
            print("No save path given. Default chosen as ./models/")
            self.save_path = os.getcwd()+"/models/"
        try:
            self.validation_split = args[4]
        except:
            print("No validation split provided, assuming folder is explicitly provided.")
            self.validation_split = 0.0

        self.labels = os.listdir(self.data_path+"/Train/")

        if(self.model_name=="C0"):
            self.img_size=(224,224)
        elif(self.model_name=="C1"):
            self.img_size=(240,240)
        elif(self.model_name=="C2"):
            self.img_size=(260,260)
        elif(self.model_name=="C3"):
            self.img_size=(300,300)
        elif(self.model_name=="C4"):
            self.img_size=(380,380)
        elif(self.model_name=="C5"):
            self.img_size=(456,456)           
        
        self.data_loader()
        self.model_chooser(classes=self.num_classes)
        
    def data_loader(self, root=os.getcwd(), batch_size=32,
        rescale_val=1./255, shear_range=0, zoom_range=0,
        rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, 
        brightness_range=None, fill_mode='nearest', 
        horizontal_flip=False, vertical_flip=False, dtype=None):
        self.batch_size = batch_size
        print("Data loader started.")
        root = self.data_path
        validation_split = self.validation_split
        folders = os.listdir(root)
        train_datagen = ImageDataGenerator(
                rescale=rescale_val,
                shear_range=shear_range,
                zoom_range=zoom_range,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip, 
                rotation_range=rotation_range, 
                width_shift_range=width_shift_range, 
                height_shift_range=height_shift_range,
                fill_mode=fill_mode,
                validation_split=validation_split)

        if("Validation" in folders):
            valid_datagen = ImageDataGenerator(rescale=rescale_val)

            train_generator = train_datagen.flow_from_directory(root+"/Train/", target_size=self.img_size, batch_size=batch_size)
            val_generator = valid_datagen.flow_from_directory(root+"/Validation", target_size=self.img_size, batch_size=batch_size)

        else:
            train_generator = train_datagen.flow_from_directory(root+"/Train/", subset="training",  target_size=self.img_size, batch_size=batch_size)
            val_generator = train_datagen.flow_from_directory(root+"/Train/", subset="validation", target_size=self.img_size, batch_size=batch_size)

        self.train_data_gen = train_generator
        self.valid_data_gen = val_generator

    def model_chooser(self, classes=2, weights=None):
        print("Model selection started.")
        name = self.model_name
        if(name=='C0'):
            self.model = efn.EfficientNetB0(include_top=True, weights=weights, classes=classes)
        elif(name=='C1'):
            self.model = efn.EfficientNetB1(include_top=True, weights=weights, classes=classes)
        elif(name=='C2'):
            self.model = efn.EfficientNetB2(include_top=True, weights=weights, classes=classes)
        elif(name=='C3'):
            self.model = efn.EfficientNetB3(include_top=True, weights=weights, classes=classes)
        elif(name=='C4'):
            self.model = efn.EfficientNetB4(include_top=True, weights=weights, classes=classes)
        elif(name=='C5'):
            self.model = efn.EfficientNetB5(include_top=True, weights=weights, classes=classes)
        
        if(classes==2):
            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ['acc'])
        elif(classes>2):
            self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ['acc'])      

    def plot(self, hist):
        # summarize history for accuracy
        plt.ion()
        plt.show()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.save_path+"/Accuracy Curve.png")
        # summarize history for loss
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.draw()
        plt.savefig(self.save_path+"/Loss Curve.png")
        plt.pause(0.001)
        

    def train(self, epochs, plot=False):
    
        print("Begin training. ", epochs, " epochs.")
        self.hist = self.model.fit(self.train_data_gen, validation_data=(self.valid_data_gen), batch_size=self.batch_size, steps_per_epoch=len(self.train_data_gen), validation_steps=len(self.valid_data_gen), epochs=epochs, verbose=1)
        if(plot):
            self.plot(self.hist)
        self.save()
    
    def save(self):
        if not os.path.exists(self.save_path):
            print("Folder does not exist. Creating...")
            os.makedirs(self.save_path)
        
        print("Saving Model to ", self.save_path)
        self.model.save(self.save_path) 
        print("Save successful!")

    def load(self, path, labels=[]):
        print("Loading model from ", path)
        self.labels = labels
        self.model = tf.keras.models.load_model(path)
        shape = self.model.layers[-1].get_output_at(0).get_shape().as_list()
        if(len(self.labels) == 0):
            self.labels = [i for i in range(1, shape[1]+1)]
        print("Load successful!")

    def evaluate(self, path, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(path, target_size=self.img_size, batch_size=batch_size)
        eval = self.model.evaluate(test_generator, steps=len(test_generator))
        print("Test loss: ", eval[0])
        print("Test Accuracy: ", eval[1]*100, "%")

        Y_pred = self.model.predict(test_generator, steps=len(test_generator))
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        self.confusion_matrix = confusion_matrix(test_generator.classes, y_pred)
        print(self.confusion_matrix)
        print('Classification Report')
        target_names = self.labels
        self.report = classification_report(test_generator.classes, y_pred, target_names=target_names)
        print(self.report)

    def predict(self, path):
        sing_f = False
        if(os.path.isdir(path)):
            files = glob(path+"/*.png")+glob(path+"/*.jpg")+glob(path+"/*.bmp")
        else:
            files = [path]
            sing_f = True
        op = []
        for i in tqdm(files): 
            fname = i.split("\\")[-1]
            img = cv2.imread(i)
            img = cv2.resize(img, self.img_size)
            img = img/255
            img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
            pred = self.model.predict(img)
            ans = [fname, self.labels[np.argmax(pred)], np.amax(pred)]
            if(sing_f):
                print("File name: ", ans[0],"\nPrediction: ", ans[1], "\nConfidence: ", ans[2])
            op.append(ans)
        df = pd.DataFrame(op, columns=["File Name", "Label", "Confidence"])
        df.to_csv("results.csv", index=False)

            
