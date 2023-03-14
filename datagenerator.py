import numpy as np
import h5py
from keras.utils import np_utils
class Generator():
    def __init__(self,hdf5path,classes,batchsize=64,preprocess=None,aug=None,binarize=True):
        self.batchSize = batchsize
        self.preprocessors = preprocess
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(hdf5path)
        self.numImages = self.db["label"].shape[0]
        
        
    def generator(self):
        while True:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["feature"][i: i + self.batchSize]
                labels = self.db["label"][i: i + self.batchSize]
                
                if self.binarize:
                        labels = np_utils.to_categorical(labels,
                                self.classes)
                        
                if self.preprocessors is not None:
                    procImages = []
                    
                    for image in images:
                        for p in self.preprocessors:
                            image=p.process(image)
                        
                        procImages.append(image)
                    
                    images=np.array(procImages)
                    
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,labels, batch_size=self.batchSize))
                       
                
                yield(images,labels)
                    
                     
                     
    def close(self):
        self.db.close()