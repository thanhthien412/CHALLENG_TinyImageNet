import os
import h5py
import numpy as np
import sys
import config
if sys.version_info[0] >= 3:
    unicode = str
    
    
class Write_HDF5():
    def __init__(self,dim,name,output,buffersize=1000):
        if(os.path.exists(output)):
            raise ValueError("The supplied `outputPath` already "
            "exists and cannot be overwritten. Manually delete "
            "the file before continuing.", output)
            
        self.db=h5py.File(output,'w')
        self.data=self.db.create_dataset(name,dim,dtype='float')
        self.label=self.db.create_dataset('label',(dim[0],),dtype='int')
        self.buffersize=buffersize
        self.buffer={'data':[],'label':[]}
        self.idx=0
        
    
    def add(self,data,label):
        self.buffer['data'].extend(data)
        self.buffer['label'].extend(label)
        
        if(len(self.buffer['data'])>=self.buffersize):
            self.merge()
            
    def merge(self):
        self.data[self.idx:self.idx+len(self.buffer['data'])]=self.buffer['data']
        self.label[self.idx:self.idx+len(self.buffer['data'])]=self.buffer['label']
        self.idx=self.idx+len(self.buffer['data'])
        self.buffer={'data':[],'label':[]}
        
    def determinelabel(self,classlabel):
        dt=h5py.special_dtype(vlen=unicode)
        labelset=self.db.create_dataset('classlabel',(len(classlabel),),dtype=dt)
        labelset[:]=classlabel
        
    def close(self):
        if(len(self.buffer['data'])>0):
            self.merge()
            
        self.db.close()