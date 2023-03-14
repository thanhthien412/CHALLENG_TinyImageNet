import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from HDF5 import Write_HDF5
from imutils import paths
import numpy as np
import json
import cv2
import os
from tqdm import tqdm

#Label_Name
le=LabelEncoder()
#Train,Test
Path=list(paths.list_images(config.TRAIN_IMAGES))
Label=[p.split(os.path.sep)[-3] for p in Path]
Label=le.fit_transform(Label)
Train_Path,Test_Path,Train_Label,Test_Label=train_test_split(Path,Label,test_size=config.NUM_TEST_IMAGES,stratify=Label,random_state=42)

#Val
Text_file=open(config.VAL_MAPPINGS).read().strip().split('\n')
Text=[line.split('\t')[:2] for line in Text_file]
Val_Path=[os.path.join(config.VAL_IMAGES,ele[0]) for ele in Text]
Val_Label=le.transform([ele[1] for ele in Text])
dataset=[
    ('train',Train_Path,Train_Label,config.TRAIN_HDF5),
    ('test',Test_Path,Test_Label,config.TEST_HDF5),
    ('val',Val_Path,Val_Label,config.VAL_HDF5)
]

#processing data
(R,G,B)=([],[],[])

for (name,Paths,Labels,output) in dataset:
    print("[INFO]: LOADING {} set".format(name))
    database=Write_HDF5((len(Paths),64,64,3),'feature',output)
    for (i,(path,label)) in enumerate(zip(tqdm(Paths),Labels)):
        image=cv2.imread(path)
        
        if(name=="train"):
            (b,g,r)=cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        database.add([image],[label])
        
    database.close()
    
print("[INFO]: Means of each channel...")
D={'R':np.mean(R),'G':np.mean(G),'B':np.mean(B)}
f=open(config.DATASET_MEAN,"w")
f.write(json.dumps(D))
f.close
