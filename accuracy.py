from keras.models import load_model
import numpy as np
import argparse
import config
import json
from preprocessdata import Meansubtraction
from datagenerator import Generator
def accuracy(preds,labels):
    rank1=0
    rank5=0
    for (p,gt) in zip(preds,labels):
        p=np.argsort(p)[::-1]
        if(gt==p[0]):
            rank1+=1
        if(gt in p[:5]):
            rank5+=1
            
    rank1 /= float(len(preds))
    rank5 /= float(len(preds))
    
    return (rank1,rank5)


ap = argparse.ArgumentParser()

ap.add_argument("-m", "--models", type=int, default=1,
    help="Number of model want to use")

args = vars(ap.parse_args())

epoches=[]
while len(epoches) < args['models']:
    epo=int(input('Related epches: '))
    epoches.append(epo)

means = json.loads(open(config.DATASET_MEAN).read())

meanprocess=Meansubtraction(means['R'],means['G'],means['B'])


testGen = Generator(config.TRAIN_HDF5,config.NUM_CLASSES, 64,[meanprocess])

print("[INFO]: LOADING MODELS")
models=[]
for num in epoches:
    models.append(load_model(config.MODEL_PATH.format(num)))

print("[INFO]: EVALUATING ACCURACY WITH ENSEMBLE METHOD")


predictions=[]
for model in models:
    predictions.append(model.predict_generator(testGen.generator(),
            steps=testGen.numImages // 64, max_queue_size=10))
    
 
predictions=np.average(predictions,0)
rank1,rank5=accuracy(predictions,testGen.db['label'])

testGen.close()

print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
