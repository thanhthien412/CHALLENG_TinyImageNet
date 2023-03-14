from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.layers import add
from keras.layers import ZeroPadding2D
from keras.regularizers import l2

class ResNet():
    @staticmethod
    def residual_module(x,K,stride,chanDim,reg=0.0005):
        shortcut=x
        
        bn1     = BatchNormalization(chanDim)(x)
        ac1     = Activation(activation='relu')(bn1)
        conv1   = Conv2D(int(K/4),(1,1),use_bias=False,kernel_regularizer=l2(reg))(ac1)
        
        bn2     = BatchNormalization(chanDim)(conv1)
        ac2     = Activation(activation='relu')(bn2)
        conv2   = Conv2D(int(K/4),(3,3),strides=stride,padding='same',use_bias=False,kernel_regularizer=l2(reg))(ac2)
        
        bn3     = BatchNormalization(chanDim)(conv2)
        ac3     = Activation(activation='relu')(bn3)
        conv3   = Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(ac3)
        
        if((stride[0]+stride[1])>2 or K!=shortcut.shape[-1]):
            shortcut = Conv2D(K,(1,1),strides=stride)(ac1)
        
        result  = add([conv3,shortcut])
        return  result
    
    @staticmethod
    def build(width,height,depth,classes,stages,filters,reg=0.0001,reduce=False):
        inputs=Input(shape=(height,width,depth))
        chanDim=-1
        
        x       = BatchNormalization(chanDim)(inputs)
        if  not reduce:
            x   = Conv2D(filters[0],(3,3),use_bias=False,padding='same',kernel_regularizer=l2(reg))(x)
        else:
            x   = Conv2D(filters[0],(5,5),use_bias=False,padding='same',kernel_regularizer=l2(reg))(x)
            x   = BatchNormalization(chanDim)(x)
            x   = Activation(activation='relu')(x)
            x   = ZeroPadding2D((1,1))(x)
            x   = MaxPooling2D((3,3),strides=(2,2))(x)
            
        for (idx,(sta,fil)) in enumerate(zip(stages,filters[1:])):
            
            stride = (1,1) if(idx==0) else (2,2)
            x   = ResNet.residual_module(x,fil,stride,chanDim,reg)
            
            for _ in range(sta-1):
                x = ResNet.residual_module(x,fil,(1,1),chanDim,reg)
                
        x       = BatchNormalization(chanDim)(x)
        x       = Activation('relu')(x)
        x       = AveragePooling2D((8,8))(x)
        x       = Flatten()(x)
        x       = Dense(classes,kernel_regularizer=l2(reg))(x)
        x       = Activation('softmax')(x)
        
        model   = Model(inputs,x,name='resnet')
        
        return model
            
class GoogleNetMini():
    @staticmethod
    def conv_module(x,D,H,W,stride,chanDim,reg=0.0005,padding='same'):
        x=Conv2D(D,(W,H),stride,padding,kernel_regularizer=l2(reg))(x)
        x=Activation('elu')(x)
        x=BatchNormalization(axis=chanDim)(x)    
        return x
    
    @staticmethod
    def inception_module(x,num1x1,num3x3reduce,num3x3,num5x5reduce,num5x5,num1x1pool,chanDim,reg=0.0005):
        first_block=GoogleNetMini.conv_module(x,num1x1,1,1,(1,1),chanDim,reg)
        
        second_block=GoogleNetMini.conv_module(x,num3x3reduce,1,1,(1,1),chanDim,reg)
        second_block=GoogleNetMini.conv_module(second_block,num3x3,3,3,(1,1),chanDim,reg)
        
        third_block=GoogleNetMini.conv_module(x,num5x5reduce,1,1,(1,1),chanDim,reg)
        third_block=GoogleNetMini.conv_module(third_block,num5x5,5,5,(1,1),chanDim,reg)
        
        fourth_block=MaxPooling2D((3,3),(1,1),padding='same')(x)
        fourth_block=GoogleNetMini.conv_module(fourth_block,num1x1pool,1,1,(1,1),chanDim,reg)
        
        return concatenate([first_block,second_block,third_block,fourth_block],chanDim)


    @staticmethod
    def build(width,height,depth,classes,reg=0.0005):
        inputs=Input(shape=(height,width,depth))
        chanDim=-1
        
        x=GoogleNetMini.conv_module(inputs,64,5,5,(1,1),chanDim,reg)
        
        x=MaxPooling2D((3,3),(2,2),padding='same')(x)
        
        x=GoogleNetMini.conv_module(x,64,1,1,(1,1),chanDim,reg)
        x=GoogleNetMini.conv_module(x,192,3,3,(1,1),chanDim,reg)
        
        x=MaxPooling2D((3,3),(2,2),padding='same')(x)
        
        x=GoogleNetMini.inception_module(x,64,96,128,16,32,32,chanDim,reg)
        
        x=GoogleNetMini.inception_module(x,128,128,192,32,96,64,chanDim,reg)
        
        x=MaxPooling2D((3,3),(2,2),padding='same')(x)
        
        x=GoogleNetMini.inception_module(x,192,96,208,16,48,64,chanDim,reg)
        
        x=GoogleNetMini.inception_module(x,160,112,224,24,64,64,chanDim,reg) 
        
        x=GoogleNetMini.inception_module(x,128,128,256,24,64,64,chanDim,reg)
        
        x=GoogleNetMini.inception_module(x,112,144,288,32,64,64,chanDim,reg)
        
        x=GoogleNetMini.inception_module(x,256,160,320,32,128,128,chanDim,reg)
        
        x=MaxPooling2D((3,3),(2,2),padding='same')(x)
        
        x=AveragePooling2D((4,4),(1,1))(x)
        
        x=Dropout(0.4)(x)
        
        x=Flatten()(x)
        
        x=Dense(classes,kernel_regularizer=l2(reg))(x)
        
        x=Activation('softmax')(x)
        
        model=Model(inputs,x,name='minigooglenet')
                                           
        return model                                             
                                                                            
                                                                            
                                                                        
                                                                   