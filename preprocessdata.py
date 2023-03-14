import numpy as np
import cv2 as cv
from sklearn.feature_extraction.image import extract_patches_2d
class Meansubtraction():
    def __init__(self,R,G,B):
        self.R=R
        self.G=G
        self.B=B
    
    def process(self,img):
        (B,G,R)=cv.split(img.astype('float32'))
        B-=self.B
        G-=self.G
        R-=self.R
        
        return cv.merge([B,G,R])
    
class PatchProcess():
    def __init__(self,width,height):
        self.width=width
        self.height=height
    
    def process(self,img):
        return extract_patches_2d(img,(self.height,self.width),1)[0]
    
class CropProcess():
    def __init__(self, width, height, horiz=True, inter=cv.INTER_AREA):
        self.width=width
        self.height=height
        self.horiz=horiz
        self.inter=inter
        
    def process(self,img):
        crops=[]
        (h,w)=img.shape[:2]
        coords = [
                [0, 0, self.width, self.height],
                [w - self.width, 0, w, self.height],
                [w - self.width, h - self.height, w, h],
                [0, h - self.height, self.width, h]]

        # compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])
        
        for (startX, startY, endX, endY) in coords:
            crop = img[startY:endY, startX:endX]
            crop = cv.resize(crop, (self.width, self.height),
            interpolation=self.inter)
            crops.append(crop)
        
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv.flip(c, 1) for c in crops]
            crops.extend(mirrors)

            # return the set of crops
        
        return np.array(crops)


class Resize():
    def __init__(self,width, height, inter=cv.INTER_AREA) -> None:
        self.width = width
        self.height = height
        self.inter = inter
        
    def process(self,img):
        (h,w)=img.shape[:2]
        dw=0
        dh=0
        
        if h<w:
            img=cv.resize(img,(self.height,w),interpolation=self.inter)
            dh=int((h-self.height)/2.0)
        else:
            img=cv.resize(img,(h,self.width),interpolation=self.inter)
            dh=int((w-self.width)/2.0)
            
        (h, w) = image.shape[:2]
        image = image[dh:h - dh, dw:w - dw]
        
        return cv.resize(image, (self.width, self.height),
                            interpolation=self.inter)           
        