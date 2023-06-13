import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import glob



model=load_model('BrainTumorModel.h5')
 
#image=cv2.imread('C:\\Users\\ISAH JEFF\\Desktop\\datasets\\datasets\\test\\no4.jpg')

#image=cv2.imread('C:\\Users\\ISAH JEFF\\Desktop\\datasets\\datasets\\test\\no (1).jpg')

image=cv2.imread('C:\\Users\\ISAH JEFF\\Desktop\\datasets\\datasets\\yes\\y42.jpg')

#image=cv2.imread('C:\\Users\\ISAH JEFF\\Desktop\\datasets\\datasets\\accuracy\\y2.jpg')




img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)


result=np.argmax(model.predict(input_img), axis=-1)

print(result)









