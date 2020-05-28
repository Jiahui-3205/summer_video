import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage import io
from tensorflow.keras import backend
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2

# from tensorflow.keras import layers

generator_3D_optimizer = tf.keras.optimizers.Adam(1e-3)
generator_2D_optimizer = tf.keras.optimizers.Adam(1e-3)
def zeropad(n,zeros=3):
    "Pad number n with zeros. Example: zeropad(7,3) == '007'"
    nstr = str(n)
    while len(nstr) < zeros:
        nstr = "0" + nstr
    return nstr

def res_block_gen_3D(model, kernal_size, filters, strides):
    
    gen = model
    
    model = keras.layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",data_format="channels_last")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = keras.layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",data_format="channels_last")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
        
    model = keras.layers.add([gen, model])   
    return model

def generator_3D(width, height, patch_size):
    

    input_pic=keras.layers.Input(shape=(patch_size,width, height,3),name='input_picture')
    model=keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters=32, strides=1,kernel_size=(3, 3)
                       , data_format='channels_last',padding='same', return_sequences=True),merge_mode='ave')(input_pic)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    model=keras.layers.UpSampling3D(size=(1,2,2))(model)
    for i in range (1):
        res_block_gen_3D(model, 3, 32, 1)

    model=keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters=32, strides=1,kernel_size=(3, 3)
                       , data_format='channels_last',padding='same', return_sequences=True),merge_mode='ave')(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    model=keras.layers.Conv3D(32,3,strides=(1, 1, 1),padding="same",data_format="channels_last")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)    
    model=keras.layers.UpSampling3D(size=(1,2,2))(model)
    model = keras.layers.LeakyReLU(alpha = 0.2)(model)
    
    for i in range (3):
        res_block_gen_3D(model, 3, 32, 1)
    
    model=keras.layers.Conv3D(32,3,strides=(1, 1, 1),padding="same",data_format="channels_last")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    
    # model=keras.layers.Conv3D(32,3,strides=(1, 1, 1),padding="same",data_format="channels_last")(model)
    # model = keras.layers.BatchNormalization(momentum = 0.5)(model)

    model=keras.layers.Conv3D(8,3,strides=(1, 1, 1),padding="same",data_format="channels_last")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    
    model=keras.layers.Conv3D(3,3,strides=(1, 1, 1),padding="same",data_format="channels_last")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)

    #model = keras.layers.Activation('tanh')(model)
	   
    generator_model = keras.Model(inputs = input_pic, outputs = model)
        
    return generator_model

def res_block_gen_2D(model, kernal_size, filters, strides):
    
    gen = model
    
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
        
    model = keras.layers.add([gen, model])   
    return model

def generator_2D(width, height,patch_size):      
    D2_input = keras.layers.Input(shape=((width), (height),3*patch_size),name='input_picture')
        
    model = keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(D2_input)
    model = keras.layers.LeakyReLU(alpha = 0.2)(model)
        
    for i in range (3):
        res_block_gen_2D(model, 3, 32, 1)

    model = keras.layers.Conv2D(filters = 12, kernel_size = 3, strides = 1, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)

    model = keras.layers.Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same")(model)
        
    generator_2D_model = keras.Model(inputs = D2_input, outputs = model)
       
    return generator_2D_model

def mean_squared_error(y_true, y_pred):
    return backend.mean(backend.mean(backend.mean(backend.square(y_pred - y_true), axis=-1)))
def vgg_loss(y_true, y_pred):
    image_shape = (720,1280,3)
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    vggloss=backend.mean(backend.square(loss_model(y_true) - loss_model(y_pred)))
    mseloss=mean_squared_error(y_true, y_pred)
    return mseloss+0.0001*vggloss
def resize_pic(group_num,patch_size):
    for i in range (group_num):
        group_number=zeropad(i,3) 
        for j in range (100):
            pic_number=zeropad(j,8)
            pic=cv2.imread("./LR/X4/%s/%s.png"%(group_number,pic_number),cv2.IMREAD_UNCHANGED)
            scale_percent = 400 # percent of original size
            width = int(pic.shape[1] * scale_percent / 100)
            height = int(pic.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(pic, dim, interpolation = cv2.INTER_CUBIC )
            pic_number_input=zeropad(j,2)
            for k in range (patch_size):
                cv2.imwrite("./patch_data/group%sframe%snum%d.jpg"%(group_number,pic_number_input,k),resized)

def train(epochs, batch_size,width, height,upscale,generator_3D_optimizer,generator_2D_optimizer,patch_size,group_size):
    generator_model_3D=generator_3D(width, height, patch_size)  
    generator_model_3D.compile(loss='mse', optimizer=generator_3D_optimizer,metrics=['mse'])

    generator_model_2D=generator_2D(width*upscale, height*upscale, patch_size)  
    generator_model_2D.compile(loss='mse', optimizer=generator_2D_optimizer,metrics=['mse'])

    batch_count=int(group_size*100/batch_size)
    skip_frame=int((patch_size-1)/2)
    for i in range (epochs):
        x = np.arange(group_size*100)
        np.random.shuffle(x)

        for j in range (batch_count):
            index=x[j*batch_size:(j+1)*batch_size]
            HR_frame=[]
            LR_frame=[]
            for k in range (batch_size):
                num=index[k]
                div = num // 100 
                mod = num % 100 
          # get pictures
                group_number=zeropad(div,3) 
                pic_number=zeropad(mod,8)
          
                HR_data = cv2.imread('./HR/%s/%s.png'%(group_number,pic_number))/255
                HR_data = np.reshape(HR_data,[1,np.shape(HR_data)[0],np.shape(HR_data)[1],np.shape(HR_data)[2]])
                LR_data = cv2.imread('./LR/X4/%s/%s.png'%(group_number,pic_number))/255
                LR_data = np.reshape(LR_data,[1,np.shape(LR_data)[0],np.shape(LR_data)[1],np.shape(LR_data)[2]])

                for m in range (skip_frame):
                    if mod-(m+1)<0:
                        front_LR = cv2.imread('./LR/X4/%s/%s.png'%(group_number,'00000000'))/255
                        front_HR = cv2.imread('./HR/%s/%s.png'%(group_number,'00000000'))/255
                        front_LR = np.reshape(front_LR,[1,np.shape(front_LR)[0],np.shape(front_LR)[1],np.shape(front_LR)[2]])
                        front_HR = np.reshape(front_HR,[1,np.shape(front_HR)[0],np.shape(front_HR)[1],np.shape(front_HR)[2]])
                        back_num=zeropad((mod+(m+1)),8)
                        back_LR = cv2.imread('./LR/X4/%s/%s.png'%(group_number,back_num))/255
                        back_HR = cv2.imread('./HR/%s/%s.png'%(group_number,back_num))/255
                        back_LR = np.reshape(back_LR,[1,np.shape(back_LR)[0],np.shape(back_LR)[1],np.shape(back_LR)[2]]) 
                        back_HR = np.reshape(back_HR,[1,np.shape(back_HR)[0],np.shape(back_HR)[1],np.shape(back_HR)[2]])              
                    elif mod+(m+1)>99:
                        front_num=zeropad((mod-(m+1)),8)
                        front_LR = cv2.imread('./LR/X4/%s/%s.png'%(group_number,front_num))/255
                        front_LR = np.reshape(front_LR,[1,np.shape(front_LR)[0],np.shape(front_LR)[1],np.shape(front_LR)[2]]) 
                        front_HR = cv2.imread('./HR/%s/%s.png'%(group_number,front_num))/255 
                        front_HR = np.reshape(front_HR,[1,np.shape(front_HR)[0],np.shape(front_HR)[1],np.shape(front_HR)[2]])                          

                        back_LR = cv2.imread('./LR/X4/%s/%s.png'%(group_number,'00000099'))/255
                        back_HR = cv2.imread('./HR/%s/%s.png'%(group_number,'00000099'))/255              
                        back_LR = np.reshape(back_LR,[1,np.shape(back_LR)[0],np.shape(back_LR)[1],np.shape(back_LR)[2]]) 
                        back_HR = np.reshape(back_HR,[1,np.shape(back_HR)[0],np.shape(back_HR)[1],np.shape(back_HR)[2]])

                    else:
                        front_num=zeropad((mod-(m+1)),8)
                        front_LR = cv2.imread('./LR/X4/%s/%s.png'%(group_number,front_num))/255
                        front_LR = np.reshape(front_LR,[1,np.shape(front_LR)[0],np.shape(front_LR)[1],np.shape(front_LR)[2]]) 
                        front_HR = cv2.imread('./HR/%s/%s.png'%(group_number,front_num))/255 
                        front_HR = np.reshape(front_HR,[1,np.shape(front_HR)[0],np.shape(front_HR)[1],np.shape(front_HR)[2]])               

                        back_num=zeropad((mod+(m+1)),8)
                        back_LR = cv2.imread('./LR/X4/%s/%s.png'%(group_number,back_num))/255           
                        back_LR = np.reshape(back_LR,[1,np.shape(back_LR)[0],np.shape(back_LR)[1],np.shape(back_LR)[2]])
                        back_HR = cv2.imread('./HR/%s/%s.png'%(group_number,back_num))/255              
                        back_HR = np.reshape(back_HR,[1,np.shape(back_HR)[0],np.shape(back_HR)[1],np.shape(back_HR)[2]])              
         
                    LR_data = np.concatenate((front_LR,LR_data),axis = 0)
                    LR_data = np.concatenate((LR_data,back_LR),axis = 0)
                    HR_data = np.concatenate((front_HR,HR_data),axis = 0)
                    HR_data = np.concatenate((HR_data,back_HR),axis = 0)            

                if k == 0:
                    HR_data=np.reshape(HR_data,[1,np.shape(HR_data)[0],np.shape(HR_data)[1],np.shape(HR_data)[2],np.shape(HR_data)[3]])
                    LR_data=np.reshape(LR_data,[1,np.shape(LR_data)[0],np.shape(LR_data)[1],np.shape(LR_data)[2],np.shape(LR_data)[3]])
                    HR_frame=HR_data
                    LR_frame=LR_data
                else:
                    HR_data=np.reshape(HR_data,[1,np.shape(HR_data)[0],np.shape(HR_data)[1],np.shape(HR_data)[2],np.shape(HR_data)[3]])
                    LR_data=np.reshape(LR_data,[1,np.shape(LR_data)[0],np.shape(LR_data)[1],np.shape(LR_data)[2],np.shape(LR_data)[3]])
                    HR_frame=np.concatenate((HR_frame,HR_data),axis = 0)
                    LR_frame=np.concatenate((LR_frame,LR_data),axis = 0)
            print('get_data_3D')
            loss_3D=generator_model_3D.train_on_batch(LR_frame,HR_frame)
            print('batch',j,'loss_3D',loss_3D)
            new_frame = generator_model_3D.predict(LR_frame)
            for u in range (batch_size):
                pic_index=index[u]
                div_new = pic_index // 100 
                mod_new = pic_index % 100

                new_clip=new_frame[u,:,:,:,:]
                group_num_new=zeropad(div_new,3) 
                frame_num_new=zeropad(mod_new,2)
                for v in range (patch_size):
                    new_frames=new_clip[v,:,:,:]
                    cv2.imwrite("./patch_data/group%sframe%snum%d.jpg"%(group_num_new,frame_num_new,v),new_frames)
            print('write successful')

            input_to_2D=[]
            HD_2D=[]  
            for w in range (batch_size):
                pic_index=index[w]
                div_new = pic_index // 100 
                mod_new = pic_index % 100
                group_num_new=zeropad(div_new,3) 
                frame_num_new=zeropad(mod_new,2)
                HD_frame_num=zeropad(mod_new,8)
                num_new=skip_frame        
                for z in range (skip_frame+1):
                    if z==0:
                        input_2D=cv2.imread("./patch_data/group%sframe%snum%d.jpg"%(group_num_new,frame_num_new,skip_frame))/255
                        HD_to_2D=cv2.imread("./HR/%s/%s.png"%(group_num_new,HD_frame_num))/255 
                        HD_to_2D=np.reshape(HD_to_2D,[1,np.shape(HD_to_2D)[0],np.shape(HD_to_2D)[1],np.shape(HD_to_2D)[2]])
                    else:
                        front_2D=cv2.imread("./patch_data/group%sframe%snum%d.jpg"%(group_num_new,frame_num_new,skip_frame-z))/255 
                        back_2D=cv2.imread("./patch_data/group%sframe%snum%d.jpg"%(group_num_new,frame_num_new,skip_frame+z))/255 
                        input_2D=np.concatenate((front_2D,input_2D),axis = 2) 
                        input_2D=np.concatenate((input_2D,back_2D),axis = 2) 
                input_2D=np.reshape(input_2D,[1,np.shape(input_2D)[0],np.shape(input_2D)[1],np.shape(input_2D)[2]])
                if w == 0:
                    input_to_2D=input_2D
                    HD_2D=HD_to_2D
                else:
                    input_to_2D=np.concatenate((input_to_2D,input_2D),axis = 0) 
                    HD_2D=np.concatenate((HD_2D,HD_to_2D),axis = 0)
            print('get_data_2D')
            loss_2D= generator_model_2D.train_on_batch(input_to_2D,HD_2D)
            print('batch',j,'loss_2D',loss_2D)
            
# 选在同一group的先后frame，然后reshape在第一axis合并，需要先生成[]，作为skip_frame=0是的情况

        print('-'*40)  
        print('epoch',i,'finished')
        print('-'*40)
        show = generator_model_2D.predict(input_to_2D)
        pic=(show[0,:,:,:])*255
        cv2.imwrite("./test_LSTM_pic/epoch%d.jpg"%(i),pic)               

resize_pic(30,3)
print('resize finish')
train(100,1,180,320,4,generator_3D_optimizer,generator_2D_optimizer,3,30) 