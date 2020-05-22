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

# from tensorflow.keras import layers
patch_size=7
generator_optimizer = tf.keras.optimizers.Adam(1e-3)
def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories

def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = io.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    return files        
                        
          
def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files


# files = load_data("./B100", ".jpg")

def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
        
    model = keras.layers.add([gen, model])   
    return model

def up_sampling_block(model, kernal_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.UpSampling2D(size = 2,interpolation="bilinear")(model)
    model = keras.layers.LeakyReLU(alpha = 0.2)(model)
    
    return model

def generator(width, height, upscale, patch_size):
    

    input_pic=keras.layers.Input(shape=(width, height,patch_size*3),name='input_picture')
    model=keras.layers.Conv2D(32,3, strides=1, padding='same',use_bias=False)(input_pic)
    model=keras.layers.Conv2D(64,3, strides=1, padding='same',use_bias=False)(input_pic)
    for index in range(1):
        model = res_block_gen(model, 3, 64, 1)
    up=int(upscale/2)
    for index in range(up):
        model = up_sampling_block(model, 3, 64, 1)
    model=keras.layers.Conv2D(64,3, strides=1, padding='same',use_bias=False)(model)
    model = keras.layers.PReLU()(model)

    #model=z   
       
            # Using 5 Residual Blocks
    for index in range(5):
        model = res_block_gen(model, 3, 64, 1)
	    
    model = keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    #model = keras.layers.add([z, model])

            
    model = keras.layers.Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model) #三通道
    model = keras.layers.Activation('tanh')(model)
	   
    generator_model = keras.Model(inputs = input_pic, outputs = model)
        
    return generator_model
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
def zeropad(n,zeros=3):
    "Pad number n with zeros. Example: zeropad(7,3) == '007'"
    nstr = str(n)
    while len(nstr) < zeros:
        nstr = "0" + nstr
    return nstr

def train(epochs, batch_size,width, height,upscale,generator_optimizer,patch_size,group_size):
    generator_model=generator(width, height, upscale, patch_size)  
    generator_model.compile(loss='mse', optimizer=generator_optimizer,metrics=['mse'])

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
          
          HR_data = mpimg.imread('./drive/My Drive/Summer_SR/HR/%s/%s.png'%(group_number,pic_number))
          LR_data = mpimg.imread('./drive/My Drive/Summer_SR/LR/X4/%s/%s.png'%(group_number,pic_number))
          for m in range (skip_frame):
            if mod-(m+1)<0:
              front = mpimg.imread('./drive/My Drive/Summer_SR/LR/X4/%s/%s.png'%(group_number,'00000000'))
              back_num=zeropad((mod+(m+1)),8)
              back = mpimg.imread('./drive/My Drive/Summer_SR/LR/X4/%s/%s.png'%(group_number,back_num))
            elif mod+(m+1)>99:
              front_num=zeropad((mod-(m+1)),8)
              front = mpimg.imread('./drive/My Drive/Summer_SR/LR/X4/%s/%s.png'%(group_number,front_num))
              back = mpimg.imread('./drive/My Drive/Summer_SR/LR/X4/%s/%s.png'%(group_number,'00000099'))
            else:
              front_num=zeropad((mod-(m+1)),8)
              front = mpimg.imread('./drive/My Drive/Summer_SR/LR/X4/%s/%s.png'%(group_number,front_num))
              back_num=zeropad((mod+(m+1)),8)
              back = mpimg.imread('./drive/My Drive/Summer_SR/LR/X4/%s/%s.png'%(group_number,back_num))
            
            LR_data = np.concatenate((front,LR_data),axis = 2)
            LR_data = np.concatenate((LR_data,back),axis = 2) 
          if k == 0:
            HR_data=np.reshape(HR_data,[1,np.shape(HR_data)[0],np.shape(HR_data)[1],np.shape(HR_data)[2]])
            LR_data=np.reshape(LR_data,[1,np.shape(LR_data)[0],np.shape(LR_data)[1],np.shape(LR_data)[2]])
            HR_frame=HR_data
            LR_frame=LR_data
          else:
            HR_data=np.reshape(HR_data,[1,np.shape(HR_data)[0],np.shape(HR_data)[1],np.shape(HR_data)[2]])
            LR_data=np.reshape(LR_data,[1,np.shape(LR_data)[0],np.shape(LR_data)[1],np.shape(LR_data)[2]])
            HR_frame=np.concatenate((HR_frame,HR_data),axis = 0)
            LR_frame=np.concatenate((LR_frame,LR_data),axis = 0)
        print('data get')
        loss=generator_model.train_on_batch(LR_frame,HR_frame)
        print('batch',j,'loss',loss)

      print('epoch',i,'loss',loss)
      print('-'*40)
      show = generator_model.predict(LR_frame)
      plt.imshow(show[0,:,:,:]) 
      plt.savefig("'./drive/My Drive/Summer_SR/test_pic/epoch_%d.jpg"%(i)) 
      


train(20,1,180,320,4,generator_optimizer,7,30) 
