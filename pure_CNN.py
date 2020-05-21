import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage import io
from tensorflow.keras import backend
from tensorflow.keras.applications.vgg19 import VGG19

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
    for index in range(4):
        model = res_block_gen(model, 3, 64, 1)
    up=int(upscale/2)
    for index in range(up):
        model = up_sampling_block(model, 3, 64, 1)
    model=keras.layers.Conv2D(64,3, strides=1, padding='same',use_bias=False)(model)
    model = keras.layers.PReLU()(model)

    #model=z   
       
            # Using 5 Residual Blocks
    for index in range(16):
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

    batch_count=int(100/batch_size)
    skip_frame=int((patch_size-1)/2)
    for i in range (epochs):       
        for j in range (group_size):
            group_number=zeropad(j,3)
            LR_files=load_data('./drive/My Drive/Summer_SR/LR/X4/%s'%group_number,'.png')
            HR_files=load_data('./drive/My Drive/Summer_SR/HR/%s'%group_number,'.png')     
            print(group_number)
            # 加两行 前后分别
            front_LR=np.zeros((skip_frame,np.shape(LR_files)[1],np.shape(LR_files)[2],np.shape(LR_files)[3]))
            back_LR=np.zeros((skip_frame,np.shape(LR_files)[1],np.shape(LR_files)[2],np.shape(LR_files)[3]))
            for p in range (skip_frame):
              front_LR[p]=LR_files[0]
              back_LR[0]=LR_files[-1]
            LR_files=np.concatenate((front_LR,LR_files),axis = 0)
            LR_files=np.concatenate((LR_files,back_LR),axis = 0)

            for k in range (batch_count):
              HR_frame=HR_files[k*batch_size:(k+1)*batch_size]
              LR_frame=[]
              for m in range (batch_size):
                frame=LR_files[k*batch_size+skip_frame]
                for n in range (skip_frame):
                  front=LR_files[k*batch_size+skip_frame-(n+1)]
                  back=LR_files[k*batch_size+skip_frame+(n+1)]
                  frame=np.concatenate((front,frame),axis = 2)
                  frame=np.concatenate((frame,back),axis = 2)
                frame=np.reshape(frame,[1,np.shape(frame)[0],np.shape(frame)[1],np.shape(frame)[2]])
                if m==0:
                  LR_frame=frame
                else:
                  LR_frame=np.concatenate((LR_frame,frame),axis = 0)  
              # processing
              loss=generator_model.train_on_batch(LR_frame,HR_frame)
              print(m)
            print ('-'*15, 'group %d' % j, '-'*15)
            print('group:',j,'loss:',loss)   
        print ('-'*15, 'Epoch %d' % i, '-'*15)
        print('epoch:',i,'loss:',loss)

        show = generator_model.predict(LR_frame) 
        plt.imshow(show[0,:,:,:])      
        plt.savefig("'./drive/My Drive/Summer_SR/epoch_%d.jpg"%(i))

train(10,2,180,320,4,generator_optimizer,5,30) 