# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:41:02 2020

@author: junaid
"""
import numpy as np
import glob
import os 
import math
import cv2
from tqdm import tqdm
import json


def Frame_Extractor(v_file, path='./', ext='.avi', frames_dir='train_1', extract_rate='all', frames_ext='.jpg'):
    """
    A method which extracts the frames from the guven video. It can ex
    Parameters
    ----------
    
    v_file : str
        Name of the video file, without extension.
    
    path : str
        Path to the video file, if the video is in the current working directory do not specify this argument.
    
    ext : str, optional
        Extension of the given Video File e.g `.avi`, `.mp4`. The default is '.avi'.
    
    frames_dir : str, optional
        Path to the directory where frames will be saved. The default is 'train_1'.
    
    extract_rate : int or str, optional
        This argument specifies how many frames should be extrcated from each 1 second of video. If the value is 
        `all` it will etract all the frames in every second i.e if the frame rate of video is 25 fps it will
        extrcat all 25 frames. Other wise specify a number if you want to etract specific numbers of frames
        per each second e.g if 5 is given it will extrcat 5 frames from each 1 second. The default is `all`.
    
    frames_ext : str, optional
        The extension for the extracted frames/images e.h '.tif' or '.jpg'. The default is '.jpg'.
    Returns
    -------
    None.
    """
    os.makedirs(frames_dir, exist_ok=True)
    # capturing the video from the given path
    if ext not in v_file:
        v_file += ext
    cap = cv2.VideoCapture(path+v_file)   
    
    frameRate = cap.get(5) #frame rate

    #duration = int(cap.get(7)/frameRate)
    os.makedirs(frames_dir+'/'+v_file, exist_ok=True)
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
            
        if type(extract_rate)==int:
            if extract_rate>frameRate:
                print('Frame rate of Given Video: {0} fps'.format(frameRate))
                raise ValueError('The value of `extract_rate` argument can not be greater than the Frame Rate of the video.')
            
            if (frameId % extract_rate == 0) and extract_rate>1:
                # storing the frames in a new folder named train_1
                filename = frames_dir + '/' + v_file+ '/'+"_frame{0}".format(count)+frames_ext;count+=1
                cv2.imwrite(filename, frame)
            elif extract_rate==1:
                if (frameId % math.floor(frameRate) == 0):
                    filename = frames_dir + '/' + v_file+ '/'+"_frame{0}".format(count)+frames_ext;count+=1
                    cv2.imwrite(filename, frame)
        elif type(extract_rate)==str:
            if extract_rate=='all':
                # storing the frames in a new folder named train_1
                filename = frames_dir + '/' + v_file+ '/'+  v_file + "_frame{0}".format(count)+frames_ext;count+=1
                cv2.imwrite(filename, frame)
            else:
                raise ValueError('Invalid Value for argument `extract_rate`, it can be either `all` or an integer value.')
    cap.release()    

def ReadFileNames(path, frames_ext='.tif'):
    """
    This method will retrieve the Folder/File names from the dataset.
    Parameters
    ----------
    path : string
      Location of the data set that needs to be preprocessed.
    Returns
    -------
    onlyfiles : list
      A list containing all the subfolder names in the dataset.
    file_names : list
      A list containing all the names of the frames/images from 'onlyfiles'.
    directories: list
      A list containing all the names of the folders containing the images.
    """
    directories = [name for name in os.listdir(path) if os.path.isdir(path+'/'+name)]
    onlyfiles = []
    file_names = []
    
    for i in range (len(directories)):
        files = glob.glob(path+'/'+directories[i]+'/*{0}'.format(frames_ext))
        names = []
        for file in files:
            names.append(file.split("\\")[1])
        file_names.append(names)
        onlyfiles.append(files)
    return onlyfiles, file_names, directories

def ToJson(obj, name, path='./', json_dir=False):
    """ 
    Write the Given Object to JSON file on the given path and name.
        
    Arguments:
    ----------
    obj: 
      The object/variable you want to write to JSON file, it can be list or dictionary.
        
    name: 
      Name for the JSON file.
        
    path: 
      Path in which you want to save the JSON file. If want to save the JSON file in the current
      directory, do not specify the value of this argument.
              
    json_dir: 
      Boolean value if set to `True` a new directory with the name of `JSON` will be added at
      the end of the given path.
                  
    Returns:
    --------
    None
    """
    if json_dir:
        os.makedirs(path+'/JSON', exist_ok=True)
        with open(path+'/JSON/{0}.json'.format(name), 'w') as f:
            json.dump(obj, f)
        f.close()
    elif not json_dir:
        if '.json' in name:
            pass
        else:
            name=name+'.json'
        with open(path+'/'+name, 'w') as f:
            json.dump(obj, f)
        f.close()

def ProcessImg(img_name, read_path, write=True, write_path=None, res_shape=(128,128)):
    '''
    This function reads the images/frames, resizes them, converts them to 
    gray scale, and return them in gray scale.

    Parameters
    ----------
    img_name : string
      Name of the images to be read from the directory.
    read_path : string
      Path from where to read the images.
    write : boolean, optional
      if you want to save the images. The default is True.
    write_path : string, optional
      Path where to write/save the images. The default is None.
    res_shape : tuple, optional
      Dimensions to which you want to resize the images. The default is (128,128).

    Raises
    ------
    TypeError
      While the 'write' argument is True, 'write_path' should have a path, it could not be None.

    Returns
    -------
    gray : image array
      Resized gray scale image.

    '''
    if write and write_path is None:
        raise TypeError('The value of argument cannot be `None` when, `write` is set to True. Provide a valid path, where processed image should be stored!')
    img=cv2.imread(read_path)
    #img=img_to_array(img)
    #Resize the Image to (227,227)
    img=cv2.resize(img,res_shape)
    
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray = np.dot(img, rgb_weights)
    
    if write:
        os.makedirs(write_path, exist_ok=True)
        cv2.imwrite(write_path+'/'+img_name, gray) 
    return gray

def GlobalNormalization(img_list, name=None, path='Train_Data', save_data=True):
    '''
    This function will apply the Global Normalization step on the 
    images that are preprocessed, and then save them as Numpy array
    type to the 'path' argument.

    Parameters
    ----------
    img_list : list
      List of preprocessed images.
    name : string, optional
      Name that will given to the saved numpy. The default is None.
    path : string, optional
      Path were to save the converted dataset. The default is 'Train_Data'.
    save_data : boolean, optional
      If you want to save the converted data or not. The default is True.

    Raises
    ------
    TypeError
      Error is due to the contradiction, if you have set the 'save_data' argument to True
      you'll have to pass the 'name', it could not be None.

    Returns
    -------
    img_arr : Numpy Array
      A numpy array that would be feed to the model for training/testing.

    '''
  
    img_arr = np.array(img_list, dtype=np.float32)
    del img_list
    batch,height,width = img_arr.shape
    #Reshape to (227,227,batch_size)
    img_arr.resize(height,width,batch)
    #Normalize
    img_arr=(img_arr-img_arr.mean())/(img_arr.std())
    #Clip negative Values
    img_arr=np.clip(img_arr,0,1)
    if save_data:
        if name==None:
            raise TypeError('The value of the `name` argument cannot be `None` type, when `save_data` is set to True. Provide value with `str` datatype.')
        if '.npy' not in name:
            name += '.npy'
        os.makedirs(path, exist_ok=True)
        np.save(path+'/'+name, img_arr)
        print('\n------ Data Save Succefully at this path: {0} ------\n'.format(path))
    return img_arr

def Vid2Frame(vid_path, frames_dir, ext_vid='.avi', frames_ext='.tif'):
    '''
    A mini function to call another fucntion Frame_Extractor,
    it will not only extract the frames from training videos but 
    also from the testing videos.

    Parameters
    ----------
    vid_path : string
      Path for the folders like for training set of videos, and testing set of videos.
    frames_dir : sting
      This is the path where the extracted frames will be saved.
    ext_vid : string, optional
      Extension of the videos. The default is '.avi'.
    frames_ext : string, optional
      Extension of the frames. The default is '.tif'.

    Returns
    -------
    None.

    '''  
    vids = glob.glob(vid_path+'/*{0}'.format(ext_vid))
    
    for vid in tqdm(vids):
        path = vid.split('\\')[0]+'/'
        v_file = vid.split('\\')[1]
        Frame_Extractor(v_file, path=path, ext=ext_vid, frames_dir=frames_dir, 
                        extract_rate='all', frames_ext=frames_ext)  
        
def Fit_Preprocessing(path, frames_ext):
    '''
    This function is used to read the frames from the paths given to it,
    and then apply the preprocessing steps and convert those preprocessd images
    into a list of processed images in gray scale.

    Parameters
    ----------
    path : string
      Address of the paths from where to read the images.
    frames_ext : string
      Extension of the frames to be read. e.g. '.tif'

    Raises
    ------
    TypeError
      Gives an error when the extension of the frames is not specified.

    Returns
    -------
    img_list : list
      list of images, that are preprocessed in gray scale.

    '''  
    if frames_ext is None:
        raise TypeError('Invalid Value for argument `frames_ext`, it cannot be None. Give proper extensions of the frames e.g: `.tif` or `.png` etc!')
    print('\n\nProcessing Images in this Dataset Path: {0}\n'.format(path))
    onlyfiles, file_names, dirs = ReadFileNames(path, frames_ext)
    img_list = []
    for i in tqdm(range(len(onlyfiles))):
        images = onlyfiles[i]
        count = 0
        for img in images:
            img.split('/')
            img_name = dirs[i]+'_'+file_names[i][count]
            write_path = 'ProcessedImages/'+path.split('/')[1]
            gray = ProcessImg(img_name, read_path=img, write=True, 
                              write_path=write_path, res_shape=(227,227))
            img_list.append(gray)
            count += 1    
    return img_list
    
if __name__=='__main__':    
    '''
      Uncomment this section first so that the videos are converted to frames and 
      stored to the Datasets Folder, after that comment it back.
    '''
    # vid_paths = ['AvenueDataset/training_videos', 'AvenueDataset/testing_videos']
    # for vid_path in vid_paths:
    #     print('\n\nExtracting Frame from the videos, Path: {0}\n'.format(vid_path))
    #     frames_dir = 'Datasets/'+vid_path
    #     Vid2Frame(vid_path, frames_dir, ext_vid='.avi', frames_ext='.tif')
    # print('\n-------- Frames are Extracted from All Videos Succesfully! --------\n')  
        
    
    #path = frames_dir
    paths = ['Datasets/UCSDped1/Train', 'Datasets/UCSDped2/Train', 
              'Datasets/AvenueDataset/training_videos']
    '''
      Uncomment this after training of the model is complete;
      and to prepare the testing dataset for testing the model.
    '''
    #paths = ['Datasets/UCSDped1/Test', 'Datasets/UCSDped2/Test', 
    #         'Datasets/AvenueDataset/testing_videos']
    
    '''
      While using this for loop for preparing the training dataset for training
      replace the Test--->Train, and when the training is complete and 
      you want to test the model, do the vise versa of the above to create the
      testing data set for testing.
    '''
    for path in paths:
        img_list = Fit_Preprocessing(path, frames_ext='.tif')
        name='Test_{0}.npy'.format(path.split('/')[1])
        img_arr = GlobalNormalization(img_list, name, path='Test_Data', save_data=True)