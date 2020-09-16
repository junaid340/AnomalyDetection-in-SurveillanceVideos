from matplotlib import pyplot as plt
import pickle

def PlotHistory(history, name, show=True, save=False, path=None):
    '''
    A function to plot the Training loss and Validation loss of the model.

    Parameters
    ----------
    history : dictionary
      Dictionary that contains all the training loss and validation loss values.
    name : string
      Name for the graph, or title for the graph
    show : boolean, optional
      The default is True.
    save : strin, optional
      To save the plot at the given destination in '.png' format. The default is False.
    path : string, optional
      Path for storing the plots of the model. The default is None.

    Raises
    ------
    ValueError
      If you want to save the plots, give a valid path where the image of plots
      could be saved.

    Returns
    -------
    None.

    '''
    plt.clf()
    plt.ioff()
    plt.figure(1)
    #plots the training loss of the model
    plt.plot(history['loss'])
    
    #plots the validation loss of the model
    plt.plot(history['val_loss'])
    
    #Assigning Titles to the graph
    plt.title('Model loss- '+name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Val Loss'], loc='upper right')
    #Saving the graph as .png image
    if save:
        if path==None:
            raise ValueError('Path cannot be None when `save` is set to True, please provide valid path.')
        plt.savefig(path+'/'+name+'_Loss.png')
    
    plt.figure(2)
    #plots accuracy of the model
    plt.plot(history['accuracy'])
    #Assigning Titles to the graph
    plt.title('Model Accuracy- '+name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Model Accuracy'], loc='lower right')
    #Saving the graph as .png image
    if save:
        if path==None:
            raise ValueError('Path cannot be None when `save` is set to True, please provide valid path.')
        plt.savefig(path+'/'+name+'_Accuracy.png')
    if show:
        plt.show()

#pass the saved history file to evaluate the model
with open('checkpoints/Model_History', 'rb') as f:
    hist = pickle.load(f)

PlotHistory(hist, name='Our_Model', show=True, save=True, path='./')