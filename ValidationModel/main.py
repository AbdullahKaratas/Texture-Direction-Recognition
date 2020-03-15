# to come
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def _plotResult(test_predictions, label_data):
    plt.figure(figsize=(20, 15), dpi=60)
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 14

    plt.rc('font', size=MEDIUM_SIZE)        
    plt.rc('axes', titlesize=BIGGER_SIZE)     
    plt.rc('axes', labelsize=BIGGER_SIZE)    
    plt.rc('xtick', labelsize=BIGGER_SIZE)    
    plt.rc('ytick', labelsize=BIGGER_SIZE)    
    plt.rc('legend', fontsize=BIGGER_SIZE)    
    plt.rc('figure', titlesize=BIGGER_SIZE)  
    plt.subplot(2, 1, 1)
    plt.plot(test_predictions, 'k', linewidth=4)
    plt.plot(label_data, '0.2')
    plt.title('Prediction of Model and Ground Truth')
    plt.ylabel('Arc Minute')
    plt.xlabel('Sample number')
    plt.legend(['Model Prediction', 'Ground Truth'], loc='lower right'),
    plt.grid(True)
    plt.plot(test_predictions)
    plt.ylim(-6.1, 6.1)
    plt.subplot(2, 1, 2)
    plt.title('Differenz of Prediction and Ground Truth')
    plt.ylabel('Arc Minute')
    plt.xlabel('Sample number')
    plt.stem(np.linspace(0,100,101), label_data - test_predictions, 'k', use_line_collection=True, markerfmt='ko', basefmt='k')
    plt.ylim(-3.1, 3.1)
    plt.grid(True)
    plt.plot([0, 101], [1, 1], '--k')
    plt.plot([0, 101],[-1, -1] ,'--k')
    plt.show()

def main():

    model = tf.keras.models.load_model('SaveModel.h5')

    print(model.summary())

    test_data = np.load('dataTest.npy')
    label_data = np.load('labelTest.npy')

    if tf.keras.backend.image_data_format() == 'channel_first':
        test_data = test_data.reshape(test_data.shape[0], 1,  test_data.shape[1], test_data.shape[2])
    else:
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)

    test_predictions = model.predict(test_data).flatten()
    _plotResult(test_predictions, label_data)


if __name__ == "__main__":
    main()
