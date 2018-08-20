from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)

# Time-Distributed (VGG16 base) imports
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import Input
from keras.layers.core import Flatten, Dropout, Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD


# In[18]:


class NetworkModel:
    
    def lstm(self): #Testing our dataset on basic LSTM model
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=(20, 512),
                       dropout=0.7))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(12, activation='softmax'))
        optimizer = Adam(lr=1e-4, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                                   metrics=['accuracy'])
        return model
    
    def time_distributed_vgg(self):
    
    #     bottleneck_train_path = 'bottleneck_features_train.npy'
    #     bottleneck_val_path = 'bottleneck_features_val.npy'
    #     top_model_weights = 'bottleneck_TOP_LAYER.h5'

        input_shape = (20, 48, 50, 3)
        input_layer = Input(shape=input_shape)
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(48, 50, 3))
        vgg = Model(input=vgg_base.input, output=vgg_base.output)

        for layer in vgg.layers[:15]:
            layer.trainable = False

        x = TimeDistributed(vgg)(input_layer)

        model = Model(input=input_layer, output=x)

        input_layer_2 = Input(shape=model.output_shape[1:])

        x = TimeDistributed(Flatten())(input_layer_2)

        lstm = LSTM(256)

        x = Bidirectional(lstm, merge_mode='concat', weights=None)(x)

        x = Dropout(rate=0.5)(x) #Dropout rate may vary

        x = Dense(12)(x) #Predict 12 classes

        preds = Activation('softmax')(x)

        model_top = Model(input=input_layer_2, output=preds)

        #model_top.load_weights(top_model_weights)

        x = model(input_layer)
        preds = model_top(x)

        final_model = Model(input=input_layer, output=preds)

        adam = Adam(lr=0.0001) #WAIT THIS ISNT ADAM! IT IS NOW. It used to be SGD
        final_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        return final_model
    
    def test_model_accuracy(self, model_topo, model_weights, test_data_dir, batch_size=32):
        '''
        Accepts a specified model topology, loads its weights, and runs a test to 
        check for model score and accuracy. Each model topology will have a different way of 
        evaluation.
        model_topo -> ['lstm', 'td_vgg']
        model_weights -> path to the *.h5 weights file of the model
        test_data_dir -> directory containing the processed test dataset
        '''
        if model_topo == 'lstm':
            model = self.lstm()
            model.load_weights(model_weights)
            X_test = np.load(os.path.join(test_data_dir, 'test_features'))
            y_test = np.load(os.path.join(test_data_dir, 'y_test'))
            one_hot_test = keras.utils.to_categorical(y_test, num_classes=12) #One-hot encode
            score, acc = model.evaluate(X_test, one_hot_test,
                            batch_size=batch_size)
        elif model_topo == 'td_vgg':
            model = self.time_distributed_vgg()
            model.load_weights(model_weights)
            generator = DataGenerator(test_data_dir)
            score, acc = model.evaluate_generator(generator.sequence_generator('test'), steps=30)
        
        print('Score: {0}, Acc: {1}'.format(score, acc))

