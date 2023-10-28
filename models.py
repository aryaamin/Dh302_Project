from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, GlobalMaxPool1D, concatenate, LSTM, Reshape, Input, Conv1D, MaxPool1D, Dropout, LocallyConnected1D, Concatenate, Dense, Flatten
from tcn import TCN

WINDOW_SIZE = 100
Fs = 100

def model_cnn():
    inputLayer = Input(shape=(3000, 1), name='inLayer')
    conv = Conv1D(16, kernel_size=5, activation='relu', padding="valid")(inputLayer)
    conv = Conv1D(16, kernel_size=5, activation='relu', padding="valid")(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv)
    conv = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv)
    conv = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = Dropout(rate=0.01)(conv)
    conv = LocallyConnected1D(128, kernel_size=3, activation='selu', padding="valid", name='layer_17', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(conv)
    conv = MaxPool1D()(conv)
    conv = Dropout(rate=0.01)(conv)
    conv = Flatten(name='flattened_model')(conv)

    nclass = 5
    outLayer = Reshape((1,conv.get_shape()[1]), name='reshape1')(conv)

    # Output layer
    outLayer = Dense(nclass, activation="softmax")(outLayer)
    outLayer = Reshape((nclass,), name='reshape2')(outLayer)

    model = Model(inputLayer, outLayer)
    
    # Compile the model
    model.compile(optimizer=Adam(0.001, amsgrad=True), sample_weight_mode="temporal", loss='categorical_crossentropy', metrics=['acc'])

    return model

def model_tcnn():
    # Define model1
    inputLayer = Input(shape=(3000, 1), name='inLayer')
    conv1 = Conv1D(64, kernel_size=200, activation='selu', strides=20, padding="valid", name='layer_11')(inputLayer)
    conv1 = Conv1D(64, kernel_size=6, activation='selu', padding="valid", name='layer_12')(conv1)
    conv1 = MaxPool1D(pool_size=3, strides=3)(conv1)
    conv1 = Conv1D(32, kernel_size=6, activation='selu', padding="valid", name='layer_13')(conv1)
    conv1 = Conv1D(32, kernel_size=6, activation='selu', padding="valid", name='layer_14')(conv1)
    conv1 = MaxPool1D(pool_size=2, strides=2)(conv1)
    conv1 = Conv1D(32, kernel_size=3, activation='selu', padding="valid", name='layer_15', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(conv1)
    conv1 = Conv1D(32, kernel_size=3, activation='selu', padding="valid", name='layer_16')(conv1)
    conv1 = MaxPool1D(pool_size=2, strides=2)(conv1)
    conv1 = Dropout(rate=0.01)(conv1)
    conv1 = LocallyConnected1D(128, kernel_size=3, activation='selu', padding="valid", name='layer_17', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(conv1)
    conv1 = MaxPool1D()(conv1)
    conv1 = Dropout(rate=0.01)(conv1)
    conv1 = Flatten(name='flattened_model1')(conv1)

    # Define model3
    conv3 = Conv1D(64, kernel_size=25, strides=3, activation='selu', padding="valid", name='layer_31')(inputLayer)
    conv3 = Conv1D(64, kernel_size=8, activation='selu', padding="valid", name='layer_32')(conv3)
    conv3 = MaxPool1D(pool_size=4, strides=4)(conv3)
    conv3 = Conv1D(32, kernel_size=8, activation='selu', padding="valid", name='layer_33')(conv3)
    conv3 = Conv1D(32, kernel_size=8, activation='selu', padding="valid", name='layer_34')(conv3)
    conv3 = MaxPool1D(pool_size=2, strides=2)(conv3)
    conv3 = Conv1D(32, kernel_size=5, activation='selu', padding="valid", name='layer_25', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(conv3)
    conv3 = Conv1D(32, kernel_size=5, activation='selu', padding="valid", name='layer_26')(conv3)
    conv3 = MaxPool1D(pool_size=2, strides=2)(conv3)
    conv3 = Dropout(rate=0.01)(conv3)
    conv3 = LocallyConnected1D(128, kernel_size=3, activation='selu', padding="valid", name='layer_37', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(conv3)
    conv3 = MaxPool1D()(conv3)
    conv3 = Dropout(rate=0.01)(conv3)
    conv3 = Flatten(name='flattened_model3')(conv3)

    nclass = 5

    outLayer = Concatenate()([conv1, conv3])
    outLayer = Reshape((1,outLayer.get_shape()[1]), name='reshape1')(outLayer)


    # TCN Layer
    outLayer = TCN(return_sequences=True, name='flaten1')(outLayer)

    # Output layer
    outLayer = Dense(nclass, activation="softmax")(outLayer)
    outLayer = Reshape((nclass,), name='reshape2')(outLayer)

    # Create the final model
    model = Model(inputLayer, outLayer)

    # Compile the model
    model.compile(optimizer=Adam(0.001, amsgrad=True), sample_weight_mode="temporal", loss='categorical_crossentropy', metrics=['acc'])

    return model

def model_lstm(n_classes=5, use_sub_layer=False, use_rnn=True, verbose=False):
    inputLayer = Input(shape=(3000, 1), name='inLayer')
    convFine = Conv1D(filters=64, kernel_size=int(Fs/2), strides=int(Fs/16), padding='same', activation='relu', name='fConv1')(inputLayer)
    convFine = MaxPool1D(pool_size=8, strides=8, name='fMaxP1')(convFine)
    convFine = Dropout(rate=0.5, name='fDrop1')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv2')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv3')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv4')(convFine)
    convFine = MaxPool1D(pool_size=4, strides=4, name='fMaxP2')(convFine)
    fineShape = convFine.get_shape()
    convFine = Flatten(name='fFlat1')(convFine)
    
    # network to learn coarse features
    convCoarse = Conv1D(filters=32, kernel_size=Fs*4, strides=int(Fs/2), padding='same', activation='relu', name='cConv1')(inputLayer)
    convCoarse = MaxPool1D(pool_size=4, strides=4, name='cMaxP1')(convCoarse)
    convCoarse = Dropout(rate=0.5, name='cDrop1')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv2')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv3')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv4')(convCoarse)
    convCoarse = MaxPool1D(pool_size=2, strides=2, name='cMaxP2')(convCoarse)
    coarseShape = convCoarse.get_shape()
    convCoarse = Flatten(name='cFlat1')(convCoarse)
    
    # concatenate coarse and fine cnns
    mergeLayer = concatenate([convFine, convCoarse], name='merge_1')
    outLayer = Dropout(rate=0.5, name='mDrop1')(mergeLayer)
    
    outLayer = Reshape((1, outLayer.get_shape()[1]), name='reshape1')(outLayer)
    outLayer = LSTM(64, return_sequences=True)(outLayer)
    outLayer = LSTM(64, return_sequences=False)(outLayer)
    outLayer = Dense(n_classes, activation='softmax', name='outLayer')(outLayer)
    
    model = Model(inputLayer, outLayer)
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])

    return model


# def model_tcnn_crf():
    
#     # Define model1
#     inputLayer = Input(shape=(3000, 1), name='inLayer')
#     conv1 = Conv1D(64, kernel_size=200, activation='selu', strides=20, padding="valid", name='layer_11')(inputLayer)
#     conv1 = Conv1D(64, kernel_size=6, activation='selu', padding="valid", name='layer_12')(conv1)
#     conv1 = MaxPool1D(pool_size=3, strides=3)(conv1)
#     conv1 = Conv1D(32, kernel_size=6, activation='selu', padding="valid", name='layer_13')(conv1)
#     conv1 = Conv1D(32, kernel_size=6, activation='selu', padding="valid", name='layer_14')(conv1)
#     conv1 = MaxPool1D(pool_size=2, strides=2)(conv1)
#     conv1 = Conv1D(32, kernel_size=3, activation='selu', padding="valid", name='layer_15', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(conv1)
#     conv1 = Conv1D(32, kernel_size=3, activation='selu', padding="valid", name='layer_16')(conv1)
#     conv1 = MaxPool1D(pool_size=2, strides=2)(conv1)
#     conv1 = Dropout(rate=0.01)(conv1)
#     conv1 = LocallyConnected1D(128, kernel_size=3, activation='selu', padding="valid", name='layer_17', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(conv1)
#     conv1 = MaxPool1D()(conv1)
#     conv1 = Dropout(rate=0.01)(conv1)
#     conv1 = Flatten(name='flattened_model1')(conv1)

#     # Define model3
#     conv3 = Conv1D(64, kernel_size=25, strides=3, activation='selu', padding="valid", name='layer_31')(inputLayer)
#     conv3 = Conv1D(64, kernel_size=8, activation='selu', padding="valid", name='layer_32')(conv3)
#     conv3 = MaxPool1D(pool_size=4, strides=4)(conv3)
#     conv3 = Conv1D(32, kernel_size=8, activation='selu', padding="valid", name='layer_33')(conv3)
#     conv3 = Conv1D(32, kernel_size=8, activation='selu', padding="valid", name='layer_34')(conv3)
#     conv3 = MaxPool1D(pool_size=2, strides=2)(conv3)
#     conv3 = Conv1D(32, kernel_size=5, activation='selu', padding="valid", name='layer_25', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(conv3)
#     conv3 = Conv1D(32, kernel_size=5, activation='selu', padding="valid", name='layer_26')(conv3)
#     conv3 = MaxPool1D(pool_size=2, strides=2)(conv3)
#     conv3 = Dropout(rate=0.01)(conv3)
#     conv3 = LocallyConnected1D(128, kernel_size=3, activation='selu', padding="valid", name='layer_37', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(conv3)
#     conv3 = MaxPool1D()(conv3)
#     conv3 = Dropout(rate=0.01)(conv3)
#     conv3 = Flatten(name='flattened_model3')(conv3)
    

#     seq_input1 = Input(shape=(None, 3000, 1))
#     seq_input3 = Input(shape=(None, 3000, 1))

#     nclass = 5
#     # encoded_sequence1 = TimeDistributed(conv1)(seq_input1)
#     # encoded_sequence3 = TimeDistributed(conv3)(seq_input3)
#     outLayer = Concatenate()([conv1, conv3])
#     outLayer = Reshape((1,outLayer.get_shape()[1]), name='reshape1')(outLayer)
    
#     outLayer = TCN(return_sequences=True, name='flaten1')(outLayer)
#     outLayer1 = Dense(nclass, activation="softmax")(outLayer)
#     outLayer1 = Reshape((nclass,), name='reshape2')(outLayer1)
    
#     model1 = Model(inputLayer, outLayer1)

#     # Compile the model
#     model1.compile(optimizer=Adam(0.001, amsgrad=True), sample_weight_mode="temporal", loss='categorical_crossentropy', metrics=['acc'])
#     # print(model1.summary())
#     seq_input = Input(shape=(None, 3000, 1))
#     encoded_sequence = TimeDistributed(model1)(seq_input)
#     encoded_sequence = Reshape((nclass,), name='reshape2')(encoded_sequence)

#     # print(encoded_sequence.shape)
#     crf = CRF(nclass, sparse_target=True)
#     out = crf(outLayer)
#     out = Dense(nclass, activation="softmax")(out)
#     out = Reshape((nclass,), name='reshape2')(out)
    
#     print(inputLayer.shape)
#     print(out.shape)
    
#     model = Model(inputLayer, out)

#     model.compile(Adam(0.001), loss='categorical_crossentropy', metrics=['acc'])
    
#     return model