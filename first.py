import os
import numpy as np
from datetime import datetime
import sys
import cPickle as pickle
import matplotlib.pyplot as pyplot
import pandas as pd
from lasagne.updates import nesterov_momentum
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano

FTRAIN='/home/curiousguy13/kaggle/facial_feature_recog/training.csv'
FTEST='/home/curiousguy13/kaggle/facial_feature_recog/test.csv'
FLOOKUP = '/home/curiousguy13/kaggle/facial_feature_recog/IdLookupTable.csv'

col=('left_eye_center_x','left_eye_center_y','right_eye_center_x',
 'right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y',
 'left_eye_outer_corner_x','left_eye_outer_corner_y',
 'right_eye_inner_corner_x','right_eye_inner_corner_y',
 'right_eye_outer_corner_x','right_eye_outer_corner_y',
 'left_eyebrow_inner_end_x','left_eyebrow_inner_end_y',
 'left_eyebrow_outer_end_x','left_eyebrow_outer_end_y',
 'right_eyebrow_inner_end_x','right_eyebrow_inner_end_y',
 'right_eyebrow_outer_end_x','right_eyebrow_outer_end_y','nose_tip_x',
 'nose_tip_y','mouth_left_corner_x','mouth_left_corner_y',
 'mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x',
 'mouth_center_top_lip_y','mouth_center_bottom_lip_x',
 'mouth_center_bottom_lip_y')

print col
def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* only if you are interested in a subset of the target columns
    """
    fname=FTEST if test else FTRAIN
    df=read_csv(os.path.expanduser(fname))  #load pandas dataframe

    #convert space separated image pixel values to numoy array
    df['Image']=df['Image'].apply(lambda im: np.fromstring(im,sep=' '))
    if cols:
        df=df[list(cols)+['Image']]


    df=df.dropna()

    #scale pixel values to[0,1] ( ############not sure why###### )
    x=np.vstack(df['Image'].values)/255
    x=x.astype(np.float32)

    #print(x.shape)
    # only FTRAIN has any target columns
    if not test:
        print df
       #print df.columns[:-1]
        y=df[df.columns[:-1]].values
        print y
        y=(y-48)/48 # scale target coordinates to [-1, 1]
        x, y=shuffle(x,y,random_state=42)   #shuffle training data
        y=y.astype(np.float32)
    else:
        y=None

    return x, y

def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def predict(fname_specialists='net1.pickle'):
    with open(fname_specialists, 'rb') as f:
        specialists = pickle.load(f)

    '''
    X = load2d(test=True)[0]
    y_pred = np.empty((X.shape[0], 0))

    for model in specialists.values():
        y_pred1 = model.predict(X)
        y_pred = np.hstack([y_pred, y_pred1])

    columns = ()
    for cols in specialists.keys():
        columns += cols
    '''


    X, _ =load(test=True)
    y_pred2=specialists.predict(X)
    y_pred2 = y_pred2 * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = DataFrame(y_pred2, columns=col)

    '''plot'''
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred2[i], ax)

    pyplot.show()

    '''submission'''
    lookup_table = read_csv(os.path.expanduser(FLOOKUP))
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
            ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=300,  # we want to train this many epochs
    verbose=1,
    )

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
    )
'''
X, y = load2d()  # load 2-d data
net2.fit(X, y)

with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)
'''
#predict('net1.pickle')


#X,y=load()


#net1.fit(X,y)


with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)

with open('net1.pickle', 'rb') as f:
            net1 = pickle.load(f)

#predict()
'''
train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()
'''

X, _ = load(test=True)
y_pred = net1.predict(X)


fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()
