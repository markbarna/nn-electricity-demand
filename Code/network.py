import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import makedirs

#################
# HYPERPARAMETERS
#################
# model to use (if False, use FFN sigmoid model)
LSTM_net = True

# include differencing and number of periods to difference
differencing = True
t = 24

# zone and features to use
# COAST, EAST, FWEST, NCENT, NORTH, SCENT, SOUTH, WEST
zone = 'WEST'
features = ['DryBulbTempC', 'Load']
y_feature = 'Load'

# variables and periods to lag
timestep = 1
lag_features = [('DryBulbTempC', timestep), ('Load', timestep)] # FFN model

# normalization range (list)
norm_y_range = [-1, 1]

# number of hidden layer neurons
hidden_n = 1

# number of epochs & batch_size for training
epochs = 5
batch_size = 100
# number of training runs
trials = 50
# custom batch number
batch = 1

##########################
# DATASET CLASS DEFINITION
##########################

class Dataset:

    def __init__(self, path, zone, features):
        self.df = pd.read_csv(path, header=[0,1], index_col=0)
        self.df = self.df[zone]
        self.df = self.df[features]

    def difference(self, diffs):
        '''Difference the dataframe to approximate stationary data. Create a copy
        of the original dataframe first. Remove the initial rows with NaNs.

        Args:
            diffs (int): Number of periods for differencing.
        '''
        self.df_ = self.df.copy()
        self.df = self.df.diff(-diffs)
        self.df = self.df.dropna(axis=0)

        self.diffs = diffs

    def undifference(self):
        '''Invert differencing on unnormalized predictions
        '''
        # pull target column from copy of original dataframe, subset to testing set
        actuals = self.df_.loc[:,[self.target_col]]
        if LSTM_net:
            #actuals = actuals[self.split+timestep:-self.diffs]
            actuals = actuals[self.split+timestep:self.split+timestep+self.prediction.shape[0]]
        else:
            actuals = actuals[self.split:-self.diffs]

        # convert prediction to dataframe
        self.prediction = pd.DataFrame(self.prediction)
        # set index to match
        idx = actuals.index
        self.prediction = self.prediction.set_index(idx)

        # undifference
        self.prediction.iloc[:,[0]] = self.prediction.values + actuals.values

        # join prediction and actuals, change index to datetime format
        self.prediction = self.prediction.join(actuals)
        self.prediction.index = pd.to_datetime(self.prediction.index)
        self.prediction.columns = ['Prediction', 'Actual']

    def lag(self, lags, y):
        '''Add lag terms to time series. For specified variable(s),
        shift specified number of periods.

        Args:
            lags (list of tuples): [(column name, periods)]
            y (string): Column name of y variable.
        '''
        # save list of regressor column names
        x_cols = list(self.df.columns)
        x_cols.remove(y)

        for lag in iter(lags):
            # for each lag period
            for i in range(1, lag[1]+1):
                # join lagged columns
                self.df = self.df.join(self.df[lag[0]].shift(i), rsuffix='_l{}'.format(i))

        # drop first rows with NaNs
        self.df = self.df.dropna(axis=0)

        # drop non-lagged variables other than the target column(y)
        self.df = self.df.drop(x_cols, axis=1)

    def split(self, test_size):
        '''Split the dataframe into training and testing sets, ensuring the the
        testing set is divisible by batch size.

        Args:
            test_size (float): Percent size of testing set (0 to 1).
        '''
        # calculate number of rows
        nrows = self.df.shape[0]

        split = self.df.shape[0] - np.rint(nrows * test_size)
        # ensure that the split point makes the training set evenly-divisible by the batch size
        split = split - split % batch_size
        split = split.astype(int)

        self.training = self.df.iloc[:split,:]
        self.testing = self.df.iloc[split:,:]

        self.split = split

    def normalize(self, y_min, y_max):
        '''Normalize the training set into the given range. Apply the training
        set normalization to the testing set. Formula:
        y_min + (x - x_min)*(y_range) / x_range

        Args:
            y_min (int): Lower bound of range to normalize within.
            y_max (int): Upper bound of range to noramlize within.
        '''
        self.y_min = y_min
        self.y_max = y_max
        self.y_range = y_max - y_min

        # find min, max, range of training set
        self.x_min = self.training.min(axis=0)
        self.x_max = self.training.max(axis=0)
        self.x_range = self.x_max - self.x_min

        # apply normalization to both sets
        self.training = self.y_min + (self.training - self.x_min) * self.y_range / self.x_range
        self.testing = self.y_min + (self.testing - self.x_min) * self.y_range / self.x_range

    def unnormalize(self, a):
        '''Invert normalization applied with normalization funtion.

        Args:
            a (array): Array to unnormalize.
        '''
        self.prediction = (self.x_min[self.target_col] +
            (a - self.y_min) * self.x_range[self.target_col] / self.y_range)


    def to_array(self, target_col, ret_train=True):
        '''Convert either training or testing set to numpy array and return inputs
        and targets as separate arrays.

        Args:
            target_col (string): Name of target variable column.
            ret_train (bool): True (default) for training set; False for testing set.

        Return:
            Two arrays: input, target
        '''
        self.target_col = target_col

        if ret_train:
            input = self.training.drop(target_col, axis=1)
            target = self.training[target_col]

        else:
            input = self.testing.drop(target_col, axis=1)
            target = self.testing[target_col]

        # convert to numpy array
        input = input.values
        # reshape to 2-D array
        target = target.values.reshape(-1,1)

        return input, target

    def to_3Darray(self, target_col, ret_train=True):
        '''Convert either training or testing set to 3D numpy array for RNN
        (samples x timesteps x features) and return inputs and targets as separate arrays,
        discarding rows with NaNs due to time shift.

        Args:
            target_col (string): Name of target variable column.
            timesteps (int): Number of timesteps to present to network.
            ret_train (bool): True (default) for training set; False for testing set.

        Return:
            Two 3D arrays: input, target
        '''
        self.target_col = target_col

        if ret_train:
            input = self.training
            # reshape input to be 3D array (samples x timesteps x features)
            input = input.values.reshape(-1, timestep, self.training.shape[1])
            # drop initial row(s) corresponding to timesteps and attach same number
            # of rows from beginning of testing set to end of training set in order to
            # maintain continuity
            target = self.training.iloc[timestep:,:]
            target = target.append(self.testing.iloc[0:timestep,:])
            # keep only target column
            target = target[target_col]

        else:
            # drop final row(s) from input set to match up with time shift of target
            # and to even batch size
            input = self.testing.iloc[:-timestep,:]
            # reshape input to be 3D array (samples x timesteps x features)
            input = input.values.reshape(-1, timestep, self.training.shape[1])
            # select target column and drop initial row(s) corresponding to timesteps
            target = self.testing.iloc[timestep:,:]
            target = target[target_col]

        # reshape to 2-D array
        target = target.values.reshape(-1,1)



        return input, target

#####################
# NETWORK DEFINITIONS
#####################

# define FFN network
# input number of features (number of variables * number of lags)
# logsig hidden layer transfer function
# linear output function with 1 x 1 dim
def create_FFN():
    FFN = Sequential()
    FFN.add(Dense(hidden_n, activation='sigmoid'))
    FFN.add(Dense(1))
    FFN.compile(optimizer='Adam', loss='MSE')
    return FFN

# define LSTM network
def create_RNN():
    RNN = Sequential()
    RNN.add(LSTM(hidden_n, stateful=True, dropout=0, batch_input_shape=(batch_size,
        timestep, len(features))))
    RNN.add(Dense(1))
    RNN.compile(optimizer='Adam', loss='MSE')
    return RNN

#########################
# LOAD DATA & PRE-PROCESS
#########################

# instantiate dataset class
dt = Dataset(path='../Data/Cleaned/preprocessed.csv', zone=zone,
    features=features)

# add lagged variables
if not LSTM_net:
    dt.lag(lags=lag_features, y=y_feature)

# difference dataset
if differencing:
    dt.difference(diffs=t)

# split dataset into training and testing sets
dt.split(test_size=0.15)

# calculate normalization and apply to training and testing sets
dt.normalize(y_min=norm_y_range[0], y_max=norm_y_range[1])

# split inputs and target, reshape for network input
if LSTM_net:
    input, target = dt.to_3Darray(target_col=y_feature)
    test_input, test_target = dt.to_3Darray(target_col=y_feature, ret_train=False)

    # calculate length of testing sets to truncate (for testing) and zero pad (for prediction)
    # to correspond with batch size
    trunc_len = test_input.shape[0] % batch_size
    pad_len = batch_size - trunc_len

else:
    input, target = dt.to_array(target_col=y_feature)
    test_input, test_target = dt.to_array(target_col=y_feature, ret_train=False)

######################
# SAVE HYPERPARAMETERS
######################

# print hyperparameters to screen and file, creating directory structure to sort

dir_name = '../Plots/'

if LSTM_net:
    stream = 'Long Short-Term Network'
    dir_name += 'LSTM/'
    if differencing:
        dir_name += 'diff/'
    else:
        dir_name += 'nodiff/'
else:
    stream = 'Feedforward Network'
    dir_name += 'FFN/'

# add hyperparameters to stream variable string

stream += '\nDifferenced: '

if differencing:
    stream += 'True'
else:
    stream += 'False'

stream += '\nNumber of periods: ' + str(t) + '\nZone: ' + zone + '\nInput feature(s): '

for i in features:
    stream += i + ' '

stream += '\nOutput feature: ' + y_feature
stream += '\nLag period(s): ' + str(timestep)
stream += ('\nNormalization output range: [' + str(norm_y_range[0]) + ',' +
    str(norm_y_range[1]) + ']')
stream += '\nHidden layer neurons: ' + str(hidden_n)
stream += '\nEpochs: ' + str(epochs)
stream += '\nTrials: ' + str(trials)
stream += '\nGroup batch number: ' + str(batch)
stream += '\n'

# print hyperparameters to screen
print(stream)

dir_name += (zone + '/lag' + str(timestep) + '/epochs' + str(epochs) + '/batch' +
    str(batch) + '/')

makedirs(dir_name, exist_ok=True)

################################
# RUN NETWORK TRAINING & TESTING
################################

# create array to hold values of error states for each run
stats = np.zeros((trials, 3))
stats[:,0] = np.arange(1, trials+1)

for k in range(trials):

    # set up loop to test at each epoch
    running_train_loss = np.zeros(epochs)
    running_test_loss = np.zeros(epochs)

    # instantiate networks
    if LSTM_net:
        RNN = create_RNN()

    else:
        FFN = create_FFN()

    for i in range(epochs):
        # train the model
        if LSTM_net:
            train_metrics = RNN.fit(x=input, y=target, batch_size=batch_size,
                epochs=1, verbose=0, shuffle=False)
        else:
            train_metrics = FFN.fit(x=input, y=target, batch_size=batch_size, epochs=1, verbose=0)

        # get training loss and add to running tracker
        train_loss = train_metrics.history['loss'][0]
        running_train_loss[i] = train_loss

        # test the model
        # for LSTM, need to truncate set to divide evenly with batch size
        if LSTM_net:
            test_loss = RNN.evaluate(x=test_input[:-trunc_len], y=test_target[:-trunc_len],
                batch_size=batch_size, verbose=0)
        else:
            test_loss = FFN.evaluate(x=test_input, y=test_target, batch_size=batch_size, verbose=0)

        running_test_loss[i] = test_loss
        print('Trial {}/{}\nEpoch {}/{}:\n\
          Training Loss: {:.6f}\n\
          Test Loss: {:.6f}'
            .format(k+1, trials, i+1, epochs, train_loss, test_loss))


    # generate predictions
    if LSTM_net:
        # add zero padding to test input to allow input size to correspond to batch size
#        test_input = np.append(test_input,
#            np.zeros((pad_len, timestep, len(features))), axis=0)
        predict = RNN.predict(x=np.append(test_input, np.zeros((pad_len, timestep, len(features))), axis=0),
            batch_size=batch_size)
        # remove padded rows
        predict = predict[:-pad_len]
    else:
        predict = FFN.predict(x=test_input)


    # invert normalization and differencing on predictions
    dt.unnormalize(a=predict)
    if differencing:
        dt.undifference()

    # calculate MAE and MAPE of prediction and add to stats array
    stats[k,1] = (dt.prediction['Prediction'] - dt.prediction['Actual']).abs().mean()
    stats[k,2] = ((dt.prediction['Prediction'] - dt.prediction['Actual']).abs() /
        dt.prediction['Actual']).mean()


# convert stats to dataframe and save
stats = pd.DataFrame(stats, columns=['Trial', 'MAE', 'MAPE'])
stats = stats.set_index('Trial')
stats.to_csv(dir_name + 'performance_stats.csv')

#####################
# PERFORMANCE METRICS
#####################

# plot comparison of testing and training accuracy
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(running_test_loss, label='Testing Loss', alpha=0.8, color=plt.cm.tab20b(1))
ax.plot(running_train_loss, label='Training Loss', alpha=0.8, color=plt.cm.tab20c(4))
ax.set_title('Running Loss')
ax.set_ylabel('Loss (MSE)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right', fontsize='medium')
fig.savefig(dir_name + 'running_loss.png')
print('Running Loss plot saved.')

# plot comparison of testing and training accuracy last 50 epochs
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(running_test_loss[-50:], label='Testing Loss', alpha=0.8, color=plt.cm.tab20b(1))
ax.plot(running_train_loss[-50:], label='Training Loss', alpha=0.8, color=plt.cm.tab20c(4))
ax.set_title('Running Loss (Last 50 Epochs)')
ax.set_ylabel('Loss (MSE)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right', fontsize='medium')
fig.savefig(dir_name + 'running_loss_50.png')
print('Running Loss (Last 50 Epochs) plot saved.')

# test prediction vs actual values at several time sizes
ax = dt.prediction.resample('12H').mean().plot(alpha=0.8, figsize=[15,10],
    color=[plt.cm.Set1(1), plt.cm.Set1(2)])
ax.set_title(zone + ' Zone Prediction v. Actual')
ax.set_xlabel('Day')
ax.set_ylabel('Load')
yticks = ['{:,.0f} MW'.format(t) for t in ax.get_yticks()]
ax.set_yticklabels(yticks)
plt.savefig(dir_name + 'daily_comparison.png')

ax = dt.prediction.groupby(dt.prediction.index.hour).mean().plot(alpha=0.8, figsize=[15,10],
    color=[plt.cm.Set1(1), plt.cm.Set1(2)])
ax.set_title(zone + ' Zone Prediction v. Actual')
ax.set_xlabel('Hour')
ax.set_ylabel('Load')
yticks = ['{:,.0f} MW'.format(t) for t in ax.get_yticks()]
ax.set_yticklabels(yticks)
ax.set_xticks(range(1,24,4))
xticks = ['{:0>2}:00'.format(t) for t in ax.get_xticks()]
ax.set_xticklabels(xticks)
plt.savefig(dir_name + 'hourly_comparison.png')
print('Comparison plots saved.')

# box plots of trial error stats
ax = stats.plot(kind='box', subplots=True, figsize=[15,10], colormap=plt.cm.tab20b)
plt.suptitle(zone + ' Zone Error Distribution')
yticks0 = ['{:,.0f} MW'.format(t) for t in ax['MAE'].get_yticks()]
ax['MAE'].set_yticklabels(yticks0)
yticks1 = ['{:.2%}'.format(t) for t in ax['MAPE'].get_yticks()]
ax['MAPE'].set_yticklabels(yticks1)
plt.savefig(dir_name + 'trial_stats.png')
print('Error stats saved.')

# write network hyperparameters & stats to file
stream += ('\nMedian mean absolute error (MAE) of trials: ' +
    '{:,.2f} MW'.format(stats.median()['MAE']))
stream += ('\nMedian mean absolute percent error (MAPE) of trials: ' +
    '{:.2%}'.format(stats.median()['MAPE']))
with open(dir_name + 'network_details.txt', 'w') as file:
    file.write(stream)

print('\n')
