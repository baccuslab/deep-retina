from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from keras.callbacks import Callback

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		pickle.dump(self.losses, open(model_basename + str(num_epochs) + "_losshistory.p", "wb"))
		#fig = plt.gcf()
		#fig.set_size_inches((20,24))
		#ax = plt.subplot()
		#ax.plot(self.losses, 'b')
		#ax.plot(self.losses, 'g')
		#ax.set_title('Training loss history', fontsize=16)
		#ax.set_xlabel('Iteration', fontsize=14)
		#ax.set_ylabel('Training Loss', fontsize=14)

		#plt.tight_layout()
		#filename = '%dLoss.png' %(num_epochs)
		#plt.savefig(filename, bbox_inches='tight')
		#plt.close()

class ValLossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('val_loss'))
		pickle.dump(self.losses, open(model_basename + str(num_epochs) + "_vallosshistory.p", "wb"))

class CorrelationHistory(Callback):
	def on_train_begin(self, logs={}):
		self.train_correlations = []
		self.test_correlations = []

	def on_epoch_end(self, epoch, logs={}):
		train_subset = range(30) #np.random.choice(X_train.shape[0], 100, replace=False) #around a minute
		test_subset = range(X_test.shape[0]) #np.random.choice(X_test.shape[0], 100, replace=False)
		train_pred = self.model.predict(X_train[train_subset])
		train_pred = train_pred.squeeze()
		test_pred = self.model.predict(X_test[test_subset]) #could change this to all?
		test_pred = test_pred.squeeze()
		train_true = y_train[train_subset].squeeze()
		test_true = y_test[test_subset].squeeze()
		# store just the pearson correlation r averaged over the samples, not the p-value
		train_pred = train_pred.flatten()
		train_true = train_true.flatten()
		test_pred = test_pred.flatten()
		test_true = test_true.flatten()
		self.train_correlations.append(pearsonr(train_pred, train_true)[0])
		self.test_correlations.append(pearsonr(test_pred, test_true)[0])
		pickle.dump(self.train_correlations, open(model_basename + str(num_epochs) + "_traincorrelations.p", "wb"))
		pickle.dump(self.test_correlations, open(model_basename + str(num_epochs) + "_testcorrelations.p", "wb"))
		fig = plt.gcf()
		fig.set_size_inches((20,24))
		ax = plt.subplot()
		ax.plot(self.train_correlations, 'b')
		ax.plot(self.test_correlations, 'g')
		ax.set_title('Train and Test Pearson Correlations', fontsize=16)
		ax.set_xlabel('Iteration', fontsize=14)
		ax.set_ylabel('Correlation', fontsize=14)

		plt.tight_layout()
		filename = '%dCorrelation.png' %(num_epochs)
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

def plot_metrics(metrics, batch_id):
    # Plot progress 
    fig = plt.gcf()
    fig.set_size_inches((20,24))
    ax1 = plt.subplot(3,2,1)
    ax1.plot(metrics['train_losses'], 'k')
    ax1.set_title('Loss history', fontsize=16)
    ax1.set_xlabel('Number of batches', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)

    ax2 = plt.subplot(3,2,2)
    ax2.plot(metrics['train_correlations'], 'b')
    ax2.plot(metrics['test_correlations'], 'g')
    ax2.set_title('Train and Test Pearson Correlations', fontsize=16)
    ax2.set_xlabel('Number of batches', fontsize=14)
    ax2.set_ylabel('Correlation', fontsize=14)

    ax3 = plt.subplot(3,2,3)
    ax3.plot(metrics['train_mse'], 'b')
    ax3.plot(metrics['test_mse'], 'g')
    ax3.set_title('Train and Test Mean Squared Error', fontsize=16)
    ax3.set_xlabel('Number of batches', fontsize=14)
    ax3.set_ylabel('MSE', fontsize=14)

    # plot num_samples*0.01 seconds of predictions vs data
    num_samples = len(metrics['train_output'])
    ax4 = plt.subplot(3,2,4)
    ax4.plot(np.linspace(0, num_samples*0.01, num_samples), metrics['train_labels'], 'k', alpha=0.7)
    ax4.plot(np.linspace(0, num_samples*0.01, num_samples), metrics['train_output'], 'r', alpha=0.7)
    ax4.set_title('Training data (black) and predictions (red)', fontsize=16)
    ax4.set_xlabel('Seconds', fontsize=14)
    ax4.set_ylabel('Probability of spiking', fontsize=14)

    num_samples = len(metrics['test_output'])
    ax5 = plt.subplot(3,2,5)
    ax5.plot(np.linspace(0, num_samples*0.01, num_samples), metrics['test_labels'], 'k', alpha=0.7)
    ax5.plot(np.linspace(0, num_samples*0.01, num_samples), metrics['test_output'], 'r', alpha=0.7)
    ax5.set_title('Test data (black) and predictions (red)', fontsize=16)
    ax5.set_xlabel('Seconds', fontsize=14)
    ax5.set_ylabel('Probability of spiking', fontsize=14)

    ax6 = plt.subplot(3,2,6)
    ax6.scatter(metrics['test_labels'], metrics['test_output'])
    data_ranges = np.linspace(np.min([np.min(metrics['test_labels']), np.min(metrics['test_output'])]), 
            np.max([np.max(metrics['test_labels']), np.max(metrics['test_output'])]), 10)
    ax6.plot(data_ranges, data_ranges, 'k--')
    ax6.set_title('Test Data vs Predictions', fontsize=16)
    ax6.set_xlabel('Test Data', fontsize=14)
    ax6.set_ylabel('Test Predictions', fontsize=14)

    filename = '%dBatches.png' %(batch_id)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def training_metrics(model, X, y, batch_size, metric_size, train_inds, test_inds, metrics):
    # get contiguous size of train_inds
    flattened_train_inds = [val for sublist in train_inds for val in sublist]
    start_ind = np.random.choice(len(flattened_train_inds)-metric_size)
    train_inds = [flattened_train_inds[i:i+batch_size] for i in range(start_ind, start_ind+metric_size, batch_size)]

    # get contiguous size of test_inds
    flattened_test_inds = [val for sublist in test_inds for val in sublist]
    start_ind = np.random.choice(len(flattened_test_inds)-metric_size)
    test_inds = [flattened_test_inds[i:i+batch_size] for i in range(start_ind, start_ind+metric_size, batch_size)]

    # Get train and test data subsets
    train_output = []
    train_labels = []
    test_output = []
    test_labels = []
    for batch in train_inds:
        new_predictions = model.predict(X[batch])
        train_output.extend(new_predictions.squeeze())
        train_labels.extend(y[batch])
    for batch in test_inds:
        new_predictions = model.predict(X[batch])
        test_output.extend(new_predictions.squeeze())
        test_labels.extend(y[batch])

    # store just the pearson correlation r, not the p-value
    train_correlations = pearsonr(train_output, train_labels)[0]
    test_correlations = pearsonr(test_output, test_labels)[0]

    # store the mean square error
    train_mse = np.mean((np.array(train_output) - np.array(train_labels))**2)
    test_mse = np.mean((np.array(test_output) - np.array(test_labels))**2)

    if bool(metrics):
        metrics['train_correlations'].extend([train_correlations])
        metrics['test_correlations'].extend([test_correlations])
        metrics['train_mse'].extend([train_mse])
        metrics['test_mse'].extend([test_mse])
        metrics['train_output'].extend(train_output)
        metrics['train_labels'].extend(train_labels)
        metrics['test_output'].extend(test_output)
        metrics['test_labels'].extend(test_labels)
    else:
        metrics['train_correlations'] = [train_correlations]
        metrics['test_correlations'] = [test_correlations]
        metrics['train_mse'] = [train_mse]
        metrics['test_mse'] = [test_mse]
        metrics['train_output'] = train_output
        metrics['train_labels'] = train_labels
        metrics['test_output'] = test_output
        metrics['test_labels'] = test_labels

    return metrics
