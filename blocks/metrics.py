from abc import ABCMeta, abstractmethod

import theano
from theano import tensor
from six import add_metaclass

from blocks.bricks.base import application, Brick


@add_metaclass(ABCMeta)
class Cost(Brick):
    @abstractmethod
    @application
    def apply(self, y, y_hat):
        pass


@add_metaclass(ABCMeta)
class CostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (batch, features).

    """
    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        return self.cost_matrix(y, y_hat).sum(axis=1).mean()

    @abstractmethod
    @application
    def cost_matrix(self, y, y_hat):
        pass



class PearsonCorrelation(Cost):
    '''Calculates the Pearson's R correlation coefficient
    for a mini-batch.'''

    @application(outputs=["correlation"])
    def apply(self, y, y_hat):
        correlations = (y - tensor.mean(y)) * (y_hat - tensor.mean(y_hat))
        correlations /= tensor.std(y) * tensor.std(y_hat)
        return tensor.mean(correlations)



class ExplainedVariance(Cost):
    '''Calculates the percent variance of the data 
    that the model explains.'''

    @application(outputs=["explained_variance"])
    def apply(self, y, y_hat):
        meanrate = tensor.mean(y)
        mse      = tensor.mean(tensor.sqr(y - y_hat))
        rate_var = tensor.var(y)
        return 1.0 - (mse / rate_var)


class MeanModelRates(Cost):
    '''Calculates the mean model output,
    mean(y_hat)'''

    @application(outputs=["mean_model_output"])
    def apply(self, y_hat):
        return tensor.mean(y_hat)


class PoissonLogLikelihood(Cost):
    '''Calculates the negative log likelihood of data y
    given predictions y_hat, according to a Poisson model.

    Assumes that y_hat > 0.'''

    @application(outputs=["neg_log_likelihood"])
    def apply(self, y, y_hat):
        aboveZero = (y > 0)
        losses = y_hat - y * tensor.log(y_hat)
        return tensor.mean(losses)
