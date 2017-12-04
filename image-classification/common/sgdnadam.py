import mxnet as mx
import math
from numpy.random import uniform

register = mx.optimizer.Optimizer.register

@register
class SGDNAdam(mx.optimizer.Optimizer):
    '''
    Just for test.
    '''
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule_decay=0.004, momentum=0.9, **kwargs):
        super(SGDNAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.m_schedule = 1.
        self.momentum = momentum

    def create_state(self, index, weight):
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # momentum
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        # assert(isinstance(weight, NDArray))
        # assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
        grad += wd * weight

        w_nadam = self._update_nadam(index, weight, grad, state, lr, wd)
        w_sgd = self._update_sgd(index, weight, grad, state, lr, wd)

        if uniform(0, 1) < 1.0/3.0:
            weight[:] += w_nadam * 0.1
        else:
            weight[:] += w_sgd

    def _update_nadam(self, index, weight, grad, state, lr, wd):
        #
        t = self._index_update_count[index]

        # warming momentum schedule
        momentum_t = self.beta1 * (1. - 0.5 * (pow(0.96, t * self.schedule_decay)))
        momentum_t_1 = self.beta1 * (1. - 0.5 * (pow(0.96, (t + 1) * self.schedule_decay)))
        self.m_schedule = self.m_schedule * momentum_t
        m_schedule_next = self.m_schedule * momentum_t_1

        # update m_t and v_t
        _, m_t, v_t = state
        m_t[:] = self.beta1 * m_t + (1. - self.beta1) * grad
        v_t[:] = self.beta2 * v_t + (1. - self.beta2) * grad * grad

        grad_prime = grad / (1. - self.m_schedule)
        m_t_prime = m_t / (1. - m_schedule_next)
        v_t_prime = v_t / (1. - pow(self.beta2, t))
        m_t_bar = (1. - momentum_t) * grad_prime + momentum_t_1 * m_t_prime

        # update weight
        return -lr * m_t_bar / (mx.nd.sqrt(v_t_prime) + self.epsilon)

    def _update_sgd(self, index, weight, grad, state, lr, wd):
        #
        mom, _, _ = state
        mom[:] = (self.momentum * mom) - (lr * grad)
        return mom
