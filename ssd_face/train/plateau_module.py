import mxnet as mx
import time, logging

from plateau_lr import PlateauScheduler
from mxnet.module import *
from mxnet.module.base_module import _as_list
from mxnet.model import BatchEndParam

class PlateauModule(Module):
    '''
    Module for plateau lr schedule.
    '''

    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=mx.context.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None):
        super(PlateauModule, self).__init__(symbol, data_names, label_names, 
                logger, context, work_load_list, fixed_param_names, state_names)

    def fit(self, train_data, 
            plateau_lr, plateau_metric=None, fn_curr_model=None, plateau_backtrace=True, 
            eval_data=None, eval_metric='acc', 
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=mx.init.Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None):
        ''' 
        overrides fit() in base_module.
        '''
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        # we will use plateau lr scheduler.
        self.logger.info('Initial base lr = %.5f', self._optimizer.lr)
        plateau_lr.reset(self._optimizer.lr)
        self._optimizer.lr_scheduler = None
        if plateau_metric is None:
            plateau_metric = eval_metric

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            if plateau_metric is not eval_metric:
                plateau_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True

                self.update_metric(eval_metric, data_batch.label)
                if plateau_metric is not eval_metric:
                    self.update_metric(plateau_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # update lr using plateau algorithm
            new_lr, is_good = plateau_lr.update_lr(plateau_metric)
            if is_good and fn_curr_model is not None:
                super(PlateauModule, self).save_params(fn_curr_model)
            if new_lr == 0.0:
                self.logger.info('Minimum LR reached. Terminate training.')
                train_data.reset()
                break
            if new_lr < self._optimizer.lr:
                self.logger.info('Update lr, from %.6f to %.6f', self._optimizer.lr, new_lr)
                self._optimizer.lr = new_lr
                if fn_curr_model is not None and plateau_backtrace:
                    self.logger.info('Reset network parameters to the previous best result.')
                    super(PlateauModule, self).load_params(fn_curr_model)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()
