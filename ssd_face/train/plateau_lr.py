import mxnet as mx
import logging

class PlateauScheduler(object):
    '''
    A learning rate scheduler that implements 'plateau' scheduling algorithm.
    '''

    def __init__(self, patient_epochs=(2, 2, 2, 2, 3, 3), factor=0.316228):
        #
        self.patient_epochs = patient_epochs
        self.factor = factor

    def reset(self, base_lr):
        self.curr_lr = base_lr
        self.curr_min_loss = -1.0 # FIXME: we assume loss is always bigger than 0,

        self.curr_eidx = 0
        self.curr_e_interval = 0

    def update_lr(self, eval_metric):
        #
        is_good = True

        if self.curr_min_loss < 0.0:
            logging.info("Setting the initial loss.")
            # print 'initializing min evaluation metric losses...'
            # if len(self.eval_weights) == 0:
            #     for name in eval_metric:
            #         self.eval_weights[name] = 1.0
            self.curr_min_loss = 0.0
            for name, val in eval_metric.get_name_value():
                self.curr_min_loss += val # * self.eval_weights[name]
        else:
            sum_loss = 0.0
            for name, val in eval_metric.get_name_value():
                sum_loss += val # * self.eval_weights[name]

            if sum_loss < self.curr_min_loss:
                self.curr_min_loss = sum_loss
                self.curr_eidx = 0
            else:
                self.curr_eidx += 1
                is_good = False
            #     is_all_good = True
            # else:
            #     is_all_good = False
            #
            # if is_all_good == False:
            #     self.curr_eidx += 1
            # else:
            #     self.curr_eidx = 0
            
            if self.curr_eidx == self.patient_epochs[self.curr_e_interval]:
                # print 'updating lr, from %.5f to %.5f.' % (self.curr_lr, self.curr_lr * self.gamma)
                self.curr_lr *= self.factor
                self.curr_eidx = 0
                self.curr_e_interval += 1

        if self.curr_e_interval == len(self.patient_epochs):
            self.curr_lr = 0.0

        return self.curr_lr, is_good
