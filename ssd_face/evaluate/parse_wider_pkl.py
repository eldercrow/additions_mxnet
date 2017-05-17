from __future__ import print_function
import sys, os
import numpy as np
import cPickle as pickle

def parse_wider_pkl(fn_pkl, root_path):
    '''
    Parse the given pkl file to generate result .txt files.
    '''
    with open(fn_pkl, 'rb') as fh:
        detections = pickle.load(fh)
        im_paths = pickle.load(fh)

    root_res = os.path.join(root_path, 'wider_res')
    if not os.path.exists(root_res):
        os.makedirs(os.path.join(root_path, 'wider_res'))
    
    import ipdb
    ipdb.set_trace()

    for i, (dets, im_path) in enumerate(zip(detections, im_paths)):
        tok = im_path.split('/')
        path_event = os.path.join(root_res, tok[-3], tok[-2])
        if not os.path.exists(path_event):
            os.makedirs(path_event)
        res_name = tok[-1].replace('.jpg', '')
        fn_res = os.path.join(path_event, res_name + '.txt')
        
        with open(fn_res, 'w') as fh:
            fh.write(res_name + '\n')
            fh.write('{}\n'.format(len(dets)))
            for d in dets:
                dstr = '{} {} {} {} {}\n'.format(d[2], d[3], d[4]-d[2], d[5]-d[3], d[1])
                fh.write(dstr)

        if i % 100 == 99:
            print('Processed {}/{} images.'.format(i+1, len(detections)))


if __name__ == '__main__':
    #
    fn_pkl = '../wider_eval_res_lighter_epoch22.pkl'
    root_path = './spotnet_lighter_epoch22'

    parse_wider_pkl(fn_pkl, root_path)
