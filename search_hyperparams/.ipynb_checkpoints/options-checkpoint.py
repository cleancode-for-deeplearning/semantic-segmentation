
import argparse
from utils.network_utils import str2list, str2list_2


def options():
	p = argparse.ArgumentParser(description = 'Arguments for image2image-translation training.')
	p.add_argument('--epoch', type = str2list_2, \
							  help = 'epoch num. writing a single number will give you a normal schedule. ' \
									 'int: normal schedule, (int/int): linear decay, ' \
									 'int, int, int: CosineAnnealingLR, each cycle_num, cycle_len, cycle_mult. ' \
									 'int, int, float: CyclicLR, each cycle_num, epoch_num, div', \
									 default = '200')
	p.add_argument('--bs', type = int, help = 'batch size', default = 4)
	p.add_argument('--lr', type = float, help = 'learning rate', default = 0.003)
	p.add_argument('--beta1', type = float, help = 'beta1 parameter for the Adam optimizer', default = 0.9)
	p.add_argument('--beta2', type = float, help = 'beta2 parameter for the Adam optimizer', default = 0.99)

	p.add_argument('--ic', type = int, help = 'input channel num', default = 3)
	p.add_argument('--oc', type = int, help = 'output channel num', default = 20)
	p.add_argument('--height', type = int, help = 'image height (2^n)', default = 256)
	p.add_argument('--width', type = int, help = 'image width (2^n)', default = 256)

	p.add_argument('--backboneMode', help = '(freeze/equal/discriminative)', default = 'freeze')
	p.add_argument('--lrDivision', type = str2list, help = 'lr division value, only in discriminative mode. requires 4 values.', default = '1, 4, 16, 64')

	p.add_argument('--normType', help = 'normalization type', default = 'batchnorm')
	p.add_argument('--actType', help = 'activation type', default = 'relu')

	p.add_argument('--encType', help = 'encoder type (unet/res1/res2)', default = 'unet')
	p.add_argument('--encMode', help = 'settings in the encoder based on enc_type, only in res1/2 currently', default = 'unet')

	p.add_argument('--decType', help = 'decoder type (unet/pspnet/danet)', default = 'unet')
	p.add_argument('--decMode', help = 'settings in the decoder based on dec_type, only in UNet currently', default = 'concat')

	p.add_argument('--printFreq', type = int, help = 'prints the loss value every few iterations', default = 100)
	p.add_argument('--visFreq', type = int, help = 'saves the visualization every few iterations', default = 100)
	p.add_argument('--visPth', help = 'path to save the visualizations', default = 'visualizations/')
	p.add_argument('--modelPth', help = 'path to save the final model', default = './experiments/model.pth')

	p.add_argument('--trnSrcPth', help = 'train src dataset path', default = '/home/ubuntu/data/preprocessed/train/images')
	p.add_argument('--trnTrgPth', help = 'train trg dataset path', default = '/home/ubuntu/data/preprocessed/train/masks')
	p.add_argument('--valSrcPth', help = 'val src dataset path', default = '/home/ubuntu/data/preprocessed/val/images')
	p.add_argument('--valTrgPth', help = 'val trg dataset path', default = '/home/ubuntu/data/preprocessed/val/masks')
	p.add_argument('--ignoreIndex', type = int, help = 'ignore index when training', default = -100)

	p.add_argument('--numWorkers', type = int, help = 'num workers for the dataloader', default = 10)
	p.add_argument('--gradAcc', type = int, help = 'split the batch into n steps', default = 1)
	p.add_argument('--multigpu', action = 'store_true', help = 'use multiple gpus')

	args = p.parse_args()

	return args
'''

from easydict import EasyDict

def options():
    args = EasyDict({
        "epoch":200,
        "bs":4,
        "lr":0.003,
        "beta1":0.9,
        "beta2":0.99,
        "ic":3,
        "oc":20,
        "height":256,
        "width":256,
        "backboneMode": 'freeze',
        "lrDivision": '1,4,16,64',
        "normType": 'batchnorm',
        "actType": 'relu',
        "encType": 'unet',
        "encMode": 'unet',
        "decType": 'unet',
        "decMode": 'concat',
        "printFreq": 100,
        "visFreq": 100,
        "visPth": 'visualizations/',
        "modelPth": 'models/model.pth',
        "trnSrcPth": '/home/ubuntu/data/preprocessed/train/images', #'data/train/src',
        "trnTrgPth":'/home/ubuntu/data/preprocessed/train/masks', #'data/train/trg',
        "valSrcPth": '/home/ubuntu/data/preprocessed/val/images',#'data/val/src',
        "valTrgPth": '/home/ubuntu/data/preprocessed/val/masks',#'data/val/trg',
        "ignoreIndex": -100,
        "numWorkers": 10,
        "gradAcc": 1,
        "multigpu": 'store_true'
 
    })
    
    return args
'''
