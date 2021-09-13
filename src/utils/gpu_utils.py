import os


def config_gpu(use_cpu=False, gpu_memory=6):
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
