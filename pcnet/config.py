import sys
import re
import os
import shutil
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

from .kwargs import PCNET_INITIALIZATION_KWARGS


class Config(object):
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)

        # Data
        data = config['data']
        self.train_data_dir = data.get('train_data_dir', './')
        self.val_data_dir = data.get('val_data_dir', './')
        self.test_data_dir = data.get('test_data_dir', './')
        self.data_filename = data.get('data_filename', 'pcrnn_data.obj')

        # SETTINGS
        # Output directory
        settings = config['settings']
        self.outdir = settings.get('outdir', None)
        if self.outdir is None:
            self.outdir = settings.get('logdir', None)
        if self.outdir is None:
            self.outdir = './pcnet_model/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if not os.path.normpath(os.path.realpath(path)) == os.path.normpath(os.path.realpath(self.outdir + '/config.ini')):
            shutil.copy2(path, self.outdir + '/config.ini')

        # Process config settings
        self.model_settings = self.build_pcnet_settings(settings)
        self.model_settings['n_iter'] = settings.getint('n_iter', 1000)
        self.model_settings['use_gpu_if_available'] = settings.getboolean('use_gpu_if_available', True)

    def __getitem__(self, item):
        return self.model_settings[item]

    def build_pcnet_settings(self, settings):
        out = {}

        out['outdir'] = self.outdir
        for kwarg in PCNET_INITIALIZATION_KWARGS:
            val = kwarg.kwarg_from_config(settings)
            out[kwarg.key] = val

        return out


