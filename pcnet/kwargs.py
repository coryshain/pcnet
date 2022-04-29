from functools import cmp_to_key

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    """

    def __init__(self, key, default_value, dtypes, descr, aliases=None):
        if aliases is None:
            aliases = []
        self.key = key
        self.default_value = default_value
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr
        self.aliases = aliases

    def dtypes_str(self):
        if len(self.dtypes) == 1:
            out = '``%s``' %self.get_type_name(self.dtypes[0])
        elif len(self.dtypes) == 2:
            out = '``%s`` or ``%s``' %(self.get_type_name(self.dtypes[0]), self.get_type_name(self.dtypes[1]))
        else:
            out = ', '.join(['``%s``' %self.get_type_name(x) for x in self.dtypes[:-1]]) + ' or ``%s``' %self.get_type_name(self.dtypes[-1])

        return out

    def get_type_name(self, x):
        if isinstance(x, type):
            return x.__name__
        if isinstance(x, str):
            return '"%s"' %x
        return str(x)

    def in_settings(self, settings):
        out = False
        if self.key in settings:
            out = True

        if not out:
            for alias in self.aliases:
                if alias in settings:
                    out = True
                    break

        return out

    def kwarg_from_config(self, settings):
        if len(self.dtypes) == 1:
            val = {
                str: settings.get,
                int: settings.getint,
                float: settings.getfloat,
                bool: settings.getboolean
            }[self.dtypes[0]](self.key, None)

            if val is None:
                for alias in self.aliases:
                    val = {
                        str: settings.get,
                        int: settings.getint,
                        float: settings.getfloat,
                        bool: settings.getboolean
                    }[self.dtypes[0]](alias, self.default_value)
                    if val is not None:
                        break

            if val is None:
                val = self.default_value

        else:
            from_settings = settings.get(self.key, None)
            if from_settings is None:
                for alias in self.aliases:
                    from_settings = settings.get(alias, None)
                    if from_settings is not None:
                        break

            if from_settings is None:
                val = self.default_value
            else:
                parsed = False
                for x in reversed(self.dtypes):
                    if x == None:
                        if from_settings == 'None':
                            val = None
                            parsed = True
                            break
                    elif isinstance(x, str):
                        if from_settings == x:
                            val = from_settings
                            parsed = True
                            break
                    else:
                        try:
                            val = x(from_settings)
                            parsed = True
                            break
                        except ValueError:
                            pass

                assert parsed, 'Invalid value "%s" received for %s' %(from_settings, self.key)

        return val

    @staticmethod
    def type_comparator(a, b):
        '''
        Types precede strings, which precede ``None``
        :param a: First element
        :param b: Second element
        :return: ``-1``, ``0``, or ``1``, depending on outcome of comparison
        '''
        if isinstance(a, type) and not isinstance(b, type):
            return -1
        elif not isinstance(a, type) and isinstance(b, type):
            return 1
        elif isinstance(a, str) and not isinstance(b, str):
            return -1
        elif isinstance(b, str) and not isinstance(a, str):
            return 1
        else:
            return 0


PCNET_INITIALIZATION_KWARGS = [

    # Global
    Kwarg(
        'outdir',
        './pcnet_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),

    # Data
    Kwarg(
        'n_features',
        50,
        int,
        "Number of spectral feature bins to use for acoustic inputs"
    ),
    Kwarg(
        'chunk_length',
        1000,
        int,
        "Number of spectral slices per training instance"
    ),

    # Optimization
    Kwarg(
        'n_epochs',
        1000,
        int,
        "Number of training epochs"
    ),
    Kwarg(
        'minibatch_size',
        32,
        int,
        "Minibatch size"
    ),
    Kwarg(
        'learning_rate',
        0.001,
        float,
        'Learning rate'
    ),
    Kwarg(
        'save_freq',
        1,
        int,
        'Save frequency (in epochs)'
    ),
    Kwarg(
        'plot_freq',
        1,
        int,
        'Plot frequency (in epochs)'
    ),
    Kwarg(
        'eval_freq',
        10,
        int,
        'Eval frequency (in epochs)'
    ),

    # Hyperparams
    Kwarg(
        'n_units',
        128,
        int,
        "Number of hidden units per layer."
    ),
    Kwarg(
        'n_layers',
        3,
        int,
        "Number of layers."
    ),
    Kwarg(
        'sparsity_regularizer_scale',
        None,
        [float, None],
        "Scale of sparsity regularizer."
    ),
    Kwarg(
        'state_regularizer_scale',
        None,
        [float, None],
        "Scale of state regularizer."
    ),
    Kwarg(
        'gate_regularizer_scale',
        None,
        [float, None],
        "Scale of gate regularizer."
    ),
    Kwarg(
        'kernel_regularizer_scale',
        None,
        [float, None],
        "Scale of kernel regularizer."
    )
]


def pcnet_kwarg_docstring():
    out = ""

    for kwarg in PCNET_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n'

    return out
