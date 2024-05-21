#!/usr/bin/env python
'''Handle mcfacts outputs as inputs

Mcfacts outputs are not all that big, so we're going to store them in RAM
'''
######## Imports ########
import numpy as np
from os.path import isfile, isdir, split, join, abspath, basename
import os
from astropy import units as u
#### Mcfacts ####
from mcfacts.outputs.mcfacts_batch_object import McfactsBatch
from mcfacts.inputs.ReadInputs import INPUT_TYPES


######## Objects ########

class McfactsHandler(object):
    '''Object to handle many batches of mcfacts objects'''

    def __init__(
                 self,
                ):
        '''Initialize mcfacts handler'''
        # Initialize empty batches array
        self._batches = {}
        # Return batch index
        self._batch_index = {}
        # Initialize nbatch
        self._nbatch = 0

    #### Properties ####
    @property
    def batches(self):
        '''Return batch labels'''
        return list(self._batches.keys())

    @property
    def batch_index(self):
        '''Return index of each batch'''
        #np.asarray(list(self._batch_index.keys()))
        return self._batch_index

    @property
    def nbatch(self):
        '''number of batches'''
        return self._nbatch

    @property
    def nsystems(self):
        '''Return the number of BBH mergers for each batch'''
        return self.batch_param('nsystems')

    @property
    def avg_systems(self):
        '''The average number of BBH mergers for each iteration'''
        return self.nsystems / self.n_iterations

    @property
    def n_iterations(self):
        '''The number of iterations for each batch'''
        return self.batch_param('n_iterations')

    @property
    def mass_smbh(self):
        '''Return the SMBH mass of each batch as an array'''
        return self.batch_param('mass_smbh')

    @property
    def common(self):
        '''Return the common attributes of different batches'''
        common_list = []
        for attr in INPUT_TYPES:
            attr_arr = self.batch_param(attr)
            unique = np.unique(attr_arr)
            if len(unique) == 1:
                common_list.append(attr)
        return common_list

    #### Methods ####
    def batch_param_single(self, batch, key):
        '''Get batch parameter'''
        return getattr(self._batches[batch], key)

    def batch_param(self, key):
        '''Get a parameter of each batch'''
        # Initialize quantity array
        quantity = np.zeros(self.nbatch, dtype=object)
        # Loop the batches
        for batch in self.batches:
            _q = self.batch_param_single(batch, key)
            if hasattr(_q, 'unit'):
                q_unit = _q.unit
                q_value = _q.value
            else:
                q_value = _q
            quantity[self.batch_index[batch]] = q_value
        # Change type of quantity
        if key in INPUT_TYPES:
            quantity = np.asarray(quantity, dtype=INPUT_TYPES[key])
        else:
            try:
                quantity = np.asarray(quantity, dtype=int)
            except:
                try:
                    quantity = np.asarray(quantity, dtype=float)
                except:
                    pass
        # Check units
        if hasattr(_q, 'unit'):
            quantity = quantity * q_unit
        # Return quantity
        return quantity

    def append_batch(self, label, batch):
        '''Append a new batch'''
        assert isinstance(batch, McfactsBatch)
        self._batches[label] = batch
        self._batch_index[label] = self._nbatch
        self._nbatch += 1
        assert self._batches.keys() == self._batch_index.keys()

    @staticmethod
    def from_runs(run_directory, **kwargs):
        '''Generate a mcfacts_batch object from runs'''
        # Check that we are looking at a valid directory
        assert isdir(run_directory)
        # Get subdirectories in directory
        subdirs = os.listdir(run_directory)
        subdirs.sort()
        # Identify output*.dat
        batch_dir = []
        # Initialize McFacts Handler
        MH = McfactsHandler()
        for _sub in subdirs:
            # identify path to sub directory
            sub = join(run_directory, _sub)
            # Skip files
            if not isdir(sub):
                continue
            # get list of files
            _files = os.listdir(sub)
            # Initialize filenames
            _fname_out = None
            _fname_log = None
            # Loop
            for _file in _files:
                # Check for output*.dat
                if (_file.startswith('output') and _file.endswith('.dat')):
                    _fname_out = abspath(join(sub, _file))
                    assert isfile(_fname_out)
                # Check for out*.log
                if (_file.startswith('out') and _file.endswith('.log')):
                    _fname_log = abspath(join(sub, _file))
                    assert isfile(_fname_log)
            # Check that we found output*.dat
            if (_fname_out is None):
                continue
            # Check that we found out*.log
            if (_fname_log is None):
                continue
            # Generate the batch
            batch = McfactsBatch.from_runs(sub)
            # Append batch
            MH.append_batch(_sub, batch)
        return MH 

                
######## Args for testing ########
def arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory', required=True, type=str,
        help="Run directory")
    opts = parser.parse_args()
    return opts

######## Main for testing ########
def main():
    opts = arg()
    MH = McfactsHandler.from_runs(opts.run_directory)

    print(basename(opts.run_directory))
    print("common attributes:", MH.common)
    print("batches:", MH.nbatch)
    print("labels:", MH.batches)
    print("mass_smbh:", MH.mass_smbh)
    print("mass_nsc:", MH.batch_param('M_nsc'))
    print("BBH mergers:",MH.nsystems)
    print("Avg BBH mergers:",MH.avg_systems)

######## Execution ########
if __name__ == "__main__":
    main()
