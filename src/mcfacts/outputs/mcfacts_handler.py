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
        # Return inverse batch index
        self._inverse_batch_index = {}
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
    def inv_batch_index(self):
        '''Return the inverse of the batch index'''
        return self._inverse_batch_index

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

    def batch_samples(self, quantity, batch_id, binary_id):
        '''Return samples from batches'''
        ## Check binary_id
        # Check that binary id is a numpy array
        assert isinstance(binary_id, np.ndarray), "binary_id is not a NumPy array"
        # Check that binary_id is one dimensional
        assert len(binary_id.shape) == 1, "binary_id should be one-dimensional"
        # Make sure we can interpret as an integer
        binary_id = np.asarray(binary_id, dtype=int)
        # Get the number of systems
        nsystems = binary_id.size
        ## Check quantitiy
        # Find a batch that isn't empty
        _example_batch = None
        for _id in self._batches.keys():
            if self._batches[_id].nsystems != 0:
                _example_batch = _id
                break
        # Check that we did find a batch that isn't empty
        assert not(_example_batch is None), "All of the batches seem to be empty"
        # Check that quantity is a string
        assert isinstance(quantity, str), "quantity should be a string"
        # Check that batches have quantity
        assert hasattr(self._batches[_example_batch], quantity), "No such quantity"
        # Check that quantity is not a scalar
        assert hasattr(getattr(self._batches[_example_batch], quantity), 'size'), "invalid quantity"
        # Check units
        if hasattr(getattr(self._batches[_example_batch],quantity), 'unit'):
            unit = getattr(getattr(self._batches[_example_batch],quantity), 'unit')
        else:
            unit = None
        # Check dtype
        dtype = getattr(getattr(self._batches[_example_batch],quantity), 'dtype')
        ## Check batch_id
        batch_id = np.asarray(batch_id, dtype=int)
        
        ## Get samples for trivial case ##
        if batch_id.size == 1:
            # If there's only one batch_id, we need to draw samples from that batch
            return getattr(self._batches[self.inv_batch_index[int(batch_id)]], quantity)[binary_id]

        ## Continue checking batch id##
        assert batch_id.shape == binary_id.shape, \
            "batch_id and binary_id should be the same shape"
        # Check that all batch ids are present
        for _id in batch_id:
            # Find the actual label for the batch
            _label = self.inv_batch_index[_id]
            assert _label in self._batches
        
        # Construct output array
        q = np.empty(nsystems, dtype=dtype)
        # Construct a filled flag array for safety reasons
        filled = np.zeros(nsystems, dtype=bool)
        # Loop the unique ids
        for _id in np.unique(batch_id):
            # Find the matching ids
            mask = batch_id == _id
            # Find the actual label for the batch
            _label = self.inv_batch_index[_id]
            # Fetch the values
            if unit is None:
                q[mask] = getattr(self._batches[_label], quantity)[binary_id[mask]]
            else:
                q[mask] = getattr(self._batches[_label], quantity)[binary_id[mask]].value
            # Update the filled flag
            filled[mask] = True
        # Check that all values have been filled
        assert np.sum(filled) == nsystems, "There was an error filling the quantity array"
        # Check the units
        if not unit is None:
            q = q * unit
        return q

    def append_batch(self, label, batch):
        '''Append a new batch'''
        assert isinstance(batch, McfactsBatch)
        self._batches[label] = batch
        self._batch_index[label] = self._nbatch
        self._inverse_batch_index[self._nbatch] = label
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
    print(MH._batches.keys())
    # Check samples
    for _id, n in enumerate(MH.nsystems):
        if n == 0:
            print("batch: %d has no samples"%_id)
        else:
            M = MH.batch_samples('M', _id, np.arange(n))
            chi_eff = MH.batch_samples('chi_eff', _id, np.arange(n))
            print("Successfully sampled batch %d"%_id)
    ## Get the first sample from every batch
    # Get the indices of nonzero nsystems
    nonzero = np.asarray(np.nonzero(MH.nsystems)).flatten()
    all_mass = MH.batch_samples("M", nonzero, np.zeros(nonzero.size, dtype=int))
    all_chi_eff = MH.batch_samples("chi_eff", nonzero, np.zeros(nonzero.size, dtype=int))
    print("Successfully sampled first sample of nonzero batches")
    ## Get the first and third sample from every batch
    # Get the indices of nonzero nsystems
    few = np.asarray(np.where(MH.nsystems > 2)).flatten()
    all_mass = MH.batch_samples("M", np.repeat(few,2), np.tile([0,2],few.size))
    all_chi_eff = MH.batch_samples("chi_eff", np.repeat(few,2), np.tile([0,2],few.size))
    print("Successfully sampled first and third sample of nonzero batches")

    

######## Execution ########
if __name__ == "__main__":
    main()
