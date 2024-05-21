import numpy as np

def dump_array_to_file(fname=None, samples_out=None):
  # Write output (ascii)
  # Vastly superior data i/o with pandas, but note no # used
  import pandas
  dframe = pandas.DataFrame(samples_out)
  fname_out_ascii= fname
  dframe.to_csv(fname_out_ascii,sep=' ',header=[f"#{x}" if x == dframe.columns[0] else x for x in dframe.columns],index=False) # # is not pre-appended...just boolean


class AGNObject(object):
    """
    An array of objects. Should include all objects of this type.
    Argument arrays should be *1d scalar arrays*.
    """

    def __init__(self, orbit_a = None, mass = None, orbit_inclination= None, orbit_e = None, **kwargs):
        self.orbit_a = orbit_a #Should be array. Semimajor axis
        self.mass = mass #Should be array. TOTAL masses.
        self.orbit_inclination = orbit_inclination #Should be array. Allows for misaligned orbits.
        self.orbit_e = orbit_e #Should be array. Allows for eccentricity.
    
    def __add_objects__(self, new_object_a = None, new_object_mass = None, new_object_inclination = None, new_object_e = None):
        """
        Adds new values to the end of existing arrays
        TODO: check that all arrays are the same length.
        TODO: check that all arrays are included
        """
        self.orbit_a = np.concatenate([self.orbit_a,new_object_a])
        self.mass = np.concatenate([self.mass,new_object_mass])
        self.orbit_inclination = np.concatenate([self.orbit_inclination,new_object_inclination])
        self.orbit_e = np.concatenate([self.orbit_e,new_object_e])

    def remove_objects(self, idx_remove = None):
        """
        idx_remove should be a numpy array of indices to change, e.g., [2, 15, 23]
        as written, this loops over all attributes, not just those in the AGNObjects class,
        i.e., we don't need a separate remove_objects method for the subclasses.
        """
        if idx_remove is None:
            return None
        
        idx_change = np.ones(len(self.mass),dtype=bool)
        idx_change[idx_remove] = False
        for attr in vars(self).keys():
            setattr(self,attr,getattr(self,attr)[idx_change])

    def return_record_array(self,**kwargs):
        """
        Right now every dtype is float.
        TODO: We will probably want to change this later. Dictionary with attribute names and keys?
        Loops over all attributes, don't need to rewrite for subclasses.
        """
        #dat_out = np.array(len(self.mass), dtype = {attr:float for attr in vars(self).keys()})
        dtype = np.dtype([(attr,'float') for attr in vars(self).keys()])
        dat_out = np.empty(len(self.mass),dtype=dtype)
        print(dat_out)
        for attr in vars(self).keys():
            print(attr)
            print(dat_out[attr])
            print(getattr(self,attr))
            dat_out[attr] = getattr(self,attr)
        return(dat_out)
    
    def init_from_file(self,fname=None):
        """
        Read in previously saved data file and create AGNObject from it.
        Odd implementation right now, have to init AGNObject and only then read from file.
        TODO: Fix so that you can just read from file.
        """
        dat_in = np.genfromtxt(fname,names=True)
        for name in dat_in.dtype.names:
            setattr(self,name,dat_in[name])

#TODO: method to print values when you just run an AGNOBject class in a jupyter notebook or similar.
#TODO: similar vein: print method? Same thing maybe?
#TODO: custom error messages when you don't supply all the fields
#TODO: add spin for stars


class AGNStar(AGNObject):
    """
    An array of single stars. Should include all objects of this type. No other objects should contain objects of this type.
    """
    def __init__(self, star_radius = None, star_X = None, star_Y = None, star_Z = None, **kwargs):
       self.star_radius = star_radius
       self.star_X = star_X
       self.star_Y = star_Y
       self.star_Z = star_Z
       super(AGNStar,self).__init__(**kwargs) #calls top level functions

    def add_stars(self, new_star_radius = None, new_star_X = None, new_star_Y = None, new_star_Z = None, **kwargs):
        self.star_radius = np.concatenate([self.star_radius,new_star_radius])
        self.star_X = np.concatenate([self.star_X,new_star_X])
        self.star_Y = np.concatenate([self.star_Y,new_star_Y])
        self.star_Z = np.concatenate([self.star_Z,new_star_Z])
        super(AGNStar,self).__add_objects__(**kwargs)
