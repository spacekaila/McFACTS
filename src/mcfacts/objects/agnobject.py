import numpy as np
from copy import deepcopy
from astropy.constants import G 
from mcfacts.setup import setupdiskstars
from mcfacts.setup import setupdiskblackholes
from mcfacts.mcfacts_random_state import rng

def dump_array_to_file(fname=None, samples_out=None):
  # Write output (ascii)
  # Vastly superior data i/o with pandas, but note no # used
  import pandas
  dframe = pandas.DataFrame(samples_out)
  fname_out_ascii= fname
  dframe.to_csv(fname_out_ascii,sep=' ',header=[f"#{x}" if x == dframe.columns[0] else x for x in dframe.columns],index=False) # # is not pre-appended...just boolean

""" 
def check_1d_length(arr_list):
    if(len({arr.shape for arr in arr_list}) > 1):
        raise ValueError('Arrays are not all 1d.')

    if( len(set(map(len,arr_list))) != 1):
        raise ValueError('Arrays are not all the same length.')
 """    

#TODO: method to print values when you just run an AGNOBject class in a jupyter notebook or similar.
#TODO: similar vein: print method? Same thing maybe?
#TODO: custom error messages when you don't supply all the fields for init and add methods
#TODO: custom error messages when input arrays to init and add methods aren't the same length
#TODO: dump_record_array writes every value as a float. Make dictionary with all attributes and datatypes? Or is a float fine?
#TODO: init_from_file: you have to initialize an empty AGNObject before you can init_from_file and that seems weird.
       #check if this is now just an AGNObject because we would want it to be an AGNStar or AGNBlackHole etc
#TODO: error if you try to pass kwarg that doesnt exist. AttributeError?
#TODO: issue: right now you can pass AGNStar arrays that are 8-elements long for the star parameters and 10-elements long 
       #for the AGNObject parameters and it doesn't complain.
#TODO: dump_array_to_file: can I just import DataFrame and to_csv so it's lower overhead?

class AGNObject(object):
    """
    An array of objects. Should include all objects of this type.
    Argument arrays should be *1d scalar arrays*.
    Everything here is for the center-of-mass around the SMBH
    """

    def __init__(self,  mass = None,
                        spin = None, #internal quantity. total J for a binary
                        spin_angle = None, #angle between J and orbit around SMBH for binary
                        orb_a = None, #location
                        orb_inc= None, #of CoM for binary around SMBH
                        #orb_ang_mom = None,  # redundant, should be computed from keplerian orbit formula for L in terms of mass, a, eccentricity
                        orb_ecc = None,
                        orb_arg_periapse = None,
                        smbh_mass = None,
                        obj_num = None,
                        id_start_val = None):
        
        #Make sure all inputs are included
        """ if mass is None: raise AttributeError("mass is not included in inputs")
        if spin is None: raise AttributeError('spin is not included in inputs')
        if spin_angle is None: raise AttributeError('spin_angle is not included in inputs')
        if orb_a is None: raise AttributeError('orb_a is not included in inputs')
        if orb_inc is None: raise AttributeError('orb_inc is not included in inputs')
#        if orb_ang_mom is None: raise AttributeError('orb_ang_mom is not included in inputs')
        if orb_ecc is None: raise AttributeError('orb_ecc is not included in inputs') """

        """

        assert mass.shape == (obj_num,),"mass: all arrays must be 1d and the same length"
        assert spin.shape == (obj_num,),"spin: all arrays must be 1d and the same length"
        assert spin_angle.shape == (obj_num,),"spin_angle: all arrays must be 1d and the same length"
        assert orb_a.shape == (obj_num,),"orb_a: all arrays must be 1d and the same length"
        assert orb_inc.shape == (obj_num,),"orb_inc: all arrays must be 1d and the same length"
#        assert orb_ang_mom.shape == (obj_num,),"orb_ang_mom: all arrays must be 1d and the same length"
        assert orb_ecc.shape == (obj_num,),"orb_ecc: all arrays must be 1d and the same length" """
        
        if mass is None:
            #creating an empty object
            #i know this is a terrible way to do things
            self.gen = None
            self.id_num = None
        else:
            if obj_num is None: obj_num = mass.size
            self.gen = np.full(obj_num,1)
            if id_start_val is None:
                self.id_num = np.arange(0,len(mass)) #creates ID numbers sequentially from 0
            else: #if we have an id_start_val aka these aren't the first objects in the disk
                self.id_num = np.arange(id_start_val, id_start_val + len(mass), 1)

        self.mass = mass #Should be array. TOTAL masses.
        self.spin = spin #Should be array
        self.spin_angle = spin_angle #should be array
        self.orb_a = orb_a #Should be array. Semimajor axis
        self.orb_inc = orb_inc #Should be array. Allows for misaligned orbits.
        #self.orb_ang_mom = orb_ang_mom #needs to be added in!
        self.orb_ecc = orb_ecc #Should be array. Allows for eccentricity.
        #self.__smbh_mass = smbh_mass
        self.orb_arg_periapse = orb_arg_periapse



    def add_objects(self, new_mass = None,
                              new_spin = None,
                              new_spin_angle = None,
                              new_orb_a = None,
                              new_orb_inc = None,
                              new_orb_ang_mom = None,
                              new_orb_ecc = None,
                              new_orb_arg_periapse = None,
                              new_gen = None,
                              new_id_num = None,
                              obj_num = None):
        """
        Adds new values to the end of existing arrays
        """

        #Make sure all inputs are included
        """ if new_mass is None: raise AttributeError('new_mass is not included in inputs')
        if new_spin is None: raise AttributeError('new_spin is not included in inputs')
        if new_spin_angle is None: raise AttributeError('new_spin_angle is not included in inputs')
        if new_a is None: raise AttributeError('new_a is not included in inputs')
        if new_inc is None: raise AttributeError('new_inc is not included in inputs')
#        if new_orb_ang_mom is None: raise AttributeError('new_orb_ang_mom is not included in inputs')
        if new_e is None: raise AttributeError('new_e is not included in inputs')

        if obj_num is None: obj_num = new_mass.size

        assert new_mass.shape == (obj_num,),"new mass: all arrays must be 1d and the same length"
        assert new_spin.shape == (obj_num,),"new_spin: all arrays must be 1d and the same length"
        assert new_spin_angle.shape == (obj_num,),"new_spin_angle: all arrays must be 1d and the same length"
        assert new_a.shape == (obj_num,),"new_a: all arrays must be 1d and the same length"
        assert new_inc.shape == (obj_num,),"new_inc: all arrays must be 1d and the same length"
#        assert new_orb_ang_mom.shape == (obj_num,),"new_orb_ang_mom: all arrays must be 1d and the same length"
        assert new_e.shape == (obj_num,),"new_e: all arrays must be 1d and the same length" """

        #new_M = new_mass + smbh_mass
        #new_M_reduced = new_mass*smbh_mass/new_M
        #new_orb_ang_mom = new_M_reduced*np.sqrt(G.to('m^3/(M_sun s^2)').value*new_M*new_a*(1-new_inc**2))
        #self.new_orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(
        #                                                               star_num=obj_num,
        #                                                               M_reduced=new_M_reduced,
        #                                                               M=new_M,
        #                                                               orb_a = new_a,
        #                                                               orb_inc=new_inc)


        self.mass = np.concatenate([self.mass,new_mass])
        self.spin = np.concatenate([self.spin,new_spin])
        self.spin_angle = np.concatenate([self.spin_angle,new_spin_angle])
        self.orb_a = np.concatenate([self.orb_a,new_orb_a])
        self.orb_ang_mom = np.concatenate([self.orb_ang_mom, new_orb_ang_mom])
        self.orb_inc = np.concatenate([self.orb_inc,new_orb_inc])
        self.orb_ecc = np.concatenate([self.orb_ecc,new_orb_ecc])
        self.orb_arg_periapse = np.concatenate([self.orb_arg_periapse,new_orb_arg_periapse])
        self.gen = np.concatenate([self.gen,new_gen])
        self.id_num = np.concatenate([self.id_num, new_id_num])

    def remove_objects(self, idx_remove = None):
        """
        Removes objects at specified indices.

        Parameters
        ----------
        idx_remove : numpy array
            Indices to remove

        Returns
        -------
        ???
        idx_remove should be a numpy array of indices to change, e.g., [2, 15, 23]
        as written, this loops over all attributes, not just those in the AGNObjects class,
        i.e., we don't need a separate remove_objects method for the subclasses.
        """

        #Check that the index array is a numpy array.
        #assert isinstance(idx_remove,np.ndarray),"idx_remove must be numpy array"

        if idx_remove is None:
            return None
        
        idx_change = np.ones(len(self.mass),dtype=bool)
        idx_change[idx_remove] = False
        for attr in vars(self).keys():
            setattr(self,attr,getattr(self,attr)[idx_change])

    
    def keep_objects(self, idx_keep = None):
        """
        Keeps objects at specified indices. E.g., a filter function.

        Parameters
        ----------
        keep_objects : numpy array
            Indices to keep, others are removed.

        Returns
        -------
        ???
        idx_keep should be a numpy array of indices to keep, e.g., [2, 15, 23]
        """

        #Check that the index array is a numpy array.
        #assert isinstance(idx_remove,np.ndarray),"idx_remove must be numpy array"

        if idx_keep is None:
            return None
        
        idx_change = np.zeros(len(self.mass),dtype=bool)
        idx_change[idx_keep] = True
        for attr in vars(self).keys():
            setattr(self,attr,getattr(self,attr)[idx_change])


    def copy(self):
        copied_object = deepcopy(self)
        return(copied_object)

    def locate(self, idx=None):
        """
        Returns objects at specified indices
        """

        #Check that index array is numpy array
        assert isinstance(idx,np.ndarray),"idx must be numpy array"

        if idx is None:
            return None
        
        idx_full = np.zeros(len(self.mass),dtype=bool)
        idx_full[idx] = True

    def sort(self, sort_attr=None):
        #Takes in one attribute array and sorts the whole class by that array
        #Sorted array indices
        sort_idx = np.argsort(sort_attr)

        #Now apply these to each of the attributes
        for attr in vars(self).keys():
            setattr(self, attr, getattr(self, attr)[sort_idx])

    def return_params(self):
        """
        Gets list of parameters present in object.

        Parameters
        ----------
        None

        Returns
        -------
        list
            parameters in object
        """
        return(list(vars(self).keys()))


    def return_record_array(self,**kwargs):
        """
        Right now every dtype is float.
        Loops over all attributes, don't need to rewrite for subclasses.
        """
        dtype = np.dtype([(attr,'float') for attr in vars(self).keys()])
        dat_out = np.empty(len(self.mass),dtype=dtype)
        for attr in vars(self).keys():
            dat_out[attr] = getattr(self, attr)
        return(dat_out)
    
    def to_file(self,fname=None):
        # Write output (ascii)
        # Vastly superior data i/o with pandas, but note no # used
        import pandas
        samples_out = self.return_record_array()
        dframe = pandas.DataFrame(samples_out)
        fname_out_ascii= fname
        dframe.to_csv(fname_out_ascii,sep=' ',header=[f"#{x}" if x == dframe.columns[0] else x for x in dframe.columns],index=False) # # is not pre-appended...just boolean

    
    def init_from_file(self,fname=None):
        """
        Read in previously saved data file and create AGNObject from it.
        Odd implementation right now, have to init AGNObject and only then read from file.
        """
        dat_in = np.genfromtxt(fname,names=True)
        for name in dat_in.dtype.names:
            setattr(self,name,dat_in[name])


class AGNStar(AGNObject):
    """
    An array of single stars. Should include all objects of this type. No other objects should contain objects of this type.
    Takes in initial helium and metals mass fractions, calculates hydrogen mass fraction from that to ensure it adds up to unity.
    """
    def __init__(self, mass = None, star_radius = None, star_Y = None, star_Z = None, star_num = None, smbh_mass = None, orb_a = None, orb_inc= None, **kwargs):
        """
        Array of single star objects.
        star_radius should be a 1d numpy array
        star_Y and star_Z can be floats or 1d numpy arrays, but must sum to 1 or less.
        """
        #Make sure all inputs are included
        #if star_radius is None: raise AttributeError('star_radius is not included in inputs')
        """ if star_Y is None: raise AttributeError('star_Y is not included in inputs')
        if star_Z is None: raise AttributeError('star_Z is not included in inputs') """

        if star_num is None: star_num = star_radius.size

        #Make sure inputs are numpy arrays
        if star_radius is None:
            self.star_radius = setupdiskstars.setup_disk_stars_radii(masses=mass)

        else:
            assert isinstance(star_radius,np.ndarray),"star_radius is not a numpy array"
            self.star_radius = star_radius

        if (np.any(star_Y + star_Z > 1.)):
            raise ValueError("star_Y and star_Z must sum to 1 or less.")

        if ((isinstance(star_Y,float) and (isinstance(star_Z,float)))):
            self.star_X = np.ones(len(star_radius)) - np.full(len(star_radius),star_Y) - np.full(len(star_radius),star_Z)
            self.star_Y = np.full(len(star_radius),star_Y)
            self.star_Z = np.full(len(star_radius),star_Z)
        
        elif ((isinstance(star_Y,np.ndarray) and isinstance(star_Z,np.ndarray))):
            assert star_radius.shape == (star_num,),"star_radius: all arrays must be 1d and the same length"
            assert star_Y.shape == (star_num,),"star_Y, array: all arrays must be 1d and the same length"
            assert star_Z.shape == (star_num,),"star_Z, array: all arrays must be 1d and the same length"

            self.star_X = 1. - star_Y - star_Z
            self.star_Y = star_Y
            self.star_Z = star_Z

        else:
            raise TypeError("star_Y and star_Z must be either both floats or numpy arrays")
        
        M = mass + smbh_mass
        M_reduced = mass*smbh_mass/M
        #self.orb_ang_mom = M_reduced*np.sqrt(G.to('m^3/(M_sun s^2)').value*M*self.orb_a*(1-self.orb_inc**2))
        self.orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(star_num=star_num,
                                                                       M_reduced=M_reduced,
                                                                       M=M,
                                                                       orb_a = orb_a,
                                                                       orb_inc=orb_inc)



        super(AGNStar,self).__init__(mass = mass,orb_a = orb_a, orb_inc=orb_inc, obj_num = star_num, **kwargs) #calls top level functions
    
    def __repr__(self):
        return('AGNStar(): {} single stars'.format(len(self.mass)))

    def add_stars(self, new_radius = None, new_Y = None, new_Z = None, obj_num = None, **kwargs):
        """
        Add new star values to the end of existing arrays.
        """

        #Make sure all inputs are included
        if new_radius is None: raise AttributeError("new_radius is not included in inputs")
        if new_Y is None: raise AttributeError("new_Y is not included in inputs")
        if new_Z is None: raise AttributeError("new_Z is not included in inputs")

        if obj_num is None: obj_num = new_radius.size

        assert new_radius.shape == (obj_num,),"new_radius: all arrays must be 1d and the same length"
        self.star_radius = np.concatenate([self.star_radius,new_radius])

        if(np.any(new_Y + new_Z) > 1.): raise ValueError("new_Y and new_Z must sum to 1 or less")

        if( (isinstance(new_Y, float)) and (isinstance(new_Z, float))):
            self.star_X = np.concatenate(self.star_X, np.full(obj_num,1.-new_Y-new_Z))
            self.star_Y = np.concatenate([self.star_Y, np.full(obj_num,new_Y)])
            self.star_Z = np.concatenate([self.star_Z, np.full(obj_num,new_Z)])
            
        if( (isinstance(new_Y, np.ndarray)) and (isinstance(new_Z, np.ndarray))):
            self.star_X = np.concatenate([self.star_X, np.ones(obj_num) - new_Y - new_Z])
            self.star_Y = np.concatenate([self.star_Y, new_Y])
            self.star_Z = np.concatenate([self.star_Z, new_Z])
        super(AGNStar,self).add_objects(obj_num = obj_num, **kwargs)


class AGNBlackHole(AGNObject):
    """
    An array of single black holes. Should include all objects of this type. No other objects should contain objects of this type.
    """
    def __init__(self, mass = None, **kwargs):

        self.orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(n_bh = len(mass))
        
        super(AGNBlackHole,self).__init__(mass = mass, **kwargs) #Calls top level functions
    
    def __repr__(self):
        return('AGNBlackHole(): {} single black holes'.format(len(self.mass)))


    def add_blackholes(self, obj_num = None, **kwargs):
        super(AGNBlackHole,self).add_objects(obj_num = obj_num, **kwargs)


class AGNBinaryStar(AGNObject):
    """
    An array of binary stars. Should include all objects of this type. No other objects should include objects of this type.
    Properties of this class:
        * scalar properties of each star:  masses, radius, Y,Z (star_mass1,star_mass2, star_radius1, star_radius2, ...)
         * vector properties of each star: individual angular momentum vectors (spin1,spin_angle_1) *relative to the z axis of the binary orbit*,
           not SMBH
         * orbit properties: use a reduced mass hamiltonian, you have 'r = r2 - r1' (vector) for which we get bin_a, bin_e, bin_inc (relative to SMBH)

    """

    def __init__(self, star_mass1 = None,
                       star_mass2 = None,
                       star_radius1 = None,
                       star_radius2 = None,
                       star_Y1 = None,
                       star_Y2 = None,
                       star_Z1 = None,
                       star_Z2 = None,
                       star_spin1 = None,
                       star_spin2 = None,
                       star_spin_angle1 = None,
                       star_spin_angle2 = None,
                       bin_e = None,
                       bin_a = None,
                       bin_inc=None,
                       cm_orb_a=None,
                       cm_orb_inc=None,
                       cm_orb_ecc=None,
                       obj_num = None,
                     **kwargs):
        
        #Make sure all inputs are included
        if star_mass1 is None: raise AttributeError("star_mass1 is not included in inputs")
        if star_mass2 is None: raise AttributeError("star_mass2 is not included in inputs")
        if star_radius1 is None: raise AttributeError("star_radius1 is not included in inputs")
        if star_radius2 is None: raise AttributeError("star_radius2 is not included in inputs")
        if star_Y1 is None: raise AttributeError("star_Y1 is not included in inputs")
        if star_Y2 is None: raise AttributeError("star_Y2 is not included in inputs")
        if star_Z1 is None: raise AttributeError("star_Z1 is not included in inputs")
        if star_Z2 is None: raise AttributeError("star_Z2 is not included in inputs")
        if star_spin1 is None: raise AttributeError("star_spin1 is not included in inputs")
        if star_spin2 is None: raise AttributeError("star_spin2 is not included in inputs")
        if star_spin_angle1 is None: raise AttributeError("star_spin_angle1 is not included in inputs")
        if star_spin_angle2 is None: raise AttributeError("star_spin_angle2 is not included in inputs")
        if bin_e is None: raise AttributeError("bin_e is not included in inputs")
        if bin_a is None: raise AttributeError("bin_a is not included in inputs")

        if obj_num is None: obj_num = star_mass1.size

        #Check that all inputs are 1d numpy arrays
        assert star_mass1.shape == (obj_num,),"star_mass1: all arrays must be 1d and the same length"
        assert star_mass2.shape == (obj_num,),"star_mass2: all arrays must be 1d and the same length"
        assert star_radius1.shape == (obj_num,),"star_radius1: all arrays must be 1d and the same length"
        assert star_radius2.shape == (obj_num,),"star_radius2: all arrays must be 1d and the same length"
        assert star_spin1.shape == (obj_num,),"star_spin1: all arrays must be 1d and the same length"
        assert star_spin2.shape == (obj_num,),"star_spin2: all arrays must be 1d and the same length"
        assert star_spin_angle1.shape == (obj_num,),"star_spin_angle1: all arrays must be 1d and the same length"
        assert star_spin_angle2.shape == (obj_num,),"star_spin_angle2: all arrays must be 1d and the same length"
        assert bin_e.shape == (obj_num,),"bin_e: all arrays must be 1d and the same length"
        assert bin_a.shape == (obj_num,),"bin_a: all arrays must be 1d and the same length"

        if (np.any(star_Y1 + star_Z1 > 1.)):
            raise ValueError("star_Y1 and star_Z1 must sum to 1 or less.")
        if (np.any(star_Y2 + star_Z2 > 1.)):
            raise ValueError("star_Y2 and star_Z2 must sum to 1 or less.")

        if((isinstance(star_Y1,float)) and (isinstance(star_Z1,float))):
            star_X1 = np.full(obj_num,1. - star_Y1 - star_Z1)
            star_Y1 = np.full(obj_num,star_Y1)
            star_Z1 = np.full(obj_num,star_Z1)
        if((isinstance(star_Y2,float)) and (isinstance(star_Z2,float))):
            star_X2 = np.full(obj_num,1. - star_Y2 - star_Z2)
            star_Y2 = np.full(obj_num,star_Y2)
            star_Z2 = np.full(obj_num,star_Z2)
        if((isinstance(star_Y1,np.ndarray)) and (isinstance(star_Z1,np.ndarray))):
            assert star_Y1.shape == (obj_num,),"star_Y1: all arrays must be 1d and the same length"
            assert star_Z1.shape == (obj_num,),"star_Z1: all arrays must be 1d and the same length"
            star_X1 = np.ones(obj_num) - star_Y1 - star_Z1
        if((isinstance(star_Y2,np.ndarray)) and (isinstance(star_Z2,np.ndarray))):
            assert star_Y2.shape == (obj_num,),"star_Y2: all arrays must be the same length"
            assert star_Z2.shape == (obj_num,),"star_Z2: all arrays must be the same length"
            star_X2 = np.ones(obj_num) - star_Y2 - star_Z2
        else: raise TypeError("star_Y1, star_Z1 and star_Y2, star_Z2 must be either both floats or numpy arrays")


        #Now assign attributes
        self.star_mass1 = star_mass1
        self.star_mass2 = star_mass2
        self.star_radius1 = star_radius1
        self.star_radius2 = star_radius2
        self.star_X1 = star_X1
        self.star_X2 = star_X2
        self.star_Y1 = star_Y1
        self.star_Y2 = star_Y2
        self.star_Z1 = star_Z1
        self.star_Z2 = star_Z2
        self.star_spin1 = star_spin1
        self.star_spin2 = star_spin2
        self.star_spin_angle1 = star_spin_angle1
        self.star_spin_angle2 = star_spin_angle2
        self.bin_e = bin_e
        self.bin_a = bin_a
        self.bin_inc = bin_inc

        #Now calculate properties for the AGNObject class aka the totals
        total_mass = star_mass1 + star_mass2
        total_spin = None # we will pass None for now, until we decide how to treat angular momentum
        total_spin_angle = None  # we will pass None for now, until we decide how to treat angular momentum

        super(AGNBinaryStar,self).__init__(mass = total_mass, spin = star_spin_angle2, spin_angle = star_spin_angle2, orb_a = cm_orb_a, orb_inc = cm_orb_inc, orb_ecc = cm_orb_ecc, obj_num = obj_num)
        

    def __repr__(self):
        return('AGNBinaryStar(): {} stellar binaries'.format(len(self.mass)))
    
    def add_binaries(self, new_star_mass1 = None,
                        new_star_radius1 = None,
                        new_star_mass2 = None,
                        new_star_radius2 = None,
                        new_star_Y1 = None,
                        new_star_Y2 = None,
                        new_star_Z1 = None,
                        new_star_Z2 = None,
                        new_star_spin1 = None,
                        new_star_spin2 = None,
                        new_star_spin_angle1 = None,
                        new_star_spin_angle2 = None,
                        new_star_orb_a1 = None,
                        new_star_orb_a2 = None,
                        new_bin_e = None,
                        new_bin_a = None,
                        new_bin_inc=None,
                        new_cm_orb_a=None,
                        new_cm_orb_inc=None,
                        new_cm_orb_ecc=None,
                        obj_num = None,
                    **kwargs):
        

        #Make sure all inputs are included
        if new_star_mass1 is None: raise AttributeError("new_star_mass1 is not included in inputs")
        if new_star_mass2 is None: raise AttributeError("new_star_mass2 is not included in inputs")
        if new_star_radius1 is None: raise AttributeError("new_star_radius1 is not included in inputs")
        if new_star_radius2 is None: raise AttributeError("new_star_radius2 is not included in inputs")
        if new_star_Y1 is None: raise AttributeError("new_star_Y1 is not included in inputs")
        if new_star_Y2 is None: raise AttributeError("new_star_Y2 is not included in inputs")
        if new_star_Z1 is None: raise AttributeError("new_star_Z1 is not included in inputs")
        if new_star_Z2 is None: raise AttributeError("new_star_Z2 is not included in inputs")
        if new_star_spin1 is None: raise AttributeError("new_star_spin1 is not included in inputs")
        if new_star_spin2 is None: raise AttributeError("new_star_spin2 is not included in inputs")
        if new_star_spin_angle1 is None: raise AttributeError("new_star_spin_angle1 is not included in inputs")
        if new_star_spin_angle2 is None: raise AttributeError("new_star_spin_angle2 is not included in inputs")
        if new_bin_e is None: raise AttributeError("new_bin_e is not included in inputs")
        if new_bin_a is None: raise AttributeError("new_bin_a is not included in inputs")

        if obj_num is None: obj_num = new_star_mass1.size

        #Check that all inputs are 1d numpy arrays
        assert new_star_mass1.shape == (obj_num,),"new_star_mass1: all arrays must be 1d and the same length"
        assert new_star_mass2.shape == (obj_num,),"new_star_mass2: all arrays must be 1d and the same length"
        assert new_star_radius1.shape == (obj_num,),"new_star_radius1: all arrays must be 1d and the same length"
        assert new_star_radius2.shape == (obj_num,),"new_star_radius2: all arrays must be 1d and the same length"
        assert new_star_spin1.shape == (obj_num,),"new_star_spin1: all arrays must be 1d and the same length"
        assert new_star_spin2.shape == (obj_num,),"new_star_spin2: all arrays must be 1d and the same length"
        assert new_star_spin_angle1.shape == (obj_num,),"new_star_spin_angle1: all arrays must be 1d and the same length"
        assert new_star_spin_angle2.shape == (obj_num,),"new_star_spin_angle2: all arrays must be 1d and the same length"
        assert new_bin_e.shape == (obj_num,),"new_bin_e: all arrays must be 1d and the same length"
        assert new_bin_a.shape == (obj_num,),"new_bin_a: all arrays must be 1d and the same length"

        if (np.any(new_star_Y1 + new_star_Z1 > 1.)):
            raise ValueError("new_star_Y1 and new_star_Z1 must sum to 1 or less.")
        if (np.any(new_star_Y2 + new_star_Z2 > 1.)):
            raise ValueError("new_star_Y2 and new_star_Z2 must sum to 1 or less.")

        if((isinstance(new_star_Y1,float)) and (isinstance(new_star_Z1,float))):
            new_star_X1 = np.full(obj_num,1. - new_star_Y1 - new_star_Z1)
            new_star_Y1 = np.full(obj_num,new_star_Y1)
            new_star_Z1 = np.full(obj_num,new_star_Z1)
        if((isinstance(new_star_Y2,float)) and (isinstance(new_star_Z2,float))):
            new_star_X2 = np.full(obj_num,1. - new_star_Y2 - new_star_Z2)
            new_star_Y2 = np.full(obj_num,new_star_Y2)
            new_star_Z2 = np.full(obj_num,new_star_Z2)
        if((isinstance(new_star_Y1,np.ndarray)) and (isinstance(new_star_Z1,np.ndarray))):
            assert new_star_Y1.shape == (obj_num,),"new_star_Y1: all arrays must be 1d and the same length"
            assert new_star_Z1.shape == (obj_num,),"new_star_Z1: all arrays must be 1d and the same length"
            new_star_X1 = np.ones(obj_num) - new_star_Y1 - new_star_Z1
        if((isinstance(new_star_Y2,np.ndarray)) and (isinstance(new_star_Z2,np.ndarray))):
            assert new_star_Y2.shape == (obj_num,),"new_star_Y2: all arrays must be the same length"
            assert new_star_Z2.shape == (obj_num,),"new_star_Z2: all arrays must be the same length"
            new_star_X2 = np.ones(obj_num) - new_star_Y2 - new_star_Z2
        else: raise TypeError("new_star_Y1, new_star_Z1 and new_star_Y2, new_star_Z2 must be either both floats or numpy arrays")


        #Now add new values
        self.star_mass1 = np.concatenate([self.star_mass1, new_star_mass1])
        self.star_mass2 = np.concatenate([self.star_mass2, new_star_mass2])
        self.star_radius1 = np.concatenate([self.star_radius1, new_star_radius1])
        self.star_radius2 = np.concatenate([self.star_radius2, new_star_radius2])
        self.star_X1 = np.concatenate([self.star_X1, new_star_X1])
        self.star_Y1 = np.concatenate([self.star_Y1, new_star_Y1])
        self.star_Z1 = np.concatenate([self.star_Z1, new_star_Z1])
        self.star_X2 = np.concatenate([self.star_X2, new_star_X2])
        self.star_Y2 = np.concatenate([self.star_Y2, new_star_Y2])
        self.star_Z2 = np.concatenate([self.star_Z2, new_star_Z2])
        self.star_spin1 = np.concatenate([self.star_spin1, new_star_spin1])
        self.star_spin2 = np.concatenate([self.star_spin2, new_star_spin2])
        self.star_spin_angle1 = np.concatenate([self.star_spin_angle1, new_star_spin_angle1])
        self.star_spin_angle2 = np.concatenate([self.star_spin_angle2, new_star_spin_angle2])
        self.bin_e = np.concatenate([self.bin_e, new_bin_e])
        self.bin_a = np.concatenate([self.bin_a, new_bin_a])

        new_total_mass = new_star_mass1 + new_star_mass2

        super(AGNBinaryStar,self).add_objects(new_mass = new_total_mass,
                                                   new_spin=None,
                                                   new_spin_angle=None,
                                                   new_a = new_cm_orb_a,
                                                   new_inc=new_cm_orb_inc,
                                                   new_e=new_cm_orb_ecc,
                                                   obj_num=obj_num)


class AGNBinaryBlackHole(AGNObject):
    """
    An array of binary black holes. Should include all objects of this type. No other objects should contain objects of this type.
    """

    def __init__(self, bh_mass1 = None,
                       bh_mass2 = None,
                       bh_orb_a1 = None,
                       bh_orb_a2 = None,
                       bh_spin1 = None,
                       bh_spin2 = None,
                       bh_spin_angle1 = None,
                       bh_spin_angle2 = None,
                       bh_orb_ang_mom1 = None,
                       bh_orb_ang_mom2 = None,
                       bin_e = None,
                       bin_a = None,
                       bin_inc=None,
                       #com_orb_a=None,
                       com_orb_inc=None,
                       com_orb_ecc=None,
                       obj_num = None,
                     **kwargs):
        

        #if obj_num is None: obj_num = bh_mass1.size

        self.mass1 = None
        self.mass2 = None
        self.orb_smbh_a1 = None
        self.orb_smbh_a2 = None
        self.spin1 = None
        self.spin2 = None
        self.spin_angle1 = None
        self.spin_angle2 = None
        self.bin_orb_a = None
        self.bin_com = None
        self.t_gw = None
        self.merger_flag = None
        self.t_mgr = None
        self.bin_orb_e = None
        self.gen1 = None
        self.gen2 = None
        self.bin_orb_ang_mom = None
        self.bin_orb_inc = None
        self.bin_smbh_orb_e = None


        #Now assign attributes
        """  self.bh_mass1 = bh_mass1
        self.bh_mass2 = bh_mass2
        self.bh_orb_a1 = bh_orb_a1
        self.bh_orb_a2 = bh_orb_a2
        self.bh_spin1 = bh_spin1
        self.bh_spin2 = bh_spin2
        self.bh_spin_angle1 = bh_spin_angle1
        self.bh_spin_angle2 = bh_spin_angle2
        self.bh_orb_ang_mom1 = bh_orb_ang_mom1
        self.bh_orb_ang_mom2 = bh_orb_ang_mom2
        self.bin_e = bin_e
        self.bin_a = bin_a
        self.bin_inc = bin_inc
        #self.com_orb_a = cm_orb_a
        self.com_orb_ecc = com_orb_ecc
        self.com_orb_inc = com_orb_inc
        self.com_orb_ang_mom = None """

        #self.com_orb_a = np.abs(bh_orb_a1 - bh_orb_a2)

        #Now calculate properties for the AGNObject class aka the totals
        #total_mass = bh_mass1 + bh_mass2
        #total_spin = None # we will pass None for now, until we decide how to treat angular momentum
        #total_spin_angle = None  # we will pass None for now, until we decide how to treat angular momentum


        """ super(AGNBinaryBlackHole,self).__init__(mass = total_mass,
                                                spin = total_spin,
                                                spin_angle = total_spin_angle,
                                                orb_a = self.cm_orb_a,
                                                orb_inc=self.cm_orb_inc,
                                                orb_ecc = self.cm_orb_ecc,
                                                orb_arg_periapse=None,# TODO ISSUE
                                                smbh_mass = None, # TODO ISSUE
                                                obj_num=obj_num) # TODO ISSUE """
        super(AGNBinaryBlackHole,self).__init__(mass = None,
                                                spin = None,
                                                spin_angle = None,
                                                orb_a = None,
                                                orb_inc=None,
                                                orb_ecc = None,
                                                orb_arg_periapse=None) # TODO ISSUE


    def __repr__(self):
        return('AGNBinaryStar(): {} black hole binaries'.format(len(self.mass)))
    
    """def add_binaries(self, new_bh_mass1 = None,
                       new_bh_mass2 = None,
                       new_bh_orb_a1 = None,
                       new_bh_orb_a2 = None,
                       new_bh_spin1 = None,
                       new_bh_spin2 = None,
                       new_bh_spin_angle1 = None,
                       new_bh_spin_angle2 = None,
                       new_bin_e = None,
                       new_bin_a = None,
                       new_bin_inc=None,
                       new_cm_orb_a=None,
                       new_cm_orb_inc=None,
                       new_cm_orb_ecc=None,
                       obj_num = None,
                    **kwargs):
        


        if obj_num is None: obj_num = new_bh_mass1.size

        self.bh_mass1 = np.concatenate([self.bh_mass1, new_bh_mass1])
        self.bh_mass2 = np.concatenate([self.bh_mass2, new_bh_mass2])
        self.bh_orb_a1 = np.concatenate([self.bh_orb_a1, new_bh_orb_a1])
        self.bh_orb_a2 = np.concatenate([self.bh_orb_a2, new_bh_orb_a2])
        self.bh_spin1 = np.concatenate([self.bh_spin1, new_bh_spin1])
        self.bh_spin2 = np.concatenate([self.bh_spin2, new_bh_spin2])
        self.bh_spin_angle1 = np.concatenate([self.bh_spin_angle1, new_bh_spin_angle1])
        self.bh_spin_angle2 = np.concatenate([self.bh_spin_angle2, new_bh_spin_angle2])
        self.bin_e = np.concatenate([self.bh_bin_e, new_bh_bin_e])
        self.bin_a = np.concatenate([self.bh_bin_a, new_bh_bin_a])
        self.bin_inc = np.concatenate([self.bin_inc, new_bin_inc])
        self.cm_orb_a = np.concatenate([self.cm_orb_a, new_cm_orb_a])
        self.cm_orb_inc = np.concatenate([self.cm_orb_inc, new_cm_orb_inc])
        self.cm_orb_ecc = np.concatenate([self.cm_orb_ecc, new_cm_orb_ecc])

        new_total_mass = new_bh_mass1 + new_bh_mass2
        new_spin = new_bh_spin1 + new_bh_spin2
        new_spin_angle = new_bh_spin_angle1 + new_bh_spin_angle2

        super(AGNBinaryBlackHole,self).add_objects(new_mass = new_total_mass,
                                                   new_spin=None,
                                                   new_spin_angle=None,
                                                   new_a = new_cm_orb_a,
                                                   new_inc=new_cm_orb_inc,
                                                   new_e=new_cm_orb_ecc,
                                                   obj_num=obj_num) """

    def add_binaries(self,
                     blackholes,
                     close_encounters,
                     retro,
                    obj_num = None,
                    **kwargs):
        
        if obj_num is None: obj_num = blackholes.mass.size

        #need to basically copy the add_new_binary.add_to_binary_array2 function

        super(AGNBinaryBlackHole,self).add_objects(new_mass = new_total_mass,
                                                   new_spin=None,
                                                   new_spin_angle=None,
                                                   new_a = new_cm_orb_a,
                                                   new_inc=new_cm_orb_inc,
                                                   new_e=new_cm_orb_ecc,
                                                   obj_num=obj_num)


obj_types = {0 : "single black hole",
             1 : "binary black hole",
             2 : "single star",
             3 : "binary star",
             4 : "exploded star"
            }

obj_direction = {0 : "orbit direction undetermined",
             1 : "prograde orbiter",
             -1 : "retrograde orbiter"}


class AGNFilingCabinet(AGNObject):
    """
    Master catalog of all objects in the disk.
    """
    def __init__(self, category = None, direction = None, agnobj = None):

        self.id_num = np.arange(0,len(category)) #creates ID numbers starting at 0 when initializing

        if agnobj is not None:
            self.category = np.full(agnobj.mass.shape,category)
        else:
            raise AttributeError("Initializing an instance of AGNFilingCabinet requires passing an instance of an AGNObject.")
        
        if direction is None:
            self.direction = np.full(agnobj.mass.shape,0)
        else:
            self.direction = np.full(agnobj.mass.shape,direction)
        #self.id_num = id_num
        #self.category = category
        #self.mass = mass
        #self.orb_a = orb_a

        self.orb_ang_mom = agnobj.orb_ang_mom


        super(AGNFilingCabinet,self).__init__(mass=agnobj.mass, spin=agnobj.spin, spin_angle=agnobj.spin_angle,orb_a=agnobj.orb_a, orb_inc=agnobj.orb_inc, orb_ecc=agnobj.orb_ecc, orb_arg_periapse=agnobj.orb_arg_periapse)

    def __repr__(self):
        """         totals = "AGN Filing Cabinet\n"
        for key in obj_types:
            #print(key,getattr(self,"category").count(key))
            totals += (f"\t{obj_types[key]}: { np.sum(getattr(self,"category") == key) }\n")
            for key2 in obj_direction:
                totals += (f"\t\t{obj_direction[key2]}: {np.sum((getattr(self,"category") == key) & (getattr(self,"direction") == key2))}\n")
        totals += f"{len(getattr(self,"category"))} objects total" """
        return(totals)

    def change_category(self, obj_id = None, new_category = None):
        getattr(self,"category")[np.isin(getattr(self,"id_num"),obj_id)] = new_category

    def change_direction(self, obj_id = None, new_direction = None):
        getattr(self,"direction")[np.isin(getattr(self,"id_num"),obj_id)] = new_direction


    def remove_objects(self, idx_remove = None):
        """
        Removes objects at specified indices.

        Parameters
        ----------
        idx_remove : numpy array
            Indices to remove

        Returns
        -------
        ???
        idx_remove should be a numpy array of indices to change, e.g., [2, 15, 23]
        as written, this loops over all attributes, not just those in the AGNObjects class,
        i.e., we don't need a separate remove_objects method for the subclasses.
        """

        #Check that the index array is a numpy array.
        #assert isinstance(idx_remove,np.ndarray),"idx_remove must be numpy array"

        if idx_remove is None:
            return None
        
        idx_change = np.ones(len(self.mass),dtype=bool)
        idx_change[idx_remove] = False
        for attr in vars(self).keys():
            print(attr)
            setattr(self,attr,getattr(self,attr)[idx_change])


    def add_objects(self, create_id = False, new_id_num = None, new_category = None, new_direction = None, **kwargs):
        if ((create_id is True) and (new_id_num is None)):
            id_start_value = self.id_num.max() + 1
            new_id_num = np.arange(id_start_value, id_start_value+len(new_category),1)
        elif ((create_id is True) and (new_id_num is not None)):
            raise AttributeError("if create_id is True then new_id_num must be None. If create_id is False then new_id_num must be array of new ID numbers.")
        
        #self.id_num = np.concatenate([self.id_num, new_id_num])

        self.category = np.concatenate([self.category, new_category])
        self.direction = np.concatenate([self.direction, new_direction])

        super(AGNFilingCabinet,self).add_objects(new_id_num=new_id_num,**kwargs)
