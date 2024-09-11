import numpy as np
from copy import deepcopy
from mcfacts.setup import setupdiskstars, setupdiskblackholes

"""
def check_1d_length(arr_list):
    if(len({arr.shape for arr in arr_list}) > 1):
        raise ValueError('Arrays are not all 1d.')

    if( len(set(map(len,arr_list))) != 1):
        raise ValueError('Arrays are not all the same length.')
 """

# TODO: similar vein: print method? Same thing maybe?
# TODO: custom error messages when you don't supply all the fields for init and add methods
# TODO: custom error messages when input arrays to init and add methods aren't the same length
# TODO: dump_record_array writes every value as a float. Make dictionary with all attributes and datatypes? Or is a float fine?
# TODO: init_from_file: you have to initialize an empty AGNObject before you can init_from_file and that seems weird.
#       check if this is now just an AGNObject because we would want it to be an AGNStar or AGNBlackHole etc
# TODO: error if you try to pass kwarg that doesnt exist. AttributeError?
# TODO: issue: right now you can pass AGNStar arrays that are 8-elements long for the star parameters and 10-elements long 
#       for the AGNObject parameters and it doesn't complain.


class AGNObject(object):
    """
    A superclass that holds parameters that apply to all objects in McFacts.
    It is formatted as an object full of arrays.
    No instances of the AGNObject class should be created, it is a superclass
    to the AGNStar, AGNBlackHole, etc. classes.
    All orbital attributes to this class are with respect to the central SMBH.
    if the subclass is a Binary object, then attributes are for the total
    quantities (total mass, etc.), not the binary components.
    No instances of the AGNObject class should be created, it is a superclass
    to the AGNStar, AGNBlackHole, etc. classes. If the 
    """

    def __init__(self,
                 mass=None,
                 spin=None,  # internal quantity. total J for a binary
                 spin_angle=None,  # angle between J and orbit around SMBH for binary
                 orb_a=None,  # location
                 orb_inc=None,  # of CoM for binary around SMBH
                 # orb_ang_mom = None,  # redundant, should be computed from keplerian orbit formula for L in terms of mass, a, eccentricity
                 orb_ecc=None,
                 orb_arg_periapse=None,
                 smbh_mass=None,
                 obj_num=None,
                 id_start_val=None):
        """
        Creates an instance of the AGNObject class.

        Parameters
        ----------
        mass : numpy array
            masses
        spin : numpy array
            spins
        spin_angle : numpy array
            spin angles
        orb_a : numpy array
            orbital semi-major axis with respect to the SMBH
        orb_inc : numpy array
            orbital inclination with respect to the SMBH
        orb_ecc : numpy array
            orbital eccentricity with respect to the SMBH
        orb_arg_periapse : numpy array
            argument of the orbital periapse with respect to the SMBH
        smbh_mass : numpy array
            mass of the SMBH
        obj_num : int, optional
            number of objects, by default None
        id_start_val : numpy array
            ID numbers for the objects, by default None
        """

        # Make sure all inputs are included
        """ if mass is None: raise AttributeError("mass is not included in inputs")
        if spin is None: raise AttributeError('spin is not included in inputs')
        if spin_angle is None: raise AttributeError('spin_angle is not included in inputs')
        if orb_a is None: raise AttributeError('orb_a is not included in inputs')
        if orb_inc is None: raise AttributeError('orb_inc is not included in inputs')
        #if orb_ang_mom is None: raise AttributeError('orb_ang_mom is not included in inputs')
        if orb_ecc is None: raise AttributeError('orb_ecc is not included in inputs') """

        """

        assert mass.shape == (obj_num,),"mass: all arrays must be 1d and the same length"
        assert spin.shape == (obj_num,),"spin: all arrays must be 1d and the same length"
        assert spin_angle.shape == (obj_num,),"spin_angle: all arrays must be 1d and the same length"
        assert orb_a.shape == (obj_num,),"orb_a: all arrays must be 1d and the same length"
        assert orb_inc.shape == (obj_num,),"orb_inc: all arrays must be 1d and the same length"
        #assert orb_ang_mom.shape == (obj_num,),"orb_ang_mom: all arrays must be 1d and the same length"
        assert orb_ecc.shape == (obj_num,),"orb_ecc: all arrays must be 1d and the same length" """

        if mass is None:
            # creating an empty object
            # i know this is a terrible way to do things
            self.gen = None
            self.id_num = None
        else:
            if obj_num is None: obj_num = mass.size
            self.gen = np.full(obj_num, 1)
            if id_start_val is None:
                self.id_num = np.arange(0, len(mass))  # creates ID numbers sequentially from 0
            else:  # if we have an id_start_val aka these aren't the first objects in the disk
                self.id_num = np.arange(id_start_val, id_start_val + len(mass), 1)

        self.mass = mass
        self.spin = spin
        self.spin_angle = spin_angle
        self.orb_a = orb_a
        self.orb_inc = orb_inc
        self.orb_ecc = orb_ecc
        self.orb_arg_periapse = orb_arg_periapse

    def add_objects(self,
                    new_mass=None,
                    new_spin=None,
                    new_spin_angle=None,
                    new_orb_a=None,
                    new_orb_inc=None,
                    new_orb_ang_mom=None,
                    new_orb_ecc=None,
                    new_orb_arg_periapse=None,
                    new_gen=None,
                    new_id_num=None,
                    obj_num=None):
        """
        Append new objects to the AGNObject. This method is not called
        directly, it is only called by the subclasses' add methods.

        Parameters
        ----------
        new_mass : numpy array
            masses to be added
        new_spin : numpy array
            spins to be added
        new_spin_angle : numpy array
            spin angles to be added
        new_orb_a : numpy array
            semi-major axes to be added
        new_orb_inc : numpy array
            orbital inclinations to be added
        new_orb_ang_mom : numpy array
            orbital angular momentum to be added
        new_orb_ecc : numpy array
            orbital eccentricities to be added
        new_orb_arg_periapse : numpy array
            orbital arguments of the periapse to be added
        new_gen : numpy array
            generations to be added
        new_id_num : numpy array,optional
            ID numbers to be added
        obj_num : int, optional
            Number of objects to be added.
        """        

        # Make sure all inputs are included
        """ if new_mass is None: raise AttributeError('new_mass is not included in inputs')
        if new_spin is None: raise AttributeError('new_spin is not included in inputs')
        if new_spin_angle is None: raise AttributeError('new_spin_angle is not included in inputs')
        if new_a is None: raise AttributeError('new_a is not included in inputs')
        if new_inc is None: raise AttributeError('new_inc is not included in inputs')
        #if new_orb_ang_mom is None: raise AttributeError('new_orb_ang_mom is not included in inputs')
        if new_e is None: raise AttributeError('new_e is not included in inputs')

        if obj_num is None: obj_num = new_mass.size

        assert new_mass.shape == (obj_num,),"new mass: all arrays must be 1d and the same length"
        assert new_spin.shape == (obj_num,),"new_spin: all arrays must be 1d and the same length"
        assert new_spin_angle.shape == (obj_num,),"new_spin_angle: all arrays must be 1d and the same length"
        assert new_a.shape == (obj_num,),"new_a: all arrays must be 1d and the same length"
        assert new_inc.shape == (obj_num,),"new_inc: all arrays must be 1d and the same length"
        #assert new_orb_ang_mom.shape == (obj_num,),"new_orb_ang_mom: all arrays must be 1d and the same length"
        assert new_e.shape == (obj_num,),"new_e: all arrays must be 1d and the same length" """

        self.mass = np.concatenate([self.mass, new_mass])
        self.spin = np.concatenate([self.spin, new_spin])
        self.spin_angle = np.concatenate([self.spin_angle, new_spin_angle])
        self.orb_a = np.concatenate([self.orb_a, new_orb_a])
        self.orb_ang_mom = np.concatenate([self.orb_ang_mom, new_orb_ang_mom])
        self.orb_inc = np.concatenate([self.orb_inc, new_orb_inc])
        self.orb_ecc = np.concatenate([self.orb_ecc, new_orb_ecc])
        self.orb_arg_periapse = np.concatenate([self.orb_arg_periapse, new_orb_arg_periapse])
        self.gen = np.concatenate([self.gen, new_gen])
        self.id_num = np.concatenate([self.id_num, new_id_num])

    def remove_index(self, idx_remove=None):
        """
        Removes objects at specified indices.

        Parameters
        ----------
        idx_remove : numpy array
            indices to remove
        """

        # Check that the index array is a numpy array.
        # assert isinstance(idx_remove,np.ndarray),"idx_remove must be numpy array"

        if idx_remove is None:
            return None

        idx_change = np.ones(len(self.mass), dtype=bool)
        idx_change[idx_remove] = False
        for attr in vars(self).keys():
            setattr(self, attr, getattr(self, attr)[idx_change])

    def remove_id_num(self, id_num_remove=None):
        """
        Filters AGNObject to remove the objects at the specified ID numbers

        Parameters
        ----------
        id_num_keep : numpy array
            ID numbers to keep, others are removed
        """
        keep_mask = ~(np.isin(getattr(self, "id_num"), id_num_remove))
        for attr in vars(self).keys():
            setattr(self, attr, getattr(self, attr)[keep_mask])

    def keep_index(self, idx_keep):
        """
        Filters AGNObject to only keep the objects at the specified indices.

        Parameters
        ----------
        idx_keep : numpy array
            indices to keep, others are removed.
        """

        # Check that the index array is a numpy array.
        # assert isinstance(idx_remove,np.ndarray),"idx_remove must be numpy array"

        if idx_keep is None:
            return None
        
        idx_change = np.zeros(len(self.mass), dtype=bool)
        idx_change[idx_keep] = True
        for attr in vars(self).keys():
            setattr(self, attr, getattr(self, attr)[idx_change])

    def keep_id_num(self, id_num_keep):
        """
        Filters AGNObject to only keep the objects at the specified ID numbers

        Parameters
        ----------
        id_num_keep : numpy array
            ID numbers to keep, others are removed
        """
        keep_mask = (np.isin(getattr(self, "id_num"), id_num_keep))
        for attr in vars(self).keys():
            setattr(self, attr, getattr(self, attr)[keep_mask])

    def at(self, id_num, attr):
        """
        Returns the attribute at the specified ID numbers

        Parameters
        ----------
        id_num : numpy array
            ID numbers of objects to return
        attr : str
            attribute to return

        Returns
        -------
        val : numpy array
            specified attribute at specified ID numbers
        """
        id_mask = (np.isin(getattr(self, "id_num"), id_num))

        try:
            val = getattr(self, attr)[id_mask]
        except:
            raise AttributeError("{} is not an attribute of the AGNObject".format(attr))

        return (val)

    def copy(self):
        """
        Creates a deep copy of the AGNObject

        Parameters
        ----------
        None

        Returns
        -------
        copied_object : AGNObject
            new copy of AGNObject with no references to original AGNObject
        """
        copied_object = deepcopy(self)
        return (copied_object)

    """ def locate(self, idx=None):

        #Returns objects at specified indices

        # Check that index array is numpy array
        assert isinstance(idx, np.ndarray),"idx must be numpy array"

        if idx is None:
            return None

        idx_full = np.zeros(len(self.mass),dtype=bool)
        idx_full[idx] = True """

    def sort(self, sort_attr=None):
        """
        Sorts all attributes of the AGNObject by the passed attribute

        Parameters
        ----------
        sort_attr : AGNObject attribute array
            array to sort the AGNObject by
        """

        # sorted indices of the array to sort by
        sort_idx = np.argsort(sort_attr)

        # Each attribute is then sorted to be in this order
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
        return (list(vars(self).keys()))

    def return_record_array(self):
        """
        Returns a numpy dictionary of all attributes in the AGNObject

        Parameters
        ----------
        None

        Returns
        -------
        dat_out : numpy dictionary
            dictionary array of all attributes in the AGNObject. Everything
            is written as a float.
        """

        dtype = np.dtype([(attr, 'float') for attr in vars(self).keys()])
        dat_out = np.empty(len(self.mass), dtype=dtype)
        for attr in vars(self).keys():
            dat_out[attr] = getattr(self, attr)
        return (dat_out)

    def to_file(self, fname=None):
        """
        Writes AGNObject to csv file. Header row started with `#` character.

        Parameters
        ----------
        fname : str
            filename including path
        """

        import pandas
        samples_out = self.return_record_array()
        dframe = pandas.DataFrame(samples_out)
        dframe.to_csv(fname, sep=' ',
                      header=[f"#{x}" if x == dframe.columns[0] else x for x in dframe.columns],
                      index=False)  # `#` is not pre-appended...just boolean

    def init_from_file(self, fname=None):
        """
        Reads in file from previous AGNObject.
           Not fully implemented. Would need to init AGNObject and then read from file?

        Parameters
        ----------
        fname : str
            file to read in
        """

        dat_in = np.genfromtxt(fname, names=True)
        for name in dat_in.dtype.names:
            setattr(self, name, dat_in[name])


class AGNStar(AGNObject):
    """
    A subclass of AGNObject for single stars. It extends AGNObject by adding
    attributes for mass, radius, and chemical composition.
    """

    def __init__(self,
                 mass=None,
                 radius=None,
                 orb_a=None,
                 orb_inc=None,
                 star_Y=None,
                 star_Z=None,
                 star_num=None,
                 smbh_mass=None,
                 **kwargs):
        """Creates an instance of the AGNStar class. This is a subclass
           of the AGNObject class. AGNStar adds additional star-specific
           parameters to the AGNObject. It calculates orbital angular
           momentum for stars.

        Parameters
        ----------
        mass : numpy array
            _description_, by default None
        radius : numpy array
            _description_, by default None
        orb_a : numpy array
            _description_, by default None
        orb_inc : numpy array
            _description_, by default None
        star_Y : numpy array
            helium fraction of stars
        star_Z : numpy array
            metals fraction of stars
        star_num : int, optional
            number of stars, by default None
        smbh_mass : float
            mass of the SMBH
        """
        # Make sure all inputs are included
        # if radius is None: raise AttributeError('radius is not included in inputs')
        """ if star_Y is None: raise AttributeError('star_Y is not included in inputs')
        if star_Z is None: raise AttributeError('star_Z is not included in inputs') """

        if star_num is None: star_num = radius.size

        # Make sure inputs are numpy arrays
        if radius is None:
            self.radius = setupdiskstars.setup_disk_stars_radii(masses=mass)

        else:
            assert isinstance(radius, np.ndarray),"radius is not a numpy array"
            self.radius = radius

        if (np.any(star_Y + star_Z > 1.)):
            raise ValueError("star_Y and star_Z must sum to 1 or less.")

        if ((isinstance(star_Y, float) and (isinstance(star_Z, float)))):
            self.star_X = np.ones(len(radius)) - np.full(len(radius),star_Y) - np.full(len(radius),star_Z)
            self.star_Y = np.full(len(radius), star_Y)
            self.star_Z = np.full(len(radius), star_Z)

        elif ((isinstance(star_Y, np.ndarray) and isinstance(star_Z, np.ndarray))):
            assert radius.shape == (star_num,), "radius: all arrays must be 1d and the same length"
            assert star_Y.shape == (star_num,), "star_Y, array: all arrays must be 1d and the same length"
            assert star_Z.shape == (star_num,), "star_Z, array: all arrays must be 1d and the same length"

            self.star_X = 1. - star_Y - star_Z
            self.star_Y = star_Y
            self.star_Z = star_Z

        else:
            raise TypeError("star_Y and star_Z must be either both floats or numpy arrays")

        mass_total = mass + smbh_mass
        mass_reduced = mass*smbh_mass/mass_total
        self.orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(star_num=star_num,
                                                                       mass_reduced=mass_reduced,
                                                                       mass_total=mass_total,
                                                                       orb_a=orb_a,
                                                                       orb_inc=orb_inc)

        super(AGNStar, self).__init__(mass=mass, orb_a=orb_a, orb_inc=orb_inc, obj_num=star_num, **kwargs)  # calls top level functions

    def __repr__(self):
        """
        Creates a string representation of AGNStar. Prints out
        the number of stars present in this instance of AGNStar.

        Returns
        -------
        totals : str
            number of stars in AGNStar
        """
        totals = 'AGNStar(): {} single stars'.format(len(self.mass))
        return (totals)

    def add_stars(self, new_radius=None, new_Y=None, new_Z=None, obj_num=None, **kwargs):
        """
        Append new stars to the end of AGNStar. This method updates the star
        specific parameters and then sends the rest to the AGNObject
        add_objects() method.

        Parameters
        ----------
        new_radius : numpy array
            radii of new stars
        new_Y : numpy array
            helium mass fraction of new stars
        new_Z : numpy array
            metals mass fraction of new stars
        obj_num : int, optional
            number of objects to be added, by default None
        """

        # Make sure all inputs are included
        if new_radius is None: raise AttributeError("new_radius is not included in inputs")
        if new_Y is None: raise AttributeError("new_Y is not included in inputs")
        if new_Z is None: raise AttributeError("new_Z is not included in inputs")

        if obj_num is None: obj_num = new_radius.size

        assert new_radius.shape == (obj_num,), "new_radius: all arrays must be 1d and the same length"
        self.radius = np.concatenate([self.radius, new_radius])

        if (np.any(new_Y + new_Z) > 1.) : raise ValueError("new_Y and new_Z must sum to 1 or less")

        if ((isinstance(new_Y, float)) and (isinstance(new_Z, float))):
            self.star_X = np.concatenate(self.star_X, np.full(obj_num, 1.-new_Y-new_Z))
            self.star_Y = np.concatenate([self.star_Y, np.full(obj_num, new_Y)])
            self.star_Z = np.concatenate([self.star_Z, np.full(obj_num, new_Z)])
            
        if ((isinstance(new_Y, np.ndarray)) and (isinstance(new_Z, np.ndarray))):
            self.star_X = np.concatenate([self.star_X, np.ones(obj_num) - new_Y - new_Z])
            self.star_Y = np.concatenate([self.star_Y, new_Y])
            self.star_Z = np.concatenate([self.star_Z, new_Z])
        super(AGNStar, self).add_objects(obj_num=obj_num, **kwargs)


class AGNBlackHole(AGNObject):
    """
    A subclass of AGNObject for single black holes. It extends AGNObject. It
    calculates orbital angular momentum for black holes.
    """
    def __init__(self, mass=None, **kwargs):
        """Creates an instance of AGNStar object.

        Parameters
        ----------
        mass : numpy array
            black hole masses
        """

        self.orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(n_bh=len(mass))

        super(AGNBlackHole, self).__init__(mass=mass, **kwargs)

    def __repr__(self):
        """
        Creates a string representation of AGNBlackHole. Prints out
        the number of black holes present in this instance of AGNBlackHole.

        Returns
        -------
        totals : str
            number of black holes in AGNBlackHole
        """

        totals = 'AGNBlackHole(): {} single black holes'.format(len(self.mass))
        return (totals)

    def add_blackholes(self, obj_num=None, **kwargs):
        """
        Append black holes to the AGNBlackHole object.

        Parameters
        ----------
        obj_num : int, optional
            number of black holes to be added, by default None
        """
        super(AGNBlackHole, self).add_objects(obj_num=obj_num, **kwargs)


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

        super(AGNBinaryBlackHole, self).add_objects(new_mass=new_total_mass,
                                                    new_spin=None,
                                                    new_spin_angle=None,
                                                    new_a = new_cm_orb_a,
                                                    new_inc=new_cm_orb_inc,
                                                    new_e=new_cm_orb_ecc,
                                                    obj_num=obj_num)


obj_types = {0 : "single black hole",
             1 : "single star",
             2 : "binary black hole",
             3 : "binary star",
             4 : "exploded star"
            }

obj_direction = {0 : "orbit direction undetermined",
                 1 : "prograde orbiter",
                -1 : "retrograde orbiter"}

obj_disk_loc = {0 : "disk location undetermined",
                1 : "outer disk",
               -1 : "inner disk"}


class AGNFilingCabinet(AGNObject):
    """
    Master catalog of all objects in the disk. Each object has a unique ID number,
    type, and orbital direction. Currently it also takes in all parameters present in AGNObject,
    but these are not updated when the instances of AGNBlackHole and AGNStar are updated.
    """
    def __init__(self,
                 id_num,
                 category,
                 orb_a,
                 mass,
                 size,
                 direction=None,
                 disk_inner_outer=None):
        """
        Creates an instance of AGNFilingCabinet. It extends AGNObject by
        recording ID numbers for each object and their category, so that
        they can be easily found in their respective AGNObjects.

        Parameters
        ----------
        id_num : numpy array
            ID numbers of the objects
        category : numpy array of ints
            category (black hole, star, etc.) of the objects
        orb_a : numpy array
            orbital semi-major axis with respect to the SMBH
        mass : numpy array
            masses of the objects (for binaries this is total mass)
        size : numpy array
            for BH this is set to -1, for stars this is set to the radius in Rsun,
            for binaries this is the binary's semi-major axis (aka separation)
            in R_g
        direction : numpy array
            direction of the orbit of the objects, optional
        disk_inner_outer : numpy array
            if the object is in the inner or outer disk
        """ 

        # Set attributes
        self.id_num = id_num
        # future: pass an int to category and it fills in the rest
        self.category = category
        self.orb_a = orb_a
        self.mass = mass
        # size is radius for stars, -1 for BH, bin_a for binary BH
        self.size = size

        # Set direction as 0 (undetermined) if not passed
        # Otherwise set as what is passed
        if direction is None:
            self.direction = np.full(id_num.shape,0)
        else:
            self.direction = direction
        
        # Set disk_inner_outer as 0 (undetermined if not passed)
        # Otherwise set as what is passed
        if disk_inner_outer is None:
            self.disk_inner_outer = np.full(id_num.shape,0)
        else:
            self.disk_inner_outer = disk_inner_outer
        
    def __repr__(self):
        """
        Creates a string representation of AGNFilingCabinet. Prints out
        the number and types of objects present in AGNFilingCabinet and
        their direction (prograde, retrograde, or undetermined). Not
        currently working.

        Returns
        -------
        totals : str
            number and types of objects in AGNFilingCabinet
        """

        totals = "AGN Filing Cabinet\n"
        for key in obj_types:
            #print(key,getattr(self,"category").count(key))
            totals += (f"\t{obj_types[key]}: { np.sum(getattr(self,"category") == key) }\n")
            for direc in obj_direction:
                totals += (f"\t\t{obj_direction[direc]}: {np.sum((getattr(self,"category") == key) & (getattr(self,"direction") == direc))}\n")
            totals += "\n"
            for loc in obj_disk_loc:
                totals += (f"\t\t{obj_disk_loc[loc]}: {np.sum((getattr(self,"category") == key) & (getattr(self,"disk_inner_outer") == loc))}\n")
        totals += f"{len(getattr(self,"category"))} objects total"
        return(totals)

    def update(self, id_num, attr, new_info):
        """Update a given attribute in AGNFilingCabinet for the given ID numbers

        Parameters
        ----------
        id_num : numpy array
            ID numbers of the objects to be changed
        attr : str
            the attribute to be changed
        new_info : numpy array
            the new data for the attribute
        """

        if not isinstance(attr, str):
            raise TypeError("`attr` must be passed as a string")
        
        try:
            getattr(self,attr)[np.isin(getattr(self,"id_num"),id_num)] = new_info
        except:
            raise AttributeError("{} is not an attribute of AGNFilingCabinet".format(attr))


    def add_objects(self, new_id_num, new_category, new_orb_a,
                    new_mass, new_size, new_direction, new_disk_inner_outer):
        """
        Append objects to the AGNFilingCabinet.

        Parameters
        ----------
        new_id_num : numpy array
            ID numbers to be added
        new_category : numpy array
            categories to be added
        new_orb_a : numpy array
            orbital semi-major axes to be added
        new_mass : numpy array
            masses to be added
        new_size : numpy array
            sizes to be added (BH: -1, stars: radii in Rsun,
            binaries: separation in R_g)
        new_direction : numpy array
            orbital directions of objects to be added
        new_disk_inner_outer : numpy array
            new inner/outer disk locations to be added
        """
        
        self.id_num = np.concatenate([self.id_num, new_id_num])
        self.category = np.concatenate([self.category, new_category])
        self.orb_a = np.concatenate([self.orb_a, new_orb_a])
        self.mass = np.concatenate([self.mass, new_mass])
        self.size = np.concatenate([self.size, new_size])
        self.direction = np.concatenate([self.direction, new_direction])
        self.disk_inner_outer = np.concatenate([self.disk_inner_outer, new_disk_inner_outer])