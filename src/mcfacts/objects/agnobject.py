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

# Empty array to pass
empty_arr = np.array([])


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
    to the AGNStar, AGNBlackHole, etc. classes.
    """

    def __init__(self,
                 mass=empty_arr,
                 spin=empty_arr,  # internal quantity. total J for a binary
                 spin_angle=empty_arr,  # angle between J and orbit around SMBH for binary
                 orb_a=empty_arr,  # location
                 orb_inc=empty_arr,  # of CoM for binary around SMBH
                 # orb_ang_mom = None,  # redundant, should be computed from keplerian orbit formula for L in terms of mass, a, eccentricity
                 orb_ecc=empty_arr,
                 orb_arg_periapse=empty_arr,
                 galaxy=empty_arr,
                 time_passed=empty_arr,
                 smbh_mass=empty_arr,
                 obj_num=0,
                 id_start_val=0):
        """
        Creates an instance of the AGNObject class.

        Parameters
        ----------
        mass : numpy array
            masses in Msun
        spin : numpy array
            spins
        spin_angle : numpy array
            spin angles in radians
        orb_a : numpy array
            orbital semi-major axis with respect to the SMBH in R_g
        orb_inc : numpy array
            orbital inclination with respect to the SMBH
        orb_ecc : numpy array
            orbital eccentricity with respect to the SMBH
        orb_arg_periapse : numpy array
            argument of the orbital periapse with respect to the SMBH
        galaxy : numpy array
            galaxy iteration
        time_passed : numpy array
            time passed
        smbh_mass : numpy array
            mass of the SMBH in Msun
        obj_num : int, optional
            number of objects, by default 0
        id_start_val : numpy array
            ID numbers for the objects, by default 0
        """

        """

        assert mass.shape == (obj_num,),"mass: all arrays must be 1d and the same length"
        assert spin.shape == (obj_num,),"spin: all arrays must be 1d and the same length"
        assert spin_angle.shape == (obj_num,),"spin_angle: all arrays must be 1d and the same length"
        assert orb_a.shape == (obj_num,),"orb_a: all arrays must be 1d and the same length"
        assert orb_inc.shape == (obj_num,),"orb_inc: all arrays must be 1d and the same length"
        #assert orb_ang_mom.shape == (obj_num,),"orb_ang_mom: all arrays must be 1d and the same length"
        assert orb_ecc.shape == (obj_num,),"orb_ecc: all arrays must be 1d and the same length" """

        if (obj_num == 0):
            obj_num = mass.shape[0]

        assert mass.shape == (obj_num,), "obj_num must match the number of objects"

        self.mass = mass
        self.spin = spin
        self.spin_angle = spin_angle
        self.orb_a = orb_a
        self.orb_inc = orb_inc
        self.orb_ecc = orb_ecc
        self.orb_arg_periapse = orb_arg_periapse
        self.gen = np.full(obj_num, 1)
        self.id_num = np.arange(id_start_val, id_start_val + obj_num, 1)
        self.galaxy = galaxy
        self.time_passed = time_passed

    def add_objects(self,
                    new_mass=empty_arr,
                    new_spin=empty_arr,
                    new_spin_angle=empty_arr,
                    new_orb_a=empty_arr,
                    new_orb_inc=empty_arr,
                    new_orb_ang_mom=empty_arr,
                    new_orb_ecc=empty_arr,
                    new_orb_arg_periapse=empty_arr,
                    new_gen=empty_arr,
                    new_galaxy=empty_arr,
                    new_time_passed=empty_arr,
                    new_id_num=empty_arr,
                    obj_num=0):
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
        new_galaxy : numpy array
            galaxy iteration to be added, set to -1 if not passed
        new_time_passed : numpy array
            time passed to be added, set to -1 if not passed
        new_id_num : numpy array,optional
            ID numbers to be added
        obj_num : int, optional
            Number of objects to be added.
        """        

        # Make sure all inputs are included
        """
        assert new_mass.shape == (obj_num,),"new mass: all arrays must be 1d and the same length"
        assert new_spin.shape == (obj_num,),"new_spin: all arrays must be 1d and the same length"
        assert new_spin_angle.shape == (obj_num,),"new_spin_angle: all arrays must be 1d and the same length"
        assert new_a.shape == (obj_num,),"new_a: all arrays must be 1d and the same length"
        assert new_inc.shape == (obj_num,),"new_inc: all arrays must be 1d and the same length"
        #assert new_orb_ang_mom.shape == (obj_num,),"new_orb_ang_mom: all arrays must be 1d and the same length"
        assert new_e.shape == (obj_num,),"new_e: all arrays must be 1d and the same length" """

        if (obj_num == 0):
            obj_num = new_mass.shape[0]

        assert new_mass.shape == (obj_num,), "obj_num must match the number of objects"

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
        self.galaxy = np.concatenate([self.galaxy, new_galaxy])
        self.time_passed = np.concatenate([self.time_passed, new_time_passed])

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

        if id_num_remove is None:
            return None
        
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

        if id_num_keep is None:
            return None

        keep_mask = (np.isin(getattr(self, "id_num"), id_num_keep))
        for attr in vars(self).keys():
            setattr(self, attr, getattr(self, attr)[keep_mask])

    def at_id_num(self, id_num, attr):
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

    def to_file(self, fname=None, col_order=None):
        """
        Writes AGNObject to csv file. Header row started with `#` character.

        Parameters
        ----------
        fname : str
            filename including path
        """

        assert fname is not None, "Need to pass filename"

        import pandas
        samples_out = self.return_record_array()
        dframe = pandas.DataFrame(samples_out)
        if col_order is not None:
            dframe = dframe[col_order]
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

        assert fname is not None, "Need to pass filename"

        dat_in = np.genfromtxt(fname, names=True)
        for name in dat_in.dtype.names:
            setattr(self, name, dat_in[name])

    def check_consistency(self):
        """
        Prints the size of each attribute to check that everything is
        consistent with each other.
        """

        # shape of the mass array (arbitrary, just need a comparison value)
        shape = self.mass.shape
        not_consistent = 0
        #print(f"mass.shape == {shape}")
        for attr in vars(self).keys():
            if (getattr(self, attr).shape != shape):
                not_consistent += 1

        if not_consistent > 0:
            print("Attributes are not all the same size!")
            for attr in vars(self).keys():
                print(f"{attr}.shape = {getattr(self, attr).shape}")
            raise AttributeError("attributes are not consistent length")
        else:
            print("attributes are consistent length")


class AGNStar(AGNObject):
    """
    A subclass of AGNObject for single stars. It extends AGNObject by adding
    attributes for mass, radius, and chemical composition.
    """

    def __init__(self,
                 mass=empty_arr,
                 orb_a=empty_arr,
                 orb_inc=empty_arr,
                 star_X=empty_arr,
                 star_Y=empty_arr,
                 star_Z=empty_arr,
                 star_num=0,
                 smbh_mass=None,
                 **kwargs):
        """Creates an instance of the AGNStar class. This is a subclass
           of the AGNObject class. AGNStar adds additional star-specific
           parameters to the AGNObject. It calculates orbital angular
           momentum for stars.

        Parameters
        ----------
        mass : numpy array
            star mass
        orb_a : numpy array
            star orbital semi-major axis with respect to the SMBH
        orb_inc : numpy array
            star orbital inclination with respect to the SMBH
        star_Y : numpy array
            helium fraction of stars
        star_Z : numpy array
            metals fraction of stars
        star_num : int, optional
            number of stars, by default 0
        smbh_mass : float
            mass of the SMBH
        """
        # Make sure all inputs are included
        # if radius is None: raise AttributeError('radius is not included in inputs')
        """ if star_Y is None: raise AttributeError('star_Y is not included in inputs')
        if star_Z is None: raise AttributeError('star_Z is not included in inputs') """

        if (star_num == 0):
            star_num = mass.shape[0]

        assert mass.shape == (star_num,), "star_num must match the number of objects"

        if mass is empty_arr:
            self.radius = empty_arr
            self.orb_ang_mom = empty_arr
        else:
            self.radius = setupdiskstars.setup_disk_stars_radius(masses=mass)
            mass_total = mass + smbh_mass
            mass_reduced = mass*smbh_mass/mass_total
            self.orb_ang_mom = setupdiskstars.setup_disk_stars_orb_ang_mom(star_num=star_num,
                                                                           mass_reduced=mass_reduced,
                                                                           mass_total=mass_total,
                                                                           orb_a=orb_a,
                                                                           orb_inc=orb_inc)

        if (np.any(star_X + star_Y + star_Z > 1.)):
            raise ValueError("star_X, star_Y, and star_Z must sum to 1 or less.")

        self.star_X = star_X
        self.star_Y = star_Y
        self.star_Z = star_Z

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

    def add_stars(self,
                  new_radius=empty_arr,
                  new_X=empty_arr,
                  new_Y=empty_arr,
                  new_Z=empty_arr,
                  star_num=0,
                  **kwargs):
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

        if (star_num == 0):
            star_num = new_radius.shape[0]

        assert new_radius.shape == (star_num,), "star_num must match the number of objects"

        if (np.any(new_X + new_Y + new_Z) > 1.): raise ValueError("new_Y and new_Z must sum to 1 or less")

        self.star_X = np.concatenate([self.star_X, new_X])
        self.star_Y = np.concatenate([self.star_Y, new_Y])
        self.star_Z = np.concatenate([self.star_Z, new_Z])
        self.radius = np.concatenate([self.radius, new_radius])

        super(AGNStar, self).add_objects(obj_num=star_num, **kwargs)


class AGNBlackHole(AGNObject):
    """
    A subclass of AGNObject for single black holes. It extends AGNObject and
    adds attributes for GW frequency and strain. This is only relevant for
    EMRIs and BBH, so if a value is not passed these attributes are set to -1.
    AGNBlackHole also calculates orbital angular momentum for black holes.
    """
    def __init__(self, mass=empty_arr,
                 gw_freq=empty_arr,
                 gw_strain=empty_arr,
                 bh_num=0,
                 **kwargs):
        """Creates an instance of AGNStar object.

        Parameters
        ----------
        mass : numpy array
            black hole masses [Msun]
        gw_freq : numpy array
            gravitational wave frequency [Hz]
        gw_strain : numpy array
            gravitational wave strain [unitless]
        """

        if (bh_num == 0):
            bh_num = mass.shape[0]

        assert mass.shape == (bh_num,), "bh_num must match the number of objects"

        if mass is empty_arr:
            self.orb_ang_mom = empty_arr
            self.gw_freq = empty_arr
            self.gw_strain = empty_arr
        else:
            self.orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(bh_num)

            if ((gw_freq is empty_arr) and (gw_strain is empty_arr)):
                self.gw_freq = np.full(bh_num, -1)
                self.gw_strain = np.full(bh_num, -1)

            elif ((gw_freq is not empty_arr) and (gw_strain is not empty_arr)):
                self.gw_freq = gw_freq
                self.gw_strain = gw_strain
            else:
                raise AttributeError("something messy with gw_freq and gw_strain")

        super(AGNBlackHole, self).__init__(mass=mass, obj_num=bh_num, **kwargs)

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

    def add_blackholes(self,
                       new_mass=empty_arr,
                       new_gw_freq=empty_arr,
                       new_gw_strain=empty_arr,
                       bh_num=0,
                       **kwargs):
        """
        Append black holes to the AGNBlackHole object.

        Parameters
        ----------
        obj_num : int, optional
            number of black holes to be added, by default None
        """

        if (bh_num == 0):
            bh_num = new_mass.shape[0]

        assert new_mass.shape == (bh_num,),"bh_num must match the number of objects"

        if new_gw_freq is empty_arr:
            self.gw_freq = np.concatenate([self.gw_freq, np.full(bh_num, -1)])
        else:
            self.gw_freq = np.concatenate([self.gw_freq, new_gw_freq])
        
        if new_gw_strain is empty_arr:
            self.gw_strain = np.concatenate([self.gw_strain, np.full(bh_num, -1)])
        else:
            self.gw_strain = np.concatenate([self.gw_strain, new_gw_strain])

        super(AGNBlackHole, self).add_objects(obj_num=bh_num, new_mass=new_mass, **kwargs)


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

    def __init__(self,
                 mass_1=empty_arr,
                 mass_2=empty_arr,
                 orb_a_1=empty_arr,
                 orb_a_2=empty_arr,
                 spin_1=empty_arr,
                 spin_2=empty_arr,
                 spin_angle_1=empty_arr,
                 spin_angle_2=empty_arr,
                 bin_sep=empty_arr,
                 bin_orb_a=empty_arr,
                 time_to_merger_gw=empty_arr,
                 flag_merging=empty_arr,
                 time_merged=empty_arr,
                 bin_ecc=empty_arr,
                 gen_1=empty_arr,
                 gen_2=empty_arr,
                 bin_orb_ang_mom=empty_arr,
                 bin_orb_inc=empty_arr,
                 bin_orb_ecc=empty_arr,
                 gw_freq=empty_arr,
                 gw_strain=empty_arr,
                 bin_bh_num=0,
                 id_num=empty_arr):

        """
        Create an instance of AGNBinaryBlackHole

        Parameters
        ----------
        mass_1 : numpy array
            mass of object 1 in Msun
        mass_2 : numpy array
            mass of object 2 in Msun
        orb_a_1 : numpy array
            orbital semi-major axis of object 1 wrt SMBH in R_g
        orb_a_2 : numpy array
            orbital semi-major axis of object 2 wrt SMBH in R_g
        spin_1 : numpy array
            dimensionless spin magnitude of object 1
        spin_2 : numpy array
            dimensionless spin magnitude of object 2
        spin_angle_1 : numpy array
            spin angle of object 1 wrt disk gas in radians
        spin_angle_2 : numpy array
            spin angle of object 2 wrt disk gas in radians
        bin_sep : numpy array
            separation of binary in R_g
            (semi-major axis around center of mass)
        bin_orb_a : numpy array
            semi-major axis of the binary's center of mass wrt SMBH in R_g
            (location in disk)
        time_to_merger_gw : numpy array
            time until binary will merge through GW alone
        flag_merging : numpy array of ints
            flag for if binary is merging this timestep (-2 if merging, 0 else)
        time_merged : numpy array
            time the binary merged (for things that already merged)
        bin_ecc : numpy array
            eccentricity of the binary around the center of mass
        gen_1 : numpy array of ints
            generation of object 1 (1 = natal black hole, no prior mergers)
        gen_2 : numpy array of ints
            generation of object 2 (1 = natal black hole, no prior mergers)
        bin_orb_ang_mom : numpy array
            angular momentum of the binary wrt SMBH (+1 prograde / -1 retrograde)
        bin_orb_inc : numpy array
            orbital inclination of the binary wrt SMBH
        bin_orb_ecc : numpy array
            orbital eccentricity of the binary wrt SMBH
        gw_freq : numpy array
            GW frequency of binary
            nu_gw = 1./pi * sqrt(G * M_bin / bin_sep^3)
        gw_strain : numpy array
            GW strain of binary
            h = (4/d_obs) *(GM_chirp/c^2)*(pi*nu_gw*GM_chirp/c^3)^(2/3)
            where m_chirp =(M_1 M_2)^(3/5) /(M_bin)^(1/5)
            For local distances, approx 
            d=cz/H0 = 3e8m/s(z)/70km/s/Mpc =3.e8 (z)/7e4 Mpc =428 Mpc
            assume 1Mpc = 3.1e22m.
            From Ned Wright's calculator
            (https://www.astro.ucla.edu/~wright/CosmoCalc.html)
            (z=0.1)=421Mpc. (z=0.5)=1909 Mpc


        """

        # Assign attributes
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.orb_a_1 = orb_a_1
        self.orb_a_2 = orb_a_2
        self.spin_1 = spin_1
        self.spin_2 = spin_2
        self.spin_angle_1 = spin_angle_1
        self.spin_angle_2 = spin_angle_2
        self.bin_sep = bin_sep
        self.bin_orb_a = bin_orb_a
        self.time_to_merger_gw = time_to_merger_gw
        self.flag_merging = flag_merging
        self.time_merged = time_merged
        self.bin_ecc = bin_ecc
        self.gen_1 = gen_1
        self.gen_2 = gen_2
        self.bin_orb_ang_mom = bin_orb_ang_mom
        self.bin_orb_inc = bin_orb_inc
        self.bin_orb_ecc = bin_orb_ecc
        self.gw_freq = gw_freq
        self.gw_strain = gw_strain
        self.id_num = id_num

        if (bin_bh_num == 0):
            bin_bh_num = mass_1.shape[0]

    def __repr__(self):
        return ('AGNBinaryStar(): {} black hole binaries'.format(len(self.mass)))
    
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
             1 : "single star",}
             #2 : "binary black hole",
             #3 : "binary star",
             #4 : "exploded star"
             #} # Other types are not in use yet

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

        # totals = "AGN Filing Cabinet\n"
        # for key in obj_types:
        #     #print(key,getattr(self,"category").count(key))
        #     totals += (f"\t{obj_types[key]}: { np.sum(getattr(self,"category") == key) }\n")
        #     for direc in obj_direction:
        #         totals += (f"\t\t{obj_direction[direc]}: {np.sum((getattr(self,"category") == key) & (getattr(self,"direction") == direc))}\n")
        #     totals += "\n"
        #     for loc in obj_disk_loc:
        #         totals += (f"\t\t{obj_disk_loc[loc]}: {np.sum((getattr(self,"category") == key) & (getattr(self,"disk_inner_outer") == loc))}\n")
        # totals += f"{len(getattr(self,"category"))} objects total"
        return()

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
