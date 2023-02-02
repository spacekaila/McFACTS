def ReadInputs():

    # ReadInputs.py
    # This reads and parses the inputs from model_choice.txt
    # should eventually change to specify infilename generically

    # create a dictionary to store numerical variables in
    input_variables = {}

    # open the main input file for reading
    model_inputs = open("model_choice.txt", 'r')

    # go through the file line by line
    for line in model_inputs:
        line = line.strip()
        # If it is NOT a comment line
        if (line.startswith('#') == 0):
            # split the line between the variable name and its value
            varname, varvalue = line.split("=")
            # remove whitespace
            varname = varname.strip()
            varvalue = varvalue.strip()
            # if the variable is the one that's a filename for the disk model, deal with it
            if (varname == 'disk_model_name'):
                disk_model_name = varvalue.strip("'")
            # otherwise, typecast to float and stick it in the dictionary
            else:
                input_variables[varname] = float(varvalue)
 
    # close the file
    model_inputs.close()

    print("I read the main input file and closed it")

    # Recast the inputs from the dictionary lookup to actual variable names
    #   !!!is there a better (automated?) way to do this?
    mass_smbh = input_variables['mass_smbh']
    trap_radius = input_variables['trap_radius']
    n_bh = input_variables['n_bh']
    mode_mbh_init = input_variables['mode_mbh_init']
    max_initial_bh_mass = input_variables['max_initial_bh_mass']
    mbh_powerlaw_index = input_variables['mbh_powerlaw_index']
    mu_spin_distribution = input_variables['mu_spin_distribution']
    sigma_spin_distribution = input_variables['sigma_spin_distribution']
    spin_torque_condition = input_variables['spin_torque_condition']
    frac_Eddington_ratio = input_variables['frac_Eddington_ratio']
    max_initial_eccentricity = input_variables['max_initial_eccentricity']
    timestep = input_variables['timestep']
    number_of_timesteps = input_variables['number_of_timesteps']

    print("I put your variables where they belong")

    # open the disk model surface density file and read it in
    # Note format is assumed to be comments with #
    #   density in SI in first column
    #   radius in r_g in second column
    #   infile = model_surface_density.txt, where model is user choice
    infile_suffix = '_surface_density.txt'
    infile = disk_model_name+infile_suffix
    surface_density_file = open(infile, 'r')
    density_list = []
    radius_list = []
    for line in surface_density_file:
        line = line.strip()
        # If it is NOT a comment line
        if (line.startswith('#') == 0):
            columns = line.split()
            density_list.append(float(columns[0]))
            radius_list.append(float(columns[1]))
    # close file
    surface_density_file.close()

    # re-cast from lists to arrays
    surface_density_array = np.array(density_list)
    disk_model_radius_array = np.array(radius_list)

    # open the disk model aspect ratio file and read it in
    # Note format is assumed to be comments with #
    #   aspect ratio in first column
    #   radius in r_g in second column must be identical to surface density file
    #       (radius is actually ignored in this file!)
    #   filename = model_aspect_ratio.txt, where model is user choice
    infile_suffix = '_aspect_ratio.txt'
    infile = disk_model_name+infile_suffix
    aspect_ratio_file = open(infile, 'r')
    aspect_ratio_list = []
    for line in aspect_ratio_file:
        line = line.strip()
        # If it is NOT a comment line
        if (line.startswith('#') == 0):
            columns = line.split()
            aspect_ratio_list.append(float(columns[0]))
    # close file
    aspect_ratio_file.close()

    # re-cast from lists to arrays
    aspect_ratio_array = np.array(aspect_ratio_list)

    # Housekeeping from input variables
    disk_outer_radius = disk_model_radius_array[-1]
    disk_inner_radius = disk_model_radius_array[0]

    print("I read and digested your disk model")

    return
