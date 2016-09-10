""" CONTENT

SIMULATION (= simulation argument):
    (VAR) Simulation variables:
        GL[O]BS
            class SimulationVariables
            class SimulatonAttributes
    (SIM) Simulation program:
        SIMU[L]ATION
        [R]UN
            class Simulation
        SHOW [Y]
    (SIM) Logic:
        [S]IMULATION FUNCTIONS
    (VAR) Modifiers:
        [D]PRIME MAPS
        [M]ASK
        [I]BO
        [B]OOST IBO
        [E]LM
    (IN) Simulation attributes loader:
        (DELETED) I[N]ITIALIZATION
        [A]RGS LOAD
            def dprime_fnc_from_str
        CONFIGURATION [F]ILE LOAD
            class AttributesSkeleton
    (IN) On-line user interaction:
        ONLINE INP[U]TS
    (OUT) Visualization of a run:
        [V]ISUALIZATION
        [W]RITERS
    Data modifiers:
        [C]ONVERTORS
        MESH GRID [X]
    Other:
        [T]EST FUNCTIONS

ANALYZE (=analysis of simulation datas):
    Analysis program:
        ANALYZE [J]
        ANA PROGRAM
    (ANA) Analyze run:
        ANA LOADS
        DATA ANALYSIS
        TO PLOTABLES [K]
    (OUT) Analyze output:
        ANA SAVE [X]
        ANA PLOTS
    Other:
        ANA OTHER

SHARED (=functions used by all modules):
    [G]RAPHICAL OUTPUT

PARAMETERS:
    [P]ARAMETRS PARSER

MAIN [Z]

Free:
    (Q)
"""

import math
import cmath
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.interpolate
import scipy.io
import ctypes
import random
import collections
import string
import copy
import time
import os.path

#? Consider wheter delete followint constant variables
SMALL = np.array([(0.0,0.0), (-4.0,0.0), (4.0,0.0), (0.0,-4.0), (0.0,4.0)])
REAL = np.vstack((scipy.io.loadmat("locations_deg.mat")['locations'],np.zeros(2)))
LOCATIONS = REAL
START = (0,0)
CRITERION = 0.95

### GL[O]BS ###
class SimulationVariables:
    """ Simulation variables

    Stores variables used by the simulation. The variables can be divided into two
    types. User defined, wich are not modified through simulation run and are 
    chosen by user, and volatile, wich are used by simulation, i.e. are modified by
    simulation.

    Variables:

        Varialbes used in the simulation. The varialbes have two main group. The variables
        wich specify the simulation run, and the variables used for storing simulation datas.

        Const variables:
            This variables arn't modified in simulation run.

            senzory_map (np.array):
                The matrix of Nx2. A row equals to a location. The locations are
                lexicographicly sorted from top left to bottom right.
            number_of_locs (integer):
                The number of the senzory locations. It is used instead of
                Senozry_map.shape[0] for convinience.
            dprime_map (np.array of POSITIVE float):
                The cell (i,j) contains dprime value at the location j for the location i
                as the center. Indeces of locations are ordered same as the locations
                in Senzory_map.
            dprime_fnc (function):
                The function used to generate Dprime_map.
            next_fixation (function):
                The function used for finding next fixation point. It is a core of the program.
                It can be e.g. ibo, elm ...
            threshold (float in [0,1]):
                Threshold used as a stopping criterion for the serach. If a postreio probability
                exceeds the threshold, the simulation is stopped.
            num_of_searches (int positive):
                Determines number of target searches.
                
        Volatile variables
            This variables stores datas, wich are modified by simulation. A varibale of type np.array
            stores datas for senzory locations. A data in np.array corresponds to a senzory location
            in Senzory_map with same index.

            visual_field (np.array):
                Stores current senzory input.
            weighted_sums (np.array):
                Stores current weighted sums used for computing posterior probabilities.
            prior_prob (double):
                Prior probability for target at a senzory location.
            post_probs (np.array)
                Current posterior probabilities for senzory locations.
            target_location (int)
                Location of the target.
            focus := int;
                Location of the current fixation.

    Disclaimer:
        Before starting a simulation, initialize volatile variables thorough initialize
        method.
    """
    def __init__(self, simulation_attributes):
        """
        Initialize simulation variables from an instance of SimulationAttributes
        Copy of attributes from simulation_attributes is SHALLOW. So be ware with
        the locations attribute in simulation_attributes.

        Args:
            simulaton_attriubtes (SimulationAttributes)
        Disclaimer:
            simulation_attributes can't contain None attributes. Excpect of id_name.
            Checked through assert.
        """
        for attr in ['locations','dprime_fnc','next_fixation',
            'threshold', 'num_of_searches']:
            if getattr(simulation_attributes,attr) is None:
                assert False, (
                    "Precondition violation: none attribute in simulation_attributes "
                    + attr
                )
        if not isinstance(simulation_attributes, SimulationAttributes):
            raise TypeError(
                "The argument isn't an instance of SimulationAttributes class"
            )
        self.senzory_map = self._locations_to_senzory_map(
            simulation_attributes.locations
        )
        self.number_of_locs = self.senzory_map.shape[0]
        self.dprime_fnc = simulation_attributes.dprime_fnc
        self.dprime_map = generate_dprime_map(self.dprime_fnc,self.senzory_map)
        self.next_fixation =  simulation_attributes.next_fixation
        self.threshold = simulation_attributes.threshold
        self.num_of_searches = simulation_attributes.num_of_searches

    def initialize(self):
#TODO: choose user defined START position
        """
        Initialize volatile simulation variables
        """
        values_type = np.dtype(float)
        self.visual_field = np.zeros(self.number_of_locs, dtype=values_type)
        self.weighted_sums = np.zeros(self.number_of_locs, dtype=values_type)
        self.prior_prob = 1.0 / np.prod(self.number_of_locs)
        self.post_probs = np.full(
            self.number_of_locs, self.prior_prob, dtype=values_type
        )
        starting_location = np.array(START)
        self.focus = get_index_of_in(starting_location,self.senzory_map)
        self.target_location = [
            x for x in xrange(self.number_of_locs) if x != self.focus
        ][random.randint(0,self.number_of_locs-2)]

    def _locations_to_senzory_map(self,locations):
        """
        Convert locations to senzory map used by simulation. Senzory map is
        locations witch (0,0) position and sorted from top left to bottom right
        corner in cartesian coordinate system.
        """
        if not np.any([np.array_equal(row,[0.,0.]) for row in locations]):
#? Shallow copy?
#A No. np.vstack returns new array
            locations = np.vstack((locations,[0.,0.]))
        return self._sort_locations(locations)

    def _sort_locations(self,locations):
        """ Sort locations from top left to bottom right

        Args:
            locations(np.array): matrix where a row is a location
        Returns:
            (np.array) sorted 2d matrix with same shape as locations
        """
        i = np.lexsort(np.transpose(locations*np.array((1,-1))))
        return locations[i]

def get_index_of_in(vector,matrix):
    """ Get an index of a row in a 2D matrix wich equels to the vector

    Args:
        vector(np.array): 1D np.array
        matrix(np.array): 2D np.array
    PreConds:
        vector.shape[0] == matrix.shape[1]
    Return:
        (int) index as integer
    """
    assert (vector.shape[0] == matrix.shape[1]), "Precondition violation"
    return np.nonzero(
        np.prod(
            matrix == np.tile(vector,(matrix.shape[0],1)),
            axis=1
        )
    )[0][0]

class SimulationAttributes(object):
    """ Manages simulation attributes

    It serves as interface between user input attributes and
    variables used by the simulation. For the use in the simulation,
    perform initialize method to generate all necessary variables.
    The variables are generated from:
        locations, dprime_fnc, next_fixation, threshold, num_of_searches,
    wich have to be assigned before the initilization of whole class.

    Methods:
        Setters/Geters (for locations, dprime_fnc, next_fixation,
            threshold, num_of_searches):
            The setters support initialization through string. 
    """
    def __init__(self,name=None):
        self._locations = None
        self._dprime_fnc = None
        self._next_fixation = None
        self._threshold = None
        self._num_of_searches = None
        if name == None:
            self._id_name = name
        else:
            self.id_name = name
        self._analyses = []
            
    @property
    def locations(self):
        return self._locations

    @locations.setter
    def locations(self,val):
        self._locations = val
    
    @property
    def dprime_fnc(self):
        return self._dprime_fnc

    @dprime_fnc.setter
    def dprime_fnc(self,val):
        """
        Args:
            val (string): dprime function specification
        """
        if val == None:
            self._dprime_fnc = None
#? We needed to change code. The attribte can be set by function value.
        elif hasattr(val,'__call__'):
            self._dprime_fnc = val
        elif isinstance(val, basestring):
            self._dprime_fnc = dprime_fnc_from_str(val)
        else:   
            raise TypeError("{} isn't string.".format(val))

    @property
    def next_fixation(self):
        return self._next_fixation

    @next_fixation.setter
    def next_fixation(self,val):
        """
        Args:
            val (string): in ['ibo','elm','cibo']
        """
        if val == None:
            self._next_fixation = None
        elif hasattr(val,'__call__'):
            self._next_fixation = val
        elif isinstance(val, basestring):
            self._next_fixation = str_to_next_fixation_fnc(val)
        else:
            raise TypeError("{} isn't string".format(val))

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self,val):
        """
        Args:
            val (string or float): in the unit interval [0,1]
        """
        if val == None:
            self._threshold = None
        elif isinstance(val, basestring):
            self._threshold = str_to_unit_interval_float(val)
        else:
            if val < 0 or val > 1:
                raise ValueError("{} isn't float in [0,1]".format(val))
            self._threshold = val

    @property
    def num_of_searches(self):
        return self._num_of_searches

    @num_of_searches.setter
    def num_of_searches(self,val):
        """
        Args:
            val (string or int): positive int
        """
        if val == None:
            self._num_of_searches = None
        elif isinstance(val, basestring):
            self._num_of_searches = str_to_positive_int(val)
        elif isinstance(val,int):
            if val < 1:
                raise ValueError("{} isn't positive int".format(val))
            self._num_of_searches = val
        else:
            raise TypeError("{} isn't string/int.".format(val))

    @property
    def id_name(self):
        return self._id_name

    @id_name.setter
    def id_name(self,val):
        """
        Args:
            val (string): id name
        """
        if val == None:
            self._id_name = None
            return
        if not isinstance(val, basestring):
            raise TypeError("{} isn't string.".format(val))
        self._id_name = val

    @property
    def analyses(self):
        return self._analyses

    @analyses.setter
    def analyses(self,val):
        raise NotImplemented("Use add_analysis")

    def add_analysis(self,val):
        self._analyses.append(AnalysisAttributes(val))

#DEBUG
    def __str__(self):
        return "loc:{0}\ndfnc:{1}\nnf:{2}\nth:{3}\nns:{4}\nname:{5}".format(
            self.locations,
            self.dprime_fnc,
            self.next_fixation,
            self.threshold,
            self.num_of_searches,
            self.id_name
        )
#END DEBUG

#? Is using fill a good programming?
#A Now we can move the function outside, because there is support for None assignment
    def fill(self, simulation_attributes):
        """
        Fill None attributes in self with the attributes from simulation_attributes.
        Filling with the attributes from simulation_attributes is performed throght
        SHALLOW copy.
        Only for id_name attribute, there is different rule for filling. Concatenation
        of id_name from self and id_name from simulation_attributes is used.

        Args:
            simulation_attributes (SimulationAttributes)
        Disclaimer:
            After fill, the instance should be use read-only. Other uses arn't
            preperly defined, because shallow copy is used.
        """
        if not isinstance(simulation_attributes,SimulationAttributes):
            raise TypeError("The argument isn't instance of SimulationAttributes")
        for attribute in ['_locations', '_dprime_fnc', '_next_fixation',
            '_threshold', '_num_of_searches']:
            if getattr(self,attribute) is None:
                setattr(self,attribute,getattr(simulation_attributes,attribute))
        if simulation_attributes.id_name is not None:
            if self._id_name is not None:
                self._id_name += '_' + simulation_attributes.id_name
            else:
                self._id_name = simulation_attributes.id_name

### SIMU[L]ATION ###
def simulation():
#TODO: Handle errors.
    """ Handles simulation commandline option

    Simulation loads SimulationAttributes as defined in simulation program options and
    in cofiguration file. Then it creates SimulationAttributes generator and passes it
    to apropriete subprogram, i.e. run or analyze.

    AccsGlobs:
        Args
    """
    verbose_print("Start simulation attributes generation.",1)
    generator = input_simulation_attributes_generator()
    verbose_print("Simulation attributes generated.",1)
    Args.simulation_function(generator)

### [R]UN ###
def run(sim_attr_generator):
#TODO: clean
#TODO: integrate analyses
    """ Run subprogram of simulation program

    It iterate through SimulationAttributes in sim_attr_generator and
    run a simulation as specified in SimulationAttributes

    Args:
        sim_attr_generator (forward iterable object with SimulationAttributes)
    """
    def analyze_and_save(simulation,simulation_attributes):
#? Ugly conf file analyses integration.
        if simulation_attributes.analyses and Args.output_file != None:
            verbose_print("Saving analyses for {0}.".format(simulation_attributes.id_name),2)
            results = analyze_datas(
                simulation.result,
                simulation_attributes.analyses
            )
            plotables  = ana_results_to_plotables(
                results,
                simulation_attributes.analyses
            )
#TODO error handling for save
            analysis_save_dm(
                results,
                plotables,
                simulation_attributes.analyses,
                simulation_attributes.id_name
            )

    def save_simulation(simulation,simulation_attributes):
        if not simulation_attributes.analyses and Args.output_file != None:
            verbose_print("Saving simulation datas of {0}.".format(
                simulation_attributes.id_name
            ),2) 
            try:
                np.save(
                    simulation_attributes.id_name,
                    simulation.result
                )
            except:
                raise EnvironmentError("Can't save data to {}.".format(
                    simulation_attributes.id_name
                ))

    verbose_print("Starting simulation run.",1)
    for i,simulation_attributes in enumerate(sim_attr_generator):
        verbose_print("Starting simulation number {0}: {1}".format(
            i,
            simulation_attributes.id_name
        ),2)
        simulation = Simulation(
            SimulationVariables(simulation_attributes)
        )
        simulation.start()
        save_simulation(simulation,simulation_attributes)
        analyze_and_save(simulation,simulation_attributes)
#? ELIF?

#? What about to join SimulationVariables and Simulation class?
#? What is better programming practise, join Simulation with functions
#? in  SIMULATION FUNCTIONS?
class Simulation():
    """ Simulation for one SimulationVariables
    
    Hendles one simulation as specified in simulation_varibles argument of __init__.
    It composes a computation, an output and an input parts of the program.
    Result of the simulation is in the attribute result. 'result' is dictionary
    and has this structure:
        'senzory_map': used senzory locations in the simulation
        'seraches': results for every search for a target
            L '0', '1', '2', ...: result  for a serach
                L 'path': fixation path. It is an np.array. Fications are stacked
                    verticaly.

    AccsGlobs:
        Args
    """

    MSG_NEXT_FIXATION =  "Press Enter to compute next fixation (Q for quit) ..."
    MSG_NEXT_SEARCH = "Press Enter to start search (Q for quit) ..."

    def __init__(self,simulation_variables):
        self.vars = simulation_variables
        self._fixation_num = None
        self._search_num = None
        self._searches = None
        self._path = None
    
    def start(self):
        self.verbose_before_simulation()
        self.save_simulation_init()
        for self._search_num in xrange(self.vars.num_of_searches):
            try:
                self.search()
            except self.EndSearch:
                continue
            except self.EndSimulation:
                break
        self.save_simulation()
        self.verbose_after_simulation()

    def verbose_before_simulation(self):
        pass

    def verbose_after_simulation(self):
        pass
   
    def save_simulation_init(self):
        self._searches = dict()

    def save_simulation(self):
#? Save answered target locations?
        self.result = {
            'senzory_map': self.vars.senzory_map,
            'searches': self._searches
        }

    def search(self):
        self.vars.initialize()
        self.verbose_before_search()
        self.save_search_init()
        while True:
            try:
                self.saccade()
            except self.Found:
                break
        self.save_search()
        self.verbose_after_search()
        
    def verbose_before_search(self):
        if Args.interactive and input_is("Q",self.MSG_NEXT_SEARCH):
            raise self.EndSimulation()
        if Args.verbose > 2:
            sys.stdout.write(B_style)
            sys.stdout.write("Starting a new search num. {}\n\n".format(self._search_num+1))
            sys.stdout.write(End_style)
        self._fixation_num = 0

    def verbose_after_search(self):
        if Args.verbose > 2:
            self._fixation_num += 1
            print_verbose("End",self.vars)

    def save_search_init(self):
        self._path = np.zeros(shape=(1,2))

    def save_search(self):
        self._searches[str(self._search_num)] = {
            'path':self._path
        }
            
    def saccade(self):
        self.verbose_before_saccade()
        self.saccade_logic()
        self.save_saccade()
        self.verbose_after_saccade()

    def verbose_before_saccade(self):
        if Args.verbose > 2:
            self._fixation_num += 1
            print_verbose(self._fixation_num,self.vars)
        if Args.interactive and input_is("Q",self.MSG_NEXT_FIXATION):
            raise self.EndSearch()
        if Args.timer > 0:
            time.sleep(Args.timer)

    def verbose_after_saccade(self):
        pass

    def save_saccade(self):
        self._path = np.vstack(
            (self._path,self.vars.senzory_map[self.vars.focus])
        )

    def saccade_logic(self):
        update_visual_field(self.vars)
        update_posterior_probs(self.vars)
        if threshold_met(self.vars):
            raise self.Found()
        self.vars.next_fixation(self.vars)

    class End(Exception):
            pass

    class EndSimulation(End):
        pass

    class EndSearch(End):
        pass

    class Found(End):
        pass

### SHOW [Y] ###
def show_data(sim_attr_generator):
#TODO description
    """ Show internal globs of the simulation

    The function handles show part of simulation program. It shows internal
    global variables to user.
    Implemented show:
        dprime

    GlobAccs:
        Args
    """
    if Args.data_to_show == 'dprime':
        show_dprime(sim_attr_generator)

def show_dprime(sim_attr_generator):
#TODO description
    """ Show dprime

    The function generates mesh grids from dprime. The mesh grids are plotted or
    saved to .mat file as specified by user.

    GlobAccs:
        Args
    """
    dprime_fnc_list = [
        (sim_attr.id_name,sim_attr.dprime_fnc) for sim_attr in sim_attr_generator
    ]

    if Args.mat_file_out != None:
        save_dict = dict()
    else:
        x_axis = int(math.ceil(math.sqrt(len(dprime_fnc_list))))
        y_axis = int(math.ceil(float(len(dprime_fnc_list)) / x_axis))
        fig, axes = plt.subplots(nrows=y_axis,ncols=x_axis)

#? Code duplication
    if len(dprime_fnc_list) == 1:
        id_name, dprime_fnc = dprime_fnc_list[0]
        mesh_X, mesh_Y, mesh_Z = dprime_fnc_to_mesh_grid(
            dprime_fnc, linspace=Args.grid_size
        )
        im = show_plot_imshow_from_mesh(
            axes, mesh_X, mesh_Y, mesh_Z, title=id_name, vmax=Args.upper_bound
        )
        fig.colorbar(im,shrink=0.8)
        plt.show()
# End code duplication
        return

    for i, (id_name, dprime_fnc) in enumerate(dprime_fnc_list):
        mesh_X, mesh_Y, mesh_Z = dprime_fnc_to_mesh_grid(
            dprime_fnc, linspace=Args.grid_size
        )
        if Args.mat_file_out != None:
            dprime_fnc[id_name] =  {'X':mesh_X, 'Y':mesh_Y, 'Z':mesh_Z}
        else:
            im = show_plot_imshow_from_mesh(
                axes.flat[i], mesh_X, mesh_Y, mesh_Z, title=id_name, vmax=Args.upper_bound
            )
    if Args.mat_file_out != None:
        scipy.io.savemat(Args.mat_file_out, save_dict)
    else:
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.8)
        plt.show()

#? Move to GRAPHICAL part
def show_plot_imshow_from_mesh(ax,X,Y,Z,title=None,vmax=None):
#TODO description
    min_x, min_y = np.min(X), np.min(Y)
    max_x, max_y = np.max(X), np.max(Y)
    im = ax.imshow(
        Z, extent=(min_x,max_x,min_y,max_y), aspect='equal',
        vmax=vmax, origin='lower')
    ax.set_title(title)
    return im

### [S]IMULATION FUNCTIONS ###
def update_visual_field(vars_):
    """ Update visual field for current focus
    
    When the fovea is shifted, the function is used to obtain senzory information
    for the current focus, i.e. it generates new senzory input for current focus
    
    Args:
        vars (SimulationVariables)
    """
    vars_.visual_field = np.random.normal(size=vars_.number_of_locs)
    vars_.visual_field = vars_.visual_field / vars_.dprime_map[vars_.focus] - 0.5
    vars_.visual_field[vars_.target_location] += 1.0
    
def update_posterior_probs(vars_):
    """ Update posterior probabilities

    Compute posterior probabilities for newly generated visual field

    Args:
        vars (SimulationVariables)
    """
    vars_.weighted_sums += np.power(vars_.dprime_map[vars_.focus],2) * vars_.visual_field
    vars_.post_probs = np.exp(vars_.weighted_sums) * vars_.prior_prob
    vars_.post_probs /= np.sum(vars_.post_probs)

def threshold_met(vars_):
    """ Checking termination criterion

    Args:
        vars (SimulationVariables)
    """
    return np.max(vars_.post_probs) > vars_.threshold

### [D]PRIME MAPS ###
Dprime_types = ['no', 'scotoma_hard', 'scotoma_soft',
    'glaucoma_hard', 'glaucoma_soft',
    'hemianopsia_left_hard', 'hemianopsia_left_soft',
    'hemianopsia_right_hard', 'hemianopsia_right_soft']

def generate_dprime_map(dprime_fnc, senzory_map):
    """ Generate dprime map for senzory locations using specified function

    The function generate matrix of dprime values for all possible locations
    using specified function.

    Args:
        dprime_fnc(functino): the function used to compute dprime values.
            The function specification:
                dprime_fnc(distance)
                Computes dprime for the distance. The distance is a (2,) np.array.
        senzory_map(np.array): 2D array of senzory locations. The locations are stored
            in rows
    Returns:
        (np.array) 2D np.array. The value at (i,j) is a dprime value at the location j if the location
            i is at the focus.
    PreConds:
        senzory_map.shape(1) == 2
        senzory_map.shape(0) > 0
    """
    assert (senzory_map.shape[1] == 2), "Precondition violation"
    assert (senzory_map.shape[0] > 0), "Precondition violation"
    locations = senzory_map.shape[0]
    dprime = np.zeros((locations,locations))
    for center in xrange(locations):
        for look in xrange(locations):
            distance = senzory_map[look] - senzory_map[center]
            dprime[center,look] = max( # avoid zero dprime
                dprime_fnc(distance),
                0.0000001
            )
    return dprime


#TODO: using global variables?
E_t, E_s, E_n, E_i = 2.4242, 2.1624, 2.8751, 2.8751
Beta_e = 2.9649
def dprime_basic(distance,d0 = 4.0,e=[E_t,E_s,E_n,E_i],beta_e=Beta_e):
    """ D_prime map function

    Args:
        distance (double pair (dx,dy)): the distance from the focus
            along x and y axis
        d0 (double): the fovea senzitivity
        e (double (e_t,e_s,e_n,e_i)): eccentricities in cardinal
            directions
        beta_e (double): fallof exponent
    Returns:
        (double) dprime senzitivity
    PreConds:
        distance.shape == (2,)
    Raises:
        NotImplementedError
    """
    assert (len(distance) == 2), "Precondition violation"
    x, y = distance[0], distance[1]
    if x >= 0 and y >= 0:
        e_x, e_y = e[0], e[1]
    elif x < 0 and y >= 0:
        e_x, e_y = e[2], e[1]
    elif x < 0 and y < 0:
        e_x, e_y = e[2], e[3]
    elif x >= 0 and y < 0:
        e_x, e_y = e[0], e[3]
    else:
        raise NotImplementedError()
    return d0 / (1.0 + ((x/e_x)**2 + (y/e_y)**2)**(beta_e/2))

def dprime_fnc(distance,d0=4,beta=1.51,beta_e=2.29,
    cb2_h=0.1,c_h=0.005,e0_h=[3.43,1.82,2.43,2.81],
    cb2_l=0,c_l=0.0029,e0_l=[3.58,3.06,3.68,2.59]):
    def K1(x,y):
        return (
            (1.0/cb2_h)
            *
            (
                c_h**2 / dprime_basic((x,y),d0,e0_h,beta_e)**(2/beta)
                -
                c_l**2 / dprime_basic((x,y),d0,e0_l,beta_e)**(2/beta)
            )
        )

    def K2(x,y):
        return c_l**2 / dprime_basic((x,y),d0,e0_l,beta_e)**(2/beta)

    x,y = distance
    return (c_h / math.sqrt(K1(x,y) * cb2_h + K2(x,y)))**beta

### [M]ASK ###
SMALL_FLOAT = 0.001

def no_mask(d0=1.0,dprime_fnc=dprime_fnc):
    def fnc(distance,d0_=d0):
        return dprime_fnc(distance,d0_)
    return fnc

def scotoma_hard(a=2.5,b=2.5,dprime_fnc=dprime_basic):
    """ Returns function witch computes dprime for an fovea with scotoma

    Args:
        a,b (float): a,b are semi-major and semi-minor axes
        dprime_fnc(function): the functio wich is maked by this function
            the function specification:
                dprime_fnc(distance):
                    Computes dprime value for distance. The distance is (2,) np.array.
    Return:
        (functio) The function has same spicification as dprime_fnc
    """
    def scotoma_fnc(distance):
        if not in_ellipse(distance[0],distance[1],a,b):
            return dprime_fnc(distance)
        else:
            return SMALL_FLOAT
    return scotoma_fnc

def scotoma_soft(a=2.5,b=2.5,slope=5.0,dprime_fnc=dprime_basic):
    """ Returns function witch computes dprime for an fovea with scotoma

    The function differs from scotoma_hard by a soft border between a visualy
    intact part and the scotoma. Sigmoidal function is used for soft transition.

    For the specification look at scotoma_hard
    +Args:
        slope (float): the slope for sigmoidal function. Bigger number closer to
            hard delimiting function
    """
    def scotoma_fnc(distance):
        return dprime_fnc(distance) * sigmoid(ellipse(distance[0],distance[1],a,b)-1,slope)
    return scotoma_fnc

def glaucoma_hard(a=2.5,b=2.5,dprime_fnc=dprime_basic):
    """ Returns function witch comptes dprime for an fovea dameged by glaucoma

    Glaucoma can cause tunel vision, where the vision is preserev only in cetnral part of
    virual field.

    Args:
        a,b (float): a,b are semi-major and semi-minor axes
        dprime_fnc(function): the functio wich is maked by this function
            the function specification:
                dprime_fnc(distance):
                    Computes dprime value for distance. The distance is (2,) np.array.
    Return:
        (functio) The function has same spicification as dprime_fnc
    """
    def scotoma_fnc(distance):
        if in_ellipse(distance[0],distance[1],a,b):
            return dprime_fnc(distance)
        else:
            return SMALL_FLOAT
    return scotoma_fnc

def glaucoma_soft(a=2.5,b=2.5,slope=5.0,dprime_fnc=dprime_basic):
    """ Returns function witch computes dprime for an fovea dameged by glaucoma

    The function differs from glaucoma_hard by a soft border between a visualy
    intact part and the scotoma. Sigmoidal function is used for soft transition.

    For the specification look at scotoma_hard
    +Args:
        slope (float): the slope for sigmoidal function. Bigger number closer to
            hard delimiting function
    """
    def scotoma_fnc(distance):
        return dprime_fnc(distance) * sigmoid(-1*(ellipse(distance[0],distance[1],a,b)-1),slope)
    return scotoma_fnc

def hemianopsia_hard(hemi='left',dprime_fnc=dprime_basic):
    """ Returns function witch computes dprime for an fovea with homonymous hemianopsia

    Homonymous hemianopsia cases blindness in a half of visual field of both eyes.
    For more info look at https://en.wikipedia.org/wiki/Homonymous_hemianopsia.

    Args:
        hemi (LEFT, RIGHT): choose wich hemifield will be blind
        dprime_fnc(function): the functio wich is maked by this function
            the function specification:
                dprime_fnc(distance):
                    Computes dprime value for distance. The distance is (2,) np.array.
    Return:
        (functio) The function has same spicification as dprime_fnc
    """
    def hemianopsia_fnc(distance):
        if (hemi == 'left' and distance[0] < 0) or (hemi == 'right' and distance[0] > 0):
            return SMALL_FLOAT
        else:
            return dprime_fnc(distance)
    return hemianopsia_fnc

def hemianopsia_soft(hemi='left',slope=5.0,dprime_fnc=dprime_basic):
    """ Returns function witch computes dprime for an fovea dameged by glaucoma

    The function differs from hemianopsia_hard by a soft border between a visualy
    intact part and the scotoma. Sigmoidal function is used for soft transition.

    For the specification look at hemianopsia_hard
    +Args:
        slope (float): the slope for sigmoidal function. Bigger number closer to
            hard delimiting function
    """
    def hemianopsia_fnc(distance):
        if (hemi == 'left'):
            return dprime_fnc(distance) * sigmoid(distance[0],slope)
        elif (hemi == 'right'):
            return dprime_fnc(distance) * sigmoid(-distance[0],slope)
        else:
            assert False, "Unimplemented branch for the argument bling"
    return hemianopsia_fnc

def in_ellipse(x,y,a,b):
    """ Disides whether the point (x,y) is inside the epllipse (a,b)

    Args:
        x,y,a,b (float): x,y is coordinate of the point
            a,b semi-major and semi-minor axes
    Returns:
        (bool) True if the point is inside the ellipse
    """
    return ellipse(x,y,a,b) <= 1

def ellipse(x,y,a,b):
    """ Compute left part of the ellipses equation

    The function computes (x/a)**2 + (y/b)**2.

    Args:
        x,y,a,b (float): x,y are a coordinate of the point
            a,b are semi-major and semi-minor axes
    Returns:
        (float) the result of the equation
    """
    return ((x/float(a))**2 + (y/float(b))**2)

def sigmoid(x,slope=1):
    return 1 / (1 + math.exp(max(min(-slope * x,100),-100)))

### [I]BO ###
#TODO global -> vars_
def ibo():
    """ Choose next fixation

    For next fixation it is picked up a position, wich maximize an information gain.
    The function implements the IBO algorithm from

    J.Najemnik, W.S.Geisler: Eye movement statistics in humans are
        consistent with an optimal search strategy. 2009.

    AccsGlobs:
        Number_of_locs
    """
    global Focus #MOD Focus deleted
    best_location = 0
    maximal_probability = -1
    for possible_fixation in xrange(Number_of_locs): #MOD Number_of_locs deleted
        probability_of_correctly_locating = compute_probability_for(possible_fixation)
        if probability_of_correctly_locating > maximal_probability:
            maximal_probability = probability_of_correctly_locating
            best_location = possible_fixation
    Focus = best_location #MOD Focus deleted

def compute_probability_for(fixation):
    """ Compute probability if the next fixation is fixation

    The function computes the probability of correctly identifying the target
    after the next fixation

    Args:
        fixation(int): next fixation for wicth to compute the probability of correctly
            identifying the target
    Returns:
        (int) Probability of correctly identifying the location of the target
            after the next fixation is made
    AccsGlobs:
        Number_of_locs, Dprime_map, Post_probs
    """
    probabilities = np.zeros(Number_of_locs) #MOD Number_of_locs deleted
    for  possible_target_location in xrange(Number_of_locs): #MOD Number_of_locs deleted
        probabilities[possible_target_location] = integrate.quad(
            integral_function,
            -np.inf, np.inf,
            args=(possible_target_location,Dprime_map[fixation]),
            epsabs=0,
            limit=100,
            full_output=1
        )[0] #MOD Dprime_map deleted
    return np.sum(Post_probs * probabilities) #MOD Post_probs deleted

def integral_function(w,i,dprime):
    """ The function inside the integral of the IBO algorithm

    Naming of variables is same as in the article

    Najemnin,Geisler(2005): Optimal eye movement strategies in visual search - Apendix
    """
    factors = 1
    for j in xrange(Number_of_locs): #MOD Number_of_locs deleted
        factors *= stats.norm.cdf(
            (
                -2.0*np.log(Post_probs[j]/Post_probs[i]) +
                dprime[j]**2 +
                2.0*dprime[i]*w +
                dprime[i]**2
            ) / (
                2.0*dprime[j]
            )
        ) #MOD Post_probs deleted
    return stats.norm.pdf(w) * factors   

### [B]OOST IBO ###
#TODO global -> vars_
def boost_initialization():
    """ Initialize library for boosted choose next fixation

    Run this function before you use boost_choose_next_fixation.
    Computation is accelerated via partial computation in C. The computation is
    implemented in integral_function.so. The file integral_function.so must implement
    two functions. The function set for initialization, the function set_location for
    restarting values for the beggining of computation and the function function
    wich implements integrated function passed to np.quad.
    """
    global Lib_c 
    Lib_c = ctypes.CDLL('./integral_function.so')
    Lib_c.set.restype = None
    Lib_c.set.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    Lib_c.set_target.restype = None
    Lib_c.set_target.argtypes = (ctypes.c_int,)
    Lib_c.function.restype =  ctypes.c_double
    Lib_c.function.argtypes = (ctypes.c_int,ctypes.c_double)

def boost_ibo():
    """ Choose next fixation.

    Same as choose_next_fixation function. But the integral function is implemented
    in C. Run boost_initialization before you use this function

    AccsGlobs:
        Number_of_locs, Lib_c, Dprime_map, Post_probs
    """
    assert ('Lib_c' in globals()), "Run boost_initialization."
    global Focus #MOD Focus deleted
    best_location = 0
    maximal_probability = -1
    for possible_fixation in xrange(Number_of_locs): #MOD Number_of_locs deleted
        Lib_c.set(Number_of_locs,  Dprime_map[possible_fixation].ctypes, Post_probs.ctypes)
            #MOD Number_of_locs deleted #MOD Dprime_map deleted #MOD Post_probs deleted
        probability_of_correctly_locating = boost_probability_for(possible_fixation)
        if probability_of_correctly_locating > maximal_probability:
            maximal_probability = probability_of_correctly_locating
            best_location = possible_fixation
    Focus = best_location #MOD Focus deleted

def boost_probability_for(fixation):
    """ Compute probability if next fixation is fixation

    Same as compute_probability_for. Only a part of the computation is perfomed
    in c function.

    Args:
        see compute_probability_for
    Returns:
        see compute_probability_for
    AccsGlobs:
        Lib_c, Post_probs
    """
    probabilities = np.zeros(Number_of_locs) #MOD Number_of_locs deleted
    for possible_target_location in xrange(Number_of_locs): #MOD Number_of_locs deleted
        Lib_c.set_target(possible_target_location)
        probabilities[possible_target_location] = integrate.quad(
            Lib_c.function,
            -np.inf, np.inf,
            epsabs=0,
            limit=50,
            full_output=1
        )[0]
    return np.sum(Post_probs * probabilities) #MOD Post_probs deleted

### [E]LM ###
def elm(vars_):
    """ Compute a next fixation using ELM algorithm

    The function use ELM algorith to compute a next fixation. The algorithm is from

    J.Najemnik, W.S.Geisler: Smple summation rule for potimal fixation selection in visual search
    Vision Research 49 (2009), 1286-1294S

    Args:
        vars (SimulationVariables)
    """
    assert (vars_.dprime_map.shape[1] == vars_.post_probs.shape[0]), "Invarian violation:" \
        " dprime map and posterior probabilies matrix have different number of locations" 
    elm_map = np.dot(vars_.dprime_map,vars_.post_probs)
    vars_.focus = np.argmax(elm_map)

### I[N]ITIALIZATION ###
#? Pass generator or generator_creater_function
def input_simulation_attributes_generator():
#TODO: description
    default_simulation_attributes = SimulationAttributes()
#TODO: don't use global REAL
#? Set id_name?
#    default_simulation_attributes.id_name = 'default'
    default_simulation_attributes.locations = REAL
    default_simulation_attributes.dprime_fnc = '5 no'
    default_simulation_attributes.next_fixation = 'elm'
    default_simulation_attributes.threshold = 0.95
    default_simulation_attributes.num_of_searches = 1
    if Args.conf_fin:
        attributes_skeleton_list = conf_file_load_attributes_skeletons(Args.conf_fin)
    else:
        attributes_skeleton_list = []
    args_simulation_attributes = load_simulation_attributes_from_args(Args)
    return simulation_attributes_generator(
        args_simulation_attributes,
        attributes_skeleton_list,
        default_simulation_attributes
    )
    
def simulation_attributes_generator(sim_attr, attr_skel_list, default_attr):
    """ SimulationAttributes from AttributesSkeleton + SimulationAttributes + Default

    It generates SimulationAttributes by filling sim_attr by SimulationAttributes
    generated from AttributesSkeleton in attr_skel_list and by default
    SimulationAttributes. Created SimulationAttributes is SHALLOW copy of
    SimulationAttribute sim_attr ,SimulationAttriubtes generated from
    a AttributesSkeleton in attr_skel_list and default SimulationAttributes

    Args:
        sim_attr (SimulationAttributes): can contain None attributes. None attributes
            will be filled and filled instance will be returned.
        attr_skel_list (list of AttributesSkeleton): AttributesSkeleton form wich
            to generate SimulationAttributes to be used for fillin None attributes
            in sim_attr
        default_attr (SimulationAttributes): default SimulationAttributes
    Yields:
        yields shallow copies of sim_attr, None attributes filled by values
        from SimulationAttributes generated from the instances in attr_skel_list
        and by default SimulationAttributes
    """
#TODO don't forget id_name
#SOLVED
    none_attr_list = none_simulation_attributes(sim_attr)
    id_name = sim_attr.id_name
    if len(attr_skel_list) == 0:
        sim_attr.fill(default_attr)
        yield sim_attr
        set_simulation_attributes(sim_attr,none_attr_list,None)
        sim_attr.id_name = id_name
        raise StopIteration()
    for attr_skel in attr_skel_list:
#? Ugly support for analyses
        sim_attr._analyses = attr_skel.analyses
        for skel_attr in attr_skel.generate_simulation_attributes():
            sim_attr.fill(skel_attr)
#? code repetition
            sim_attr.fill(default_attr)
            yield sim_attr
            set_simulation_attributes(sim_attr,none_attr_list,None)
            sim_attr.id_name = id_name

def none_simulation_attributes(sim_attr):
#TODO description
    attr_l = ['locations', 'dprime_fnc', 'next_fixation', 'threshold',
        'num_of_searches']
    return [attr for attr in attr_l if getattr(sim_attr,attr) is None]

def set_simulation_attributes(sim_attr,attr_l,val):
#TODO description
    for attr in attr_l:
        setattr(sim_attr,attr,val)



### [A]RGS LOAD ###
def load_simulation_attributes_from_args(args):
    """ Loads simulation attributes from command line arguments

    Args:
        args (class): class wich contains information about how to initialize
            variables.
            Structure of args:
                args
                    |- senzory_map_file
                    |- dprime_map_file
                    |- dprime_map_type
                    |- dprime_map_str
                    |- algorithm
                    |- num_of_searches
                    L  threshold
    Returns:
        (SimulationAttributes)
    """
    simulation_attributes = SimulationAttributes()
    if hasattr(args,'output_file'):
        simulation_attributes.id_name = args.output_file
    simulation_attributes.locations = load_locations_from_args(args.senzory_map_file)
    simulation_attributes.dprime_fnc = load_dprime_fnc_from_args(args)
    simulation_attributes.next_fixation = args.algorithm
    simulation_attributes.num_of_searches = args.num_of_searches
    simulation_attributes.threshold = args.threshold
    return simulation_attributes

def load_locations_from_args(file_name):
    """ Loads locations from argument options

    Locations is np.array of (N,2) shape. A row equals to a senzory location.
    Supported files are:
        .mat - locations are stored in 'locations' key

    Args:
        file_name (str): The name of file containing locations for a target.
            If the name is empty, the defeault senzory locations are taken.
    Returns:
        (np.array) locations or None
    """
    if not file_name:
        locations = None
    else:
        locations = load_senzory_locations(file_name)
    return locations

def load_senzory_locations(file_name):
    """ Load senzory locations from the file

    The function returns np.array of senzory locations found in the file.

    Args:
        file_name (str): name of file from wich to load locations.
            Excpected file types are .mat and .csv
    Returns:
        (np.array) An array of (N,2) shape. The locations are stored in rows.
    """
    check_file_existence(file_name)
    _, ext = os.path.splitext(file_name)
    if ext == '.mat':
        return load_senzory_locations_from_matlab(file_name)
    elif ext == '.csv':
        return load_senzory_locations_from_csv(file_name)
    else:
        raise ValueError("Unknown file type at {}. Expected .mat or .csv".format(file_name))

def load_senzory_locations_from_matlab(file_name):
    """ Load senzory locations stored in .mat file

    The function loads senzory locations stored in .mat file under 'locations' key name
    and returns np.array of (2,N) shape

    Args:
        file_name (str): name of .mat file
    Returns:
        (np.array) An array of (N,2) shape. The locations are stored in rows.
    """
    check_file_existence(file_name)
    try:
        return scipy.io.loadmat(file_name)['locations']
    except KeyError:
        raise EnvironmentError(
            str("Senzory locations are expected "
            "to be stored in 'locations' variable in {} file".format(file_name))
        )
    except:
        raise EnvironmentError("Unable to load senzory map from {}".format(file_name))

def load_senzory_locations_from_csv(file_name):
    """ Load senzory locations from .csv file

    The function loads senzory locations stored in .csv file. Number_of_locs are
    excpected to be stored in rows.

    Args:
        file_name (str): name of .csv file
    Return:
        (np.array) An array of (N,2) shape. The locations ara stored in rows.
    """
    check_file_existence(file_name)
    try:
        return np.loadtxt(file_name, dtype=float, delimiter=',')
    except:
        raise EnvironmentError("Unable to load senzory map from {}".format(file_name))

def check_file_existence(file_name):
    """ Check whether the file exist

    If the file doesn't exist, the function raise IOError.

    Args:
        file_name (str): the name of file to check for existence
    """
    if not os.path.isfile(file_name):
        raise IOError("{} doesn't exist or isn't a file".format(file_name))

def load_dprime_fnc_from_args(args):
    """ Loads dprime function as specified in the arguments options

    The dprime function can be specified in following ways.:
        1. file
        2. simple argument initialization
        3. structured argument initialization

    Args:
        args (class): The initialization specification. For corresponding
            type of initialization as mentioned above, there are these
            attributes.:
                1. dprime_map_fin (file stream)
                2. dprime_map_type (str)
                3. dprime_map_str (str)
    Returns:
        (function) dprime function None
    """
    if args.dprime_map_fin:
        dprime_fnc = dprime_fnc_from_fin_one_line(args.dprime_map_fin)
    elif args.dprime_map_str:
        dprime_fnc = dprime_fnc_from_str_argument(args.dprime_map_str)
    elif args.dprime_map_type:
        dprime_fnc = dprime_fnc_from_type(args.dprime_map_type)
    else:
        dprime_fnc = None
    return dprime_fnc

def dprime_fnc_from_fin_one_line(fin):
    """ Generate dprime function as specified in file istream

    The function returns dprime function. The specification for the dprime
    function is given in the file istream. The specification is expected to be
    only and only one line.

    Args:
        fin (stdin): input stream from where the specification is read
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    try:
        fnc = dprime_fnc_from_fin(fin)
        next_line = fin.readline()
        if next_line != '':
            raise EnvironmentError("Unknown lines in specification.")
        return fnc
    except (EnvironmentError,ValueError), Argument:
        raise EnvironmentError(
            "Error in the dprime configuration {0}. {1}".format(fin.name, Argument)
        )

def dprime_fnc_from_fin(fin):
    """ Generate dprime function as specified in file istream.

    Args:
        fin (stdin): input stream from where the specification is read
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    line = fin.readline()
    return dprime_fnc_from_str(line)

def dprime_fnc_from_str(line):
    """ Generate dprime function from the string

    The function returns dprime function, wich can be used for a dprime map generation.
    The specification for the dprime function is given in string. It is expected, that
    the string has at leas 2 arguments.

    Args:
        line (str): the specification string. Format is:
            (d0 = central sensitivity) (mask = type of mask) [specifications for the mask]
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    dprime_info = line.split()
    if len(dprime_info) < 2:
        raise ValueError("Too few arguments. Excpected (d0) (mask).")
    fovea_info = dprime_info.pop(0)
    try:
        fovea_sensitivity = float(fovea_info)
        if fovea_sensitivity <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("Excpected positive float. Instead {} was found.".format(fovea_info))
    mask = dprime_info.pop(0)
    if mask == 'no':
        return create_no_mask_fnc(dprime_info,no_mask(fovea_sensitivity))
    elif mask == 'scotoma':
        return create_scotoma_fnc(dprime_info,no_mask(fovea_sensitivity))    
    elif mask == 'glaucoma':
        return create_glaucoma_fnc(dprime_info,no_mask(fovea_sensitivity))
    elif mask == 'hemianopsia':
        return create_hemianopsia_fnc(dprime_info,no_mask(fovea_sensitivity))
    else:
        raise ValueError("Unknown dprime map mask {}.".format(mask))

def create_no_mask_fnc(specifications,dprime_fnc):
    """
    Args:
        specifications (list of str): empy
        dprime_fnc (function): the function to be masked
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    if len(specifications) > 0:
        raise ValueError("Unknown arguments for no mask.")
    return dprime_fnc

def create_scotoma_fnc(specifications,dprime_fnc):
    """ Create dprime function with scotoma

    The function creates a dprime function with scotoma according to the specification.
    The basic function to be masked is dprime_fnc.

    Args:
        specifications (list of str): the specifications for scotoma masking. Format is:
            (delimiting = [hard,soft]) (a) (b) [if soft than slope]
        dprime_fnc (function): the function to be masked
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    if len(specifications) < 1:
        raise ValueError("Too few arguments for scotoma mask")
    delimiting_type = specifications.pop(0)
    if delimiting_type == 'soft':
        try:
            check_arguments_number(specifications,3)
            a, b, slope = [str_to_positive_float(s) for s in specifications]
            return scotoma_soft(a,b,slope,dprime_fnc)
        except ValueError, Argument:
            raise ValueError("Specification error for scotoma soft. {}".format(Argument))
    elif delimiting_type == 'hard':
        try:
            check_arguments_number(specifications,2)
            a, b = [str_to_positive_float(s) for s in specifications]
            return scotoma_hard(a,b,dprime_fnc)
        except ValueError, Argument:
            raise ValueError("Specification error for scotoma hard. {}".format(Argument))
    else:
        raise ValueError(
            "Specification error for scotoma. Expected [hard,soft], but found {}.".format(delimiting_type)
        )

def create_glaucoma_fnc(specifications,dprime_fnc):
    """ Create dprime function with glaucoma

    The function creates a dprime function with glaucoma according to the specification.
    The basic function to be masked is dprime_fnc.

    Args:
        specifications (list of str): the specifications for glaucoma masking. Format is:
            (delimiting = [hard,soft]) (a) (b) [if soft than slope]
        dprime_fnc (function): the function to be masked
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    if len(specifications) < 1:
        raise ValueError("Too few arguments for glaucoma mask")
    delimiting_type = specifications.pop(0)
    if delimiting_type == 'soft':
        try:
            check_arguments_number(specifications,3)
            a, b, slope = [str_to_positive_float(s) for s in specifications]
            return glaucoma_soft(a,b,slope,dprime_fnc)
        except ValueError, Argument:
            raise ValueError("Specification error for glaucoma soft. {}".format(Argument))
    elif delimiting_type == 'hard':
        try:
            check_arguments_number(specifications,2)
            a, b = [str_to_positive_float(s) for s in specifications]
            return glaucoma_hard(a,b,dprime_fnc)
        except ValueError, Argument:
            raise ValueError("Specification error for galucoma hard. {}".format(Argument))
    else:
        raise ValueError(
            "Specification error for glaucoma. Expected [hard,soft], but found {}.".format(delimiting_type)
        )

def create_hemianopsia_fnc(specifications,dprime_fnc):
    """ Create dprime function with hemianopsia

    The function creates a dprime function with hemianopsia according to the specification.
    The basic function to be masked is dprime_fnc.

    Args:
        specifications (list of str): the specifications for hemianopsia masking. Format is:
            (delimiting = [hard,soft]) (left/right) [if soft than slope]
        dprime_fnc (function): the function to be masked
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    if len(specifications) < 1:
        raise ValueError("Too few arguments for hemianopsia mask")
    delimiting_type = specifications.pop(0)
    if delimiting_type == 'soft':
        try:
            check_arguments_number(specifications,2)
            hemi = specifications.pop(0)
            check_argument_in_choices(hemi,['left','right'])
            [slope] = [str_to_positive_float(s) for s in specifications]
            return hemianopsia_soft(hemi,slope,dprime_fnc)
        except ValueError, Argument:
            raise ValueError("Specification error for hemianopsia soft. {}".format(Argument))
    elif delimiting_type == 'hard':
        try:
            check_arguments_number(specifications,1)
            hemi = specifications.pop(0)
            check_argument_in_choices(hemi,['left','right'])
            return hemianopsia_hard(hemi,dprime_fnc)
        except ValueError, Argument:
            raise ValueError("Specification error for hemianopsia hard. {}".format(Argument))
    else:
        raise ValueError(
            "Specification error for hemianopsia. Expected [hard,soft], but found {}.".format(delimiting_type)
        )

def check_arguments_number(args_list,excpected_num):
    """ Check whether args_list has expected_num of elements.

    If the condition isn't satisfied, the function raises ValueError with
    specific error massage.

    Args:
        args_list (list): list of arguments to be checked
        excpected_num (int): expected number of arguemnts in the list
    """
    if not len(args_list) == excpected_num:
        raise ValueError(
             "Expected {0} arguements, but found {1}.".format(
                 excpected_num, len(args_list)
             )
         )

def check_argument_in_choices(arg,choices):
    """ Check whether arg is in list of choices

    If the condition isn't satisfied, the function raises ValueError with
    specific error massage.

    Args:
        arg (str): argument
        choices (list of str): possible choices
    """    
    if not arg in choices:
        raise ValueError(
            "Unknown argument {0}. Expected one from {1}".format(
                arg, choices
            ).translate(None,"'")
        )

def dprime_fnc_from_type(dprime_type):
    """ Create dprime function masked by dprime_type

    The function creates dprime function with default values.

    Args:
        dprime_type (str): dprime type as specified in Dprime_types global variables
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    GlobAccs:
        Dprime_types
    """

    dprime_fncs = [no_mask(),
        scotoma_hard(),
        scotoma_soft(),
        glaucoma_hard(),
        glaucoma_soft(),
        hemianopsia_hard('left'),
        hemianopsia_soft('left'),
        hemianopsia_hard('right'),
        hemianopsia_soft('right')
    ]
    if not dprime_type in Dprime_types:
        raise ValueError("Unknown dprime type {}.".format(dprime_type))
    return dprime_fncs[Dprime_types.index(dprime_type)]

def dprime_fnc_from_str_argument(str_arg):
    """ Create dprime function from string provided as argument
    
    The function returns dprime function, wich can be used for a dprime map generation.
    The specification for the dprime function is given in string. It is expected, that
    specification arguments are delimited by ':'.

    Args:
        line (str): the specification string. Format is:
            (d0 = central sensitivity):(mask = type of mask):[specifications for the mask]
    Returns:
        (function) dprime function, wich can be used for a dprime map generation
    """
    try:
        return dprime_fnc_from_str(
            str_arg.translate(string.maketrans(':',' '))
        )
    except ValueError, Argument:
        raise ValueError("Dprime argument error (in {0}). {1}".format(str_arg, Argument))

### CONFIGURATION [F]ILE LOAD ###
class AttributesSkeleton(SimulationAttributes):
    """ Parametric version for SimulationAttributes

    The class can contain parameteric definitions for the SimulationAttributes
    attributes. A parameter is string starting with '$'. '$' can't be used elsewhere.
    It is advised use a-z, A-Z, 0-9 and '_' for the naming of a parameter.
    To generate SimulationAttributes instance use generate_simulation_attributes.

    Methods:
        Setters (for dprime_fnc, next_fixation, threshold, num_of_searches):
            Can be a definition with parameters
    Disclaimer:
        To access instance variables, generate SimulationAttibutes
        by generate_simulation_attriubetes method. Other way of accessing doesn't
        have to have consistent behaviour.
    """
    def __init__(self,id_name_base):
#? id_name_base, optional?
        """
        Args:
            id_name_base (string): a base name from wich to generate
                SimulationAttributes id_name
            parameters_values (dict): dictionary for parameters and their values.
                A key is name of parameter, i.e. string. A value is list of values
                for the parameter.
        """
        SimulationAttributes.__init__(self)
        self._id_name_base = id_name_base
        self._params_in_attrs = dict()
        self._params_vals = dict()
        self._attrs_def = dict()
   
    @SimulationAttributes.dprime_fnc.setter
    def dprime_fnc(self,val):
        if isinstance(val, basestring):
            val = string.join(val.split())
            if '$' not in val:
                SimulationAttributes.dprime_fnc.fset(self,val)
                return
        self._insert_parametric_attribute('dprime_fnc',val)

    @SimulationAttributes.next_fixation.setter
    def next_fixation(self,val):
        if isinstance(val, basestring):
            val = val.strip()
            if '$' in val:
                self._insert_parametric_attribute('next_fixation', val)
                return
        SimulationAttributes.next_fixation.fset(self,val)

    @SimulationAttributes.threshold.setter
    def threshold(self,val):
        if isinstance(val, basestring):
            val = val.strip()
            if '$' in val:
                self._insert_parametric_attribute('threshold', val)
                return
        SimulationAttributes.threshold.fset(self,val)

    @SimulationAttributes.num_of_searches.setter
    def num_of_searches(self,val):
        if isinstance(val, basestring):
            val = val.strip()
            if '$' in val:
                self._insert_parametric_attribute('num_of_searches', val)
                return
        SimulationAttributes.num_of_searches.fset(self,val)

#? Parametrization of analysis? How to implement.
#DEBUG
    def __str__(self):
        return "loc:{0}\ndfnc:{1}\nnf:{2}\nth:{3}\nns:{4}\npvr:{5}\npvl:{6}\nvd:{7}\nname:{8}".format(
            "censored",
            self.dprime_fnc,
            self.next_fixation,
            self.threshold,
            self.num_of_searches,
            self._params_in_attrs,
            self._params_vals,
            self._attrs_def,
            self._id_name
        )
#END DEBUG
        
    def add_parameter(self,par_name,par_vals):
        """
        Add parameter-values pair. par_name must have correct parameter name
        definition and par_vals can't be empty. If the par_name is already
        present, it is overwritten by new par_vals specification.

        Args:
            par_name (string): parameter name
            par_vals (list): list of values of par_name
        """
        self._check_for_parameter_syntax(par_name)
        if len(par_vals) == 0:
            raise ValueError("Empty value list for parameter {}".format(par_name))
        self._params_vals[par_name] = par_vals

    def generate_simulation_attributes(self):
        """
        Returns:
            (self) Returns self with attributes filled as spesified
                in definitions. (!IT IS SHALLOW COPY!)
        Disclaimer:
            Generated SimulationAttributes should be use only for read.
            Other uses don't have defined behaviour.
        """
        parameter_values = dict() #extract parameters used in attributes definitions
        for par, _ in self._params_in_attrs.viewitems():
            parameter_values[par] = self._params_vals[par]
        if len(parameter_values) == 0:
#? UGLY
            self.id_name = self._id_name_base
            yield self
            raise StopIteration()
        for pars_assignment in generate_assignment(parameter_values):
            attributes_definitions = copy.deepcopy(self._attrs_def)
            for par_name, par_val in pars_assignment:
                attributes = self._params_in_attrs.get(par_name,[])
                for attribute in attributes:
                    attributes_definitions[attribute] = (
                        attributes_definitions[attribute].replace(par_name,str(par_val))
                    )
            yield self._generate_attributes_from_assignment(
                attributes_definitions,pars_assignment
            )

    def _check_for_parameter_syntax(self,parameter):
        """
        Check for correct parameter syntax. The parameter has to start with '$'.
        '$' can be only one in parameter. Whitespaces are forbidden.
        """
        err_msg = "Illegal parameter name {}.".format(parameter)
        if len(parameter) == 0:
            raise ValueError(err_msg + " Empty parameter name")
        if parameter[0] != '$':
            raise ValueError(err_msg + " Parameter must start with '$'")
        if parameter != string.join(parameter.split()).translate(None,' '):
            raise ValueError(err_msg + " Parameter can't contain whitepaces")
        if ('$' in parameter and parameter[0] != '$') or (parameter.count('$') > 1):
            raise ValueError(
                err_msg + " Wrong parameter specification in {}".format(parameter)
            )

    def _insert_parametric_attribute(self,attr_name,attr_def_str):
        """
        Takes an attribute definition as a string contaning '$'.
        Check whether the definition is syntacticly correct and
        values of parameters can be correctly substituted to the definitions.
        If it passes the tests, the definition is stored to _attr_defs and
        the name of parametrs in the definition is stored in _params_in_attrs.

        Args:
            attr_name (string): the name of attribute to be defined
            attr_def_str (list of string): difinition for attr_name.
                It is string containing '$'.
        """
        self._check_for_definition_correctness(attr_name,attr_def_str)
        self._attrs_def.update([(attr_name,attr_def_str)])
        for attr_def in attr_def_str.split():
            if '$' in attr_def:
                self._params_in_attrs.setdefault(attr_def,set()).add(attr_name)

    def _check_for_definition_correctness(self,attr_name,attr_def_str):
        """
        Check that the attr_def_str is in correct form.
        First it checks wheter all parameters in attr_def_str are
        in correct form and are present in _params_vals dictionary.
        Then it checks that the parameter values can be used in the
        attr_def_str
        """
        pars_vals = dict()
        # parameters syntax check
        for attr_def in attr_def_str.split():
            if '$' not in attr_def:
                continue
            self._check_for_parameter_syntax(attr_def)
            if attr_def not in self._params_vals.viewkeys():
                raise ValueError(
                    "Using undefined paramter {0} in {1}".format(attr_def,attr_def_str)
                )
            pars_vals[attr_def] = self._params_vals[attr_def]
        # parameters values substitution check
        test_s = SimulationAttributes()
        for pars_assignment in generate_assignment(pars_vals):
            for par_name, par_val in pars_assignment:
                attr_def_str = attr_def_str.replace(par_name,str(par_val))
            try:
                setattr(test_s,attr_name,attr_def_str)
            except ValueError, e:
                raise ValueError(
                    "Unleagal value for a parameter in attribute {3}. '{0}' ({1}). {2}".format(
                        attr_def_str,
                        str(pars_assignment)[2:-2].translate(
                            string.maketrans(',','=')
                        ).replace(')= (',',').translate(None," '"),
                        e,
                        attr_name
                    )
                )
        
    def _generate_attributes_from_assignment(self,attributes_definitions,pars_assignment):
        def translate_value(value):
            if isinstance(value,float):
                return "{0:.1f}".format(value)
            else:
                return str(value)

        if len(pars_assignment) == 0:
            self.id_name = self._id_name_base
        else:
            self.id_name = '{0}_{1}'.format(
                self._id_name_base,
#TODO: clean
#                str(pars_assignment).translate(None," '$[]"
#                    ).replace('),(','_').translate(None,'()'
#                    ).translate(string.maketrans(',','-'))
                string.join(
                    [
                        var_name[1:] + '-' + translate_value(value)
                        for var_name, value in pars_assignment
                    
                    ],
                    '_'
                )
            )
        for attr_name,attr_def in attributes_definitions.viewitems():
            setattr(self,attr_name,attr_def)
        return self

def generate_assignment(parameters):
    """ Generate assignment for parametrs-values dictionary

    It generates all possible combination for parameters-values found
    in dictionary.

    Args:
        parameters (dict): values of keys must be list or at least 
            interable
    Returns:
        (list of pair) list of assignment pari for parameter-value.
    """
    if len(parameters) == 0:
        yield []
        raise StopIteration()
    cp_pars = copy.deepcopy(parameters)
    par, values = cp_pars.popitem()
    for val in values:
        for r in generate_assignment(cp_pars):
            yield r + [(par,val)]

#? Add support for variable order of linse + warning if double definition or
#? definition is missing
#TODO: add support for analysis
def conf_file_load_attributes_skeletons(finput):
    """ Load simulations attributes from configuration files

    Configuration file must contain at least one run definition. The run
    definition must contain following defininition in the following order.
        RUN:(name: string)
        VAR:(number: positive int)
        LOCATIONS:(DEFAULT|FILE|+)
        DPRIME:(DEFAULT|FILE|+)
        ALGORITHM:(elm|ibo|cibo)
        THRESHOLD:(positive float)
        NSEARCHES:(positive int)
        ANALYSES:(positive int)

    Args:
        finput (input stream): configuration input stream
    Returns:
        (list of AttributesSkeleton)
    """

    class fin_wrap:
        """
        Line counter and empty line skipper.
        """
        class EndFile(Exception):
            pass

        def __init__(self,f):
            self.fin = f
            self.line = 0

        def readline(self):
            while (True):
                line = self.fin.readline()
                self.line += 1
#? How to handle end file. In this way, we loos checking for missing attributes
#? definition.
                if len(line) == 0:
                    raise self.EndFile()
                if len(line.rstrip()) == 0:
                    continue
                return line

    skeleton_list = []
    fin = fin_wrap(finput)
    while(True):
        try:
            skeleton_list.append(conf_load_run(fin))
        except fin_wrap.EndFile:
            break
        except (EnvironmentError,ValueError), e:
            raise EnvironmentError("Error in {0} at line {1}. {2}".format(
                    finput.name, fin.line, e
                )
            )
    return skeleton_list

def conf_load_run(fin):
    """ Load run definition

    Returns:
        (AttributesSkeleton)
    """
    name = conf_load_run_specification(fin)
    attr_skeleton = AttributesSkeleton(name)
    conf_load_parameters(fin,attr_skeleton)
    conf_load_attributes_skeleton(fin,attr_skeleton)
    return attr_skeleton

def conf_load_run_specification(fin):
    """ Loads specification line for run

    Specification format is RUN:name. name is nonempty string, wich will be
    used as run core ID.

    Args:
        fin (input stream)
    Returns:
        (string) name for the run
    """
    err_msg = "Unknown specification. Excpected RUN:'name'."
    spec = fin.readline().strip().split(':')
    if len(spec) != 2 or spec[0] != 'RUN':
        raise EnvironmentError(err_msg)
    name  =  spec[1].strip()
    if len(name) == 0:
        raise EnvironmentError("Excpected non empty name for RUN(RUN:'name').")
    return name

def conf_load_parameters(fin,skeleton):
    """ Loads parameters as specified in the input stream

    The inpustream has to start with VAR:(nonnegative int) line.

    Args:
        fin (input stream)
        skeleton (AttributesSkeleton): where to load parameters
    """
    err_msg = "Unknown specification. Excpected VAR:(nonnegative int)"
    spec = fin.readline().strip().split(':')
    if len(spec) != 2 or spec[0] != 'VAR' or not str_is_nonnegative_int(spec[1]):
        raise EnvironmentError(err_msg)
    num_of_vars = int(spec[1])
    pars_list = []
    for _ in xrange(num_of_vars):
        par_name, par_list = conf_load_parameter(fin)
        if par_name in pars_list:
            raise EnvironmentError("Parameter {} already defined.".format(par_name))
        pars_list.append(par_name)
        skeleton.add_parameter(par_name, par_list)

def conf_load_parameter(fin):
    """ Loads one line with a parameter definiciton

    The function loads a line and converts it to a parameter form.
    The line should start with the parameter name ($(string)),
    followed by '=' and end by values specification.
    Supported values specifications are:
        list, linspace and range

    Args:
        fin (input stream): input stream contianing a line with
            parameter definiction.
    Returns:
        (tuple) Pair of parameter name and list of parameter values.
    """
    err_msg = "Unknown parameter definition. Excpected $par_name=(list|range|linspace)."
    spec = fin.readline().strip().split('=')
    if len(spec) != 2:
        raise EnvironmentError(err_msg)
    par_name, par_def = [s.strip() for s in spec]
    if len(par_def) > 1 and par_def[0] == '[' and par_def[-1] == ']':
        return par_name, conf_load_par_list(par_def)
    elif len(par_def) > 3 and par_def.count(':') == 2 and par_def[-1] == 'l':
        return par_name, conf_load_par_linspace(par_def)
    elif par_def.count(':') == 2:
        return par_name, conf_load_par_range(par_def)
    else:
        raise EnvironmentError(err_msg + " Found {0} for {1}".format(par_def,par_name))

def conf_load_par_list(par_def):
    """ Convert list parameter defiction to list
    
    The function takes as an argument a list of strings as a string.
    An element in the list string has to be enclosed in apostrophe.
    Proper format is ['val1', 'val2', 'val3', ...]

    Args:
        par_def (string): list specification in correct form, i.e. the string
            starts with '[' and ends with ']', elements are separated by commas
            and are enclosed by apostrophes
    Returns:
        (list of strings)
    """
    par_def = par_def[1:-1].split(',')
    par_list = list()
    for p in par_def:
        par_list.append(p.strip())
    return par_list

def conf_load_par_linspace(par_def):
    """ Convert linspace parameter definition to list

    The function takes as an argument a linspace specification in the form
    start:end:numl and create linspace list by np.linspace(start,end,num).

    Args:
        par_def (string): linspace specification in correct form, i.e. the string
            ends with 'l' and contains exactly two ':'
    Returns:
        (list of floats)
    """
    s,e,l = par_def[:-1].split(':')
    try:
        s = float(s)
        e = float(e)
        l = str_to_positive_float(l)
    except ValueError, e:
        raise ValueError(
            "Excpected float1:float2:positive_floatl for the linspace definition. {}".format(e)
        )
    par_list =  list(np.linspace(s,e,l))
    if len(par_list) == 0:
        raise ValueError("No parameter values generated.")
    return par_list

def conf_load_par_range(par_def):
    """ Convert range parameter deficiton to list

    The function takes as an argument a range specification in the form
    start:end:step and crate range list by np.arange(start,end,step).

    Args:
        par_def (string): range specification in correct for, i.e. the
            string contains exactly two ':'
    Returns:
        (list of floats)
    """
    try:
        s,e,n = [float(i) for i in par_def.split(':')]
    except ValueError, e:
        raise ValueError(
            "Excpected float1:float2:float3 for the range defiction. {}".format(e)
        )
    par_list = list(np.arange(s,e,n))
    if len(par_list) == 0:
        raise ValueError("No parameter values generated.")
    return par_list

def conf_load_attributes_skeleton(fin,skeleton):
    """ Loads attributes from input stream

    The functios loads simulation attributes from input stream.
    Attibutes can contain parameters.
    Configuration file must contain following defininition in the following order.
        LOCATIONS:(DEFAULT|FILE|+)
        DPRIME:(DEFAULT|FILE|+)
        ALGORITHM:(elm|ibo|cibo)
        THRESHOLD:(positive float)
        NSEARCHES:(positive int)
        ANALYSES:(positive int)
    For DPRIME, ALGORTHM, THRESHOLD, NSEARCH, there is support for parameters.

    Args:
        fin (input stream)
        sekelton (AttirubesSkeleton): where to load attibutes
    """
    conf_load_skeleton_locations(fin,skeleton)
    conf_load_skeleton_dprime_fnc(fin,skeleton)
    conf_load_skeleton_algorithm(fin,skeleton)
    conf_load_skeleton_threshold(fin,skeleton)
    conf_load_skeleton_num_of_searches(fin,skeleton)
    conf_load_skeleton_analyses(fin,skeleton)


def conf_load_skeleton_locations(fin,skeleton):
    """ Loads senzory locations as specified in input stream

    The function loads the senzory location into skeleton.
    No parameters are supported. Expected format LOCATIONS:(DEFAULT|FILE|+)

    Args:
        fin (input stream): input stream from configuration file
        skeleton (AttributesSkeleton)
    """
    action = conf_load_skeleton_locations_specification(fin)
    if action == 'DEFAULT':
        locations = REAL
    elif action == 'FILE':
        locations = conf_load_senzory_locations_file(fin)
    elif action == '+':
        locations = conf_load_senzory_locations_in(fin)
    else:
        raise EnvironmentError(
            "Unknown action option for LOCATIONS. Expected one of (DEFAULT|FILE|+)"
        )
    skeleton.locations = locations

def conf_load_skeleton_locations_specification(fin):
    """
    Loads an input line, check whether it contains with LOCATIONS:option and
    returns the option part.

    Args:
        fin (input stream)
    Returns:
        (string) an option for the location specification.
    """
    err_msg = "Unknown variable specification. Expected LOCATIONS:(DEFAULT|FILE|+)"
    spec = fin.readline().split(':')
    if len(spec) != 2 or spec[0] != 'LOCATIONS':
        raise EnvironmentError(err_msg)
    return spec[1].strip()

def conf_load_senzory_locations_file(fin):
    """ Load senzory locations from file

    Loads senzory locations from file. The name of the file is specified
    in configuration file input stream.

    Args:
        fin (file stream): input stream from the configuration file
    Returns:
        (np.array) matrix, (N,2) shape, of locations
    """
    file_name = fin.readline().strip()
    if file_name == '':
        raise EnvironmentError("Expected file name for LOCATIONS")
    return load_senzory_locations(file_name)

def conf_load_senzory_locations_in(fin):
    """ Load senzory locations from configuratio file

    Loads direclty senzory locations from configuration file.
    On the first line, there should be specified number of locations.
    One location on one row, numbers delimited by comma.

    Args:
        fin (file stream): input stream from the configuration file
    Returns:
        (np.array) matrix, (N,2) shape, of locations
    """
    try:
        num = int(fin.readline())
        if num < 1:
            raise EnvironmentError
    except:
        raise EnvironmentError("Expected positive integer for specifying a number of locations.")
    locations = np.zeros(shape=(num,2))
    for i in xrange(num):
        locations[i] = conf_load_location(fin)
    return locations

def conf_load_location(fin):
    """ Load one location from input stream

    The location is expected to be two numbers separeted by a comma.

    Args:
        fin (input stream): input steram from the configuration file
    Returns:
        (np.array) location, (2,) shape
    """
    err_msg = "Expected two numbers separeted by comma."
    line = fin.readline().split(',')
    if len(line) != 2:
        raise EnvironmentError(err_msg)
    try:
        x,y = float(line[0]), float(line[1])
    except:
        raise EnvironmentError(err_msg)
    return np.array([x,y])

def conf_load_skeleton_dprime_fnc(fin,skeleton):
    """ Loads dprime function as specified in input stream

    The function loads the dprime function into the skeleton.
    Paramers can be present in the dprime function specificaton.
    Excpected format is DPRIME:(DEFAULT|+)

    Args:
        fin (input stream)
        skeleton (AttributesSkeleton)
    """
    action = conf_load_skeleton_dprime_fnc_specification(fin)
    if action == 'DEFAULT':
        skeleton.dprime_fnc = dprime_basic
    elif action == '+':
        conf_load_skeleton_dprime_fnc_plus(fin,skeleton)
    else:
        raise EnvironmentError(
            "Unknown action option for DPRIME. Expected one of (DEFAULT|+)"
        )

def conf_load_skeleton_dprime_fnc_specification(fin):
    """
    Loads an input line, check whether it contains with DPRIME:option and
    returns the option part.

    Args:
        fin (input stream)
    Returns:
        (string) an option for the dprime function specification.
    """
    err_msg = "Unknown variable specification. Expected DPRIME:(DEFAULT|+)"
    spec = fin.readline().split(':')
    if len(spec) != 2 or spec[0] != 'DPRIME':
        raise EnvironmentError(err_msg)
    return spec[1].strip()

def conf_load_skeleton_dprime_fnc_plus(fin,skeleton):
    """ Loads dprime function as specified in input stream

    The function loads a function specification from input stream
    and generates the function. The specification can contain parameters

    Args:
        fin (input stream)
        skeleton (AttributesSkeleton)
    """
    line = fin.readline().strip()
    skeleton.dprime_fnc = line.strip().translate(string.maketrans(':',' ')) 

def conf_load_skeleton_algorithm(fin,skeleton):
    """ Loads search algorithm as specified in input stream

    The function loads the seach algorithm into skeleton.
    The algorithm type can be a parameter.
    Excpectedformat is ALGORITHM:((elm|ibo|cibo)|VAR)

    Args:
        fin (input stream)
        skeleton (AttributesSkeleton)
    """
    err_msg = "Unknown variable specification. Expeted ALGORITHM:(elm|ibo|cibo)"
    spec = fin.readline().split(':')
    if len(spec) != 2 or spec[0] != 'ALGORITHM':
        raise EnvironmentError(err_msg)
    skeleton.next_fixation = spec[1].strip()

def conf_load_skeleton_threshold(fin,skeleton):
    """ Loads threshold as specified in input stream

    The function loads the threshold into skeleton.
    The threshold can be a parameter.
    Excpectedformat is THRESHOLD:(unit float|VAR)

    Args:
        fin (input stream)
        skeleton (AttributesSkeleton)
    """
    spec = fin.readline().split(':')
    if len(spec) != 2 or spec[0] != 'THRESHOLD':
            raise EnvironmentError(err_msg)
    skeleton.threshold = spec[1].strip()

def conf_load_skeleton_num_of_searches(fin,skeleton):
    """ Loads number of searches as specified in input stream

    The function loads the number of seraches into skeleton.
    The number can be a parameter.
    Excpected format is NSEARCHES:(positive int|VAR)

    Args:
        fin (input stream)
        skeleton (AttributesSkeleton)
    """
    err_msg = "Unknown variable specification. Expected NSEARCHES:(positive int)"
    spec = fin.readline().split(':')
    if len(spec) != 2 or spec[0] != 'NSEARCHES':
            raise EnvironmentError(err_msg)
    skeleton.num_of_searches = spec[1].strip()

def conf_load_skeleton_analyses(fin,skeleton):
    """ Loads analyses as specified in input stream

    The function loads analyses as specified in input stream.
    First line is expected to be ANALYSES:num, where num is the number of
    analyses specified in following lines.

    Args:
        fin (input stream)
        skeleton (AttributesSkeleton)
    """
    err_msg = "Unknown variable specification. Expected ANALYSES:(positive int)"
    spec = fin.readline().split(':')
    if len(spec) != 2 or spec[0] != 'ANALYSES':
        raise EnvironmentError(err_msg)
    anal_num = str_to_nonnegative_int(spec[1])
    for _ in xrange(anal_num):
        anal_line = fin.readline().strip()
        skeleton.add_analysis(anal_line)

### ONLINE INP[U]TS ###
def input_is(string,msg):
    """ Wait for user input

    Function wait until user enters an input. Return True if the input match
    the argument.

    Args:
        string(str): string agains the input is checked
        msg(str): Message for a user to inform him what to do
    Return:
        (bool) True if an input equals to string otherwise it returns false
    """
    user_input = raw_input(msg)
    sys.stdout.write('\n')
    return user_input == string

### [V]ISUALIZATION ###
import sys
import colorama

Width = 11

T_style = colorama.Style.BRIGHT + colorama.Fore.RED + '\033[1m'
F_style = colorama.Style.BRIGHT + colorama.Fore.YELLOW + '\033[1m'
B_style = '\033[1m'
End_style = colorama.Style.RESET_ALL + '\033[0m'

def verbose_print(text,verbose_level):
    """ Print text for verbose level

    The function print the text at the standard output if verbose_level
    isn't smaller than user specified verbosity level.

    Args:
        text (string)
        verbose_level (int)
    """
    if Args.verbose >= verbose_level:
        print '\t' * (verbose_level-1) + text

def print_verbose(fix_num,vars_):
#? Move fixation number to var_?
    """ Verbose console output

    Args:
        fix_num(int): number of fix_numation from the start
        vars (SimulationVariables)
    AccsGlobs:
        T_style, F_style, End_style
    """
    sys.stdout.write("Fixation {}:\n".format(fix_num))
    sys.stdout.write("\tTreshold: {}\n".format(vars_.threshold))
    sys.stdout.write("\tTarget location: " +
        T_style + deg_tuple_to_str(tuple(vars_.senzory_map[vars_.target_location])) + '\n'
    )
    sys.stdout.write(End_style)
    sys.stdout.write("\tCurrent Focus: " +
        F_style + deg_tuple_to_str(tuple(vars_.senzory_map[vars_.focus])) + '\n'
    )
    sys.stdout.write(End_style)
    sys.stdout.write("\tCell:")
    sys.stdout.write(B_style)
    sys.stdout.write(" posterior probability\n")
    sys.stdout.write(End_style)
    sys.stdout.write("\t      coordinates in deg\n")
    sys.stdout.write('\n')
    post_probs_map = vector_to_locations_map_matrix(vars_.post_probs,vars_.senzory_map)
    target, focus = [
        vector_indx_to_map_matrix_indx(i,vars_.senzory_map)
        for i in (vars_.target_location,vars_.focus)
    ]
    deg_locations = vector_to_locations_map_matrix(vars_.senzory_map,vars_.senzory_map)
    print_matrices(
        [post_probs_map,deg_locations],
        [post_probs_writer(target,focus),deg_locations_writer]
    )
    sys.stdout.write('\n')

def deg_tuple_to_str(tup):
    """ Convert coordinate in degree to string

    Convert tuple of floats to string. Floats are printed with 2 decimal precision.

    Args:
        tup(tuple of float):
    Returns:
        argument converted in to the string
    """
    if len(tup) == 0:
        return "()"
    str = '('
    for x in tup:
        str += "{0:.2f}, ".format(x)
    str = str[:-2] + ')'
    return str

def print_matrices(matrices,writers):
    """ Print matrices styles with writers

    Print out a table with the same shape as matrices. How to print out
    cells is provided in writes.

    Args:
        matrices(list of np.array): matrices to be printed
        writers(list of funcions): printer of a cell
            writer(cell,width,coordinate): cell is matrix[x][y]
                                width is width of printed cell
                                cooridnate is cells position in matrix
    PreConds:
        len(matrices) > 0
        len(matrices) == len(writers)
        equality of x and y axis of matrices
    """
    assert (len(matrices) > 0), "Precondition violation: matrices can't be void"
    assert (len(matrices) == len(writers)), "Precondition violation: number matrices and writers don't match"
    rows, cols = matrices[0].shape[:2]
    assert (sum(
            [(rows,cols)  == matrix.shape[:2] for matrix in matrices]
        ) == len(matrices)), "Precondition violation: matrices don't have same number of rows and columns"
    print_first_line_for(cols)
    for i in xrange(rows-1):
        for matrix,writer in zip(matrices,writers):
            print_row(matrix[i],writer,i)
        print_line_for(cols)
    if rows > 0:
        for matrix,writer in zip(matrices,writers):
            print_row(matrix[-1],writer,rows-1)
    print_last_line_for(cols)

def print_row(row,writer,x):
    """ Print a row with a writer

    Args:
        row(np.array): a row to be printed
        writer(funcions):  printer of a cell
            writer(cell,width,coordinate): cell is matrix[x][y]
                                width is width of printed cell
                                coordinate is cells position in matrix
        x(int): x coordinate of the row
    AccsGlobs:
        Width
    """
    sys.stdout.write(unichr(0x2503))
    for n in xrange(row.shape[0]-1):
        writer(row[n],Width,(x,n))
        sys.stdout.write(unichr(0x2502))
    if row.shape[0] > 0:
        writer(row[-1],Width,(x,row.shape[0]-1))
    sys.stdout.write(unichr(0x2503) + '\n')

def print_first_line_for(cols):
    sys.stdout.write(unichr(0x250F))
    for n in xrange(cols-1):
        sys.stdout.write(unichr(0x2501) * Width + unichr(0x252F))
    if cols > 0:
        sys.stdout.write(unichr(0x2501) * Width)
    sys.stdout.write(unichr(0x2513) + '\n')
    
def print_line_for(cols):
    sys.stdout.write(unichr(0x2520))
    for n in xrange(cols-1):
        sys.stdout.write(unichr(0x2500) * Width)
        sys.stdout.write(unichr(0x253C))
    if cols > 0:
        sys.stdout.write(unichr(0x2500) * Width)
    sys.stdout.write(unichr(0x2528) + '\n') 

def print_last_line_for(cols):
    sys.stdout.write(unichr(0x2517))
    for n in xrange(cols-1):
        sys.stdout.write(unichr(0x2501) * Width + unichr(0x2537))
    if cols > 0:
        sys.stdout.write(unichr(0x2501) * Width)
    sys.stdout.write(unichr(0x251B) + '\n')

### [W]RITERS ###
# Functions wich controles how particular global matrices are printed.
# Writers have spedific form so that they can be used as a modifier for
# print_matrices function.
Precision = 4

def post_probs_writer(target,focus):
    """ Writer for post probabilities

    The function is used as a input for print_matrices function. The function
    is for writing of Post_probs cells. It highlights the target and focus cells.
    
    Args:
        target(tuple of int)
        focus(tuple of int)
    GlobAccs:
        T_style, F_style, End_style, B_style, Precision
    Returns:
        (f(x,y,z)) function for print_matrices
    """
    def writer(cell,width,(x,y)):
        if (x,y) == focus:
            sys.stdout.write(F_style)
        if (x,y) == target:
            sys.stdout.write(T_style)
        if (x,y) == focus and (x,y) == target:
            i = 0
            outstr = "{0:^{width}.{precision}f}".format(
                cell,width=width,precision=Precision
            )
            for c in outstr:
                if i % 2 == 0:
                    sys.stdout.write(T_style)
                else:
                    sys.stdout.write(F_style)
                sys.stdout.write(c)
                i +=1
        elif not np.isnan(cell):
            sys.stdout.write(B_style)
            sys.stdout.write("{0:^{width}.{precision}f}".format(
                cell,width=width,precision=Precision)
            )
        elif np.isnan(cell):
            sys.stdout.write("{0:^{width}s}".format('X',width=width))
            return
        else:
            assert False, "Unimplemented if branch"
        sys.stdout.write(End_style)
    return writer

def deg_locations_writer(cell,width,(x,y)):
    """ Writer for senzory locations

    The function is used as a input for print_matrices function. The function
    is for writing of Senzory_map cells

    Args:
        As required in print_matrices
    """
    if np.any(np.isnan(cell)):
        sys.stdout.write(' ' * width)
    else:
        sys.stdout.write(
            "{0:^{width}s}".format(
                "{0:.2f},{1:.2f}".format(cell[0],cell[1]),
                width=width
            )
        )

### [C]ONVERTORS ###
import itertools as it

def vector_to_locations_map_matrix(vector,senzory_map):
    """ Convert vector to locations map

    In IBO simulation, a vector is used for storing values from visual field.
    The function is used to convert the vector to a matrix. The matrix has
    much better visualization.

    Args:
        vector (np.array): vector of datas, e.g. Post_probs, Visual_field
        senzory_map (np.array): locations for wich the argument vector stores datas.
            One location per row.
    Returns:
        (np.array) Matrix filled with values from the vector, so that the value occupies
            similar position as specified in senzory map. Other values are np.nan.
    Disclaimer:
        Check for np.nan, use np.isnan() function.
    """
    xs = dict(zip(np.unique(senzory_map[:,0]), it.count()))
    ys = dict(zip(np.negative(np.unique(senzory_map[:,1])), it.count()))
    matrix = np.full((len(ys),len(xs)) + vector.shape[1:], np.nan, dtype=float)
    for i in xrange(vector.shape[0]):
        x,y = senzory_map[i]
        matrix[(ys[y],xs[x])] = vector[i]
    return matrix

def vector_indx_to_map_matrix_indx(index,senzory_map):
    """ Convert index to map index

    The function convert an index used to access arrays in IBO, e.g. Post_probs,
    to the index for locations map created via convert_to_lcoations_map

    Args:
        index(int): index for accessing IBO simulation array
        senzory_map (np.array): senzory map used as a model for a locations
            map matrix
    Returns:
        (tuple) index of vector index in locations map matrix, if the vector
            was transformed using vector_to_locations_map_matrix
    """
    xs = dict(zip(np.unique(senzory_map[:,0]), it.count()))
    ys = dict(zip(np.negative(np.unique(senzory_map[:,1])), it.count()))
    x, y = senzory_map[index]
    return ys[y],xs[x]

def str_to_nonnegative_float(string):
    value = float(string)
    if value < 0:
        msg = "{} is a not nonnegative number".format(string)
        raise ValueError(msg)
    return value

def str_to_positive_float(string):
    value = float(string)
    if value > 0:
        return value
    msg = "{} is a not nonnegative number".format(string)
    raise ValueError(msg)

def str_to_positive_int(string):
    value = int(float((string)))
    if value > 0:
        return value
    msg = "{} is not positive number".format(string)
    raise ValueError(msg)

def str_to_nonnegative_int(string):
    value = int(string)
    if value >= 0:
        return value
    msg = "{} is not positive number".format(string)
    raise ValueError(msg)

def str_is_nonnegative_int(string):
    try:
        _ = str_to_nonnegative_int(string)
        return True
    except ValueError:
        return False

def str_to_unit_interval_float(string):
    value = float(string)
    if value < 0 or value > 1:
        msg = "{} is not in [0,1]".format(string)
        raise ValueError(msg)
    return value

def str_to_next_fixation_fnc(string):
    if string == "elm":
        return elm
    elif string == "ibo":
        return ibo
    elif string == "cibo":
        boost_initialization()
        return boost_ibo
    else:
        msg = "{} is not in [elm,ibo,cibo]".format(string)
        raise ValueError(msg)

def dprime_map_to_dprime_origin(dprime_map, senzory_map):
    """ Returns dprime values for the origin

    Dprime map contains dprime values for every possible locaitons taken as origin.
    The function returns dprime values for (0.,0.)

    Args:
        dprime_map (np.array): dprime map
        senzory_map (np.array): senzory map for the dprime map
    Returns:
        (np.array) Dprime values for (0.,0.) as origin
    """
    return dprime_map[get_index_of_in(np.zeros(2), senzory_map)]

def dprime_fnc_to_mesh_grid(dprime_fnc, frame=(14,14), linspace=(100,100)):
    """ Create mash grid from dprime_fnc

    The function uses dprime_fnc for computing mesh grid.
    
    Args:
        frame (pair of positive float)): size of the mesh grid. The frame contains
            length of x and y axis. Mesh is centred to the origin, i.e. (0,0).
        linspace (pair of positive int): number of values for mesh grid along
            the axes. linspace[0] is for x axis, linspace[1] is for y axis.
    Rerturns
        (np.array), (np.array), (np.array): mesh grid for x, y and z axis. Shape is equale
            to (linspace[1],linspace[0])
    """
    assert len(frame) == 2, "Precondition violation"
    assert frame[0] > 0 and frame[1] > 0, "Precondition violation"
    assert len(linspace) == 2, "Precondition violatio"
    assert linspace[0] > 0 and linspace[1] > 0, "Precondition violation"
    mesh_X, mesh_Y = np.meshgrid(
        np.linspace(-(frame[0]/2.0),(frame[0]/2.0),linspace[0]), 
        np.linspace(-(frame[1]/2.0),(frame[1]/2.0),linspace[1]),
        indexing='xy'
    )
    mesh_Z = np.zeros(mesh_X.shape,dtype=float)
    for (x,y),_ in np.ndenumerate(mesh_X):
        mesh_Z[x,y] = dprime_fnc(np.array((mesh_X[x,y],mesh_Y[x,y])))
    return mesh_X, mesh_Y, mesh_Z

### [T]EST FUNCTIONS ### 
def stop_watch(function):
    start = time.clock()
    function()
    end = time.clock()
    print end - start

def test():
    initialize()
    update_visual_field()
    update_posterior_probs()

def test_boost():
    global LOCATIONS
    boost_initialization();
    LOCATIONS = REAL
    initialize()
    update_visual_field()
    update_posterior_probs()

### ANALYZE [J] ###
#TODO: delte after cleaning
#START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: 
#START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: 
#START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: 
#START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: 
#START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: 
#START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: 
#START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: START NEW: 
class AnalysisAttributes(object):
    """
    Attributes:
        method (string): one of 'FSD', 'SD'
        numbers ((int,int)): a pair of positive integers
        range (((float,float),(float,float))): a pair of pair of numbers.
            ((xmin,xmax),(ymin,ymax)). Stisfies xmin<=xmax and ymin<=ymax.
    """
    def __init__(self,specification=None):
        self._method = None
        self._numbers = None
        self._range = None
        if specification == None:
            return
        elif isinstance(specification,basestring):
            vals = specification.split()
            if len(vals) == 1:
                self.method= vals[0]
                self.numbers = {
                    'FSD':(100,100),
                    'SD': (20,20)
                }[self.method]
            elif len(vals) == 3:
                self.method = vals[0]
                self.numbers = string.join(vals[1:])
            elif len(vals) == 7:
                self.method = vals[0]
                self.numbers = string.join([vals[3],vals[6]])
                try:
                    self.range = string.join(vals[1:3] + vals[4:6])
                except ValueError, e:
                    raise ValueError(
                        "Range specification error. {}".format(e)
                    )
            else:
                raise ValueError(
                    "Unsupported number of values in '{}'".format(specification)
                )
        else:
            raise ValueError(
                "Unsupported init argument value type: {}".format(specification)
            )

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self,val):
        """
        Args:
            val (string): val has to be 'FSD' or 'SD'
        """
        analysis_methods = ['FSD','SD']
        if val not in analysis_methods:
            raise ValueError(
                '{0} not in {1}'.format(val,analysis_methods).replace("'",'')
            )
        self._method = val

    @property
    def numbers(self):
        return self._numbers

    @numbers.setter
    def numbers(self,val):
        """
        Args:
            val (string): 'xnum ynum'.
                ([int,int]): a pair of positive integers
        """
        if isinstance(val,basestring):
            try:
                val = [str_to_positive_int(s) for s in val.split()]
            except ValueError, e:
                raise ValueError(
                    "Error in numbers definition. {}".format(e)
                )
        else:
            val = [int(v) for v in val if v > 0]
        if len(val) != 2:
            raise ValueError(
                "Expected two positive int specification. Found: {}".format(val)
            )
        self._numbers = tuple(val)

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self,val):
        """
        Args:
            val (string): 'xmin xmax ymin ymax'
                ([[float,float],[float,float]])
        """
        if isinstance(val,basestring):
            try:
                val = [float(s) for s in val.split()]
                if len(val) != 4:
                    raise ValueError("Excpected four string floats")
                val = [[val[0],val[1]],[val[2],val[3]]]
            except ValueError, e:
                raise ValueError(
                    "Error in range definition. {}".format(e)
                )
        if len(val) != 2:
            raise ValueError("Expected two pairs for the range")
        if len(val[0]) != 2 or len(val[1]) != 2:
            raise ValueError("Expected pair of pairs")
        if val[0][0] > val[0][1] or val[1][0] > val[1][1]:
            raise ValueError("{0} > {1} or {2} > {3}".format(
                    val[0][0],val[0][1],val[1][0],val[1][1]
                )
            )
        self._range = tuple([tuple(v) for v in val])

    def __str__(self):
        name =  "{0}_{1}-{2}".format(
            self.method,
            *self.numbers
        )
        if self.range == None:
            return name
        return "{0}_{1}-{2}-{3}-{4}".format(
            name,
            *(self.range[0]+self.range[1])
        )

### ANALYZE PROGRAM ###    
def analyze():
    """ Anlyze subprogram
    """
    Args.analyze_function()

def analyze_run():
    """ run subprogram of anlyze program

    Loads sumulations data and performs anlaysis.
    """
    file_datas_dict = load_datas(Args.data_files)
    plotables_dict = dict()
    for file_name, datas in file_datas_dict.viewitems():
        analized_datas = analyze_datas(datas,Args.analysis_attributes)
        plotables = ana_results_to_plotables(
            analized_datas,
            Args.analysis_attributes
        )
        if Args.dm_file_out:
            analysis_save_dm(
                analized_datas,
                plotables,
                Args.analysis_attributes,
                Args.dm_file_out
            )
        if Args.mat_file_out:
            analysis_save(
                plotables,
                Args.analysis_attributes,
                Args.mat_file_out
            )
        if Args.verbose:
            plotables_dict[file_name] = plotables
    if Args.verbose:
        ana_plot_figures(plotables_dict,Args.analysis_attributes)
        
def analyze_show():
    """ show subprogram of anylze program

    Displays plots for datas analyzed by run program. Works only with .mat file.
    """
    def mat_to_title(mat_file):
        mat_split = mat_file.split('_')
        while (mat_split.pop() not in ANALYSIS_METHODS):
            pass
        return string.join(mat_split,'_') + '*.mat'

    plotables = []
    for mat_file in Args.plotable_files:
        plotables.extend(
            [
                ((val.squeeze(),key), "{0}: {1}".format(mat_to_title(mat_file),key))
                for key,val in scipy.io.loadmat(mat_file).viewitems()
                if not (key.startswith('__') and key.endswith('__'))
            ]
        )
    ana_plot_graphs(*zip(*plotables),show=True)

### ANALYZE LOADS ###
def load_datas(data_files):
    """ Load data from list of files

    The function loads datas from files. The datas from all files are merged.
    In one data files.

    Args:
        data_files(list of str): list of names of files to be analyzed
    Returns:
        (dict) for complete description look in analyze_data documentation.
    """
    datas = dict()
    for file_name in data_files:
        try:
            datas[file_name] = np.load(file_name).item()
        except:
            raise EnvironmentError("Can't load {}.".format(file_name))
    if Args.join:
        return {'':ana_merge_datas(datas)}
    else:
        return datas

def ana_merge_datas(datas):
    """ Merges datas into one data

    Args:
        datas(dict of data): a key is data name and a value is data itself.
            Stracture of the data is specified in the documentation of
            analyze_data.
    Returns:
        (dict) simulation data dictionary
    """
    return {
        'searches':ana_merge_searches(datas),
        'senzory_map':ana_merge_senzory_map(datas)
    }

def ana_merge_searches(datas):
    """ Merges search pahts in datas

    Args:
        datas(dict of data): a key is data name and a value is data itself.
    Returns:
        (dict) dictionary of search paths
    Disclaimer:
        By merge, there isn't created new type. Only SHALLOW copy is used.
    """
    return dict([
        (name+'|'+path,data['searches'][path])
        for name,data in datas.viewitems()
            for path in data['searches'].viewkeys()
    ])

def ana_merge_senzory_map(datas):
    """ Extract senzory map from datas

    It loads senzory map from datas. The senzory maps in datas are merged
    to one senzory map for the purpose of analysis.


    Args:
        datas(list of data): datas is list of datas from data files.
    Returns:
        (np.array) senzory map for all data in datas
    """
#TODO: improve senzory map merging
    return iter(datas.viewvalues()).next()['senzory_map']

### DATA ANALYSIS ###
ANALYSIS_METHODS = ['FSD', 'SD']
def analyze_datas(data_dict,analysis_attributes):
    """ Datas from run to second step analysis

    Analyze datas in data_dict, wich are simulation results. The type
    of analysis performed is specified in analysis_attributes. It takes
    an analysis specification from analysis_attributes, performs analysis
    on data_dict and stores the result to the list.

    Args:
        data_dict (dict): datas with a simulation run results
            conaints:
                |-  'senozry_map': senzory locations used by the simulation
                L   'searches' : dictionary datas from searches.
                    L   '0', '1', ...: datas from one search.
                        L   'path': fixations path
        analysis_attributes (list of AnalysisAttributes)
    Returns:
        (list of analysis) One analysis consists of a pair, where first value
            indicate analysis data type and second value is data from the
            analysis. Analysis data type are:
                SMAP ... senzory map type. It is a matrix with (N,3) shape.
                    A row conains three values. First two are a location and
                    the last one is data value bound with the location.
                PTS ... points type. A matrix of a shape (N,2).
                    A row contains a point in 2D space.
    """
    analysis_methods = [
        ana_attr.method for ana_attr in analysis_attributes
    ]
    anal_fnc = {
        'FSD': dec_tupl(ana_fixations_spatial_distribution, 'SMAP'),
        'SD': dec_tupl(ana_saccades_distribution, 'PTS')
    }
    return [anal_fnc[ana_name](data_dict) for ana_name in analysis_methods]

def ana_fixations_spatial_distribution(data_dict):
    """ Fixations spatial distribution analysis (FSD)

    Args:
        data_dict(dict): data from the simulation run. For FSD analysis, data_dict has
            to contain 'senzory_map' and 'paht' in 'searches'. For more info about
            the data structure of data_dict, look at the description in analyze_datas.
    Returns:
        (np.array=SMAP)  A matrix of a shape (N,3). A row contains three values.
            First two represent a locaiton and the last value is number of
            the fixations for the location.
    """
    counter = collections.Counter()
    for search in data_dict['searches'].viewvalues():
        for fixation in search['path'][1:]: # Exclude first fixation. It is always (0,0).
            counter[tuple(fixation)] += 1
    frequency_map = np.zeros((data_dict['senzory_map'].shape[0],1))
    for i in xrange(len(frequency_map)):
        frequency_map[i] = counter[tuple(data_dict['senzory_map'][i])]
    return np.hstack((data_dict['senzory_map'],frequency_map))

def ana_saccades_distribution(data_dict):
    """ Saccades distribution analysis (SD)

    Args:
        data_dict(dict): data from the simulation run. For SD analysis, data dict
            has to contain 'path' in 'searches'. For more info about the data
            structure of data_dict, look at the description in analyze_datas.

    Returns:
        (np.array=PTS) An array of all saccades in polar coordinate. The degree is used
        instead of radian. The saccades are stacked horizontaly, i.e. a row in the matrix
        is one saccade.
    """
#? Work with zero saccades?
#? Need to ask Filip
    saccades = list()
    for search in data_dict['searches'].viewvalues():
        for shaft, tip in zip(search['path'][:-1],search['path'][1:]):
            # saccades.append(compute_saccade(shaft,tip))
            x, d = compute_saccade(shaft,tip)
            if x != 0:
                saccades.append((x,d))
    return np.array(saccades)

def compute_saccade(shaft,tip):
    """ Compute saccade starting at shaft and ending in tip

    Returned saccade is in polar coordinate. The degree is used instead of radians.

    Args:
        shaft(np.array): starting position for saccade
        tip(np.array): terminal postition for saccade
    Returns:
        (tuple) saccade in polar coordinate
    """
    x, p  = cmath.polar(complex(*(tip - shaft)))
    return x, math.degrees(p)

### TO PLOTABLES [K] ###
def ana_results_to_plotables(ana_results,analysis_attributes):
    """ Covert result to plotable matrices

    The function takes results from the analysis and convert datas to matrices
    wich can be easily ploted.

    Args:
        ana_results (list of tuples): a list of pairs. A pair contains analysis
            data and string name of analysis data type. Supported analysis data
            types are SMAP and PTS.
        analysis_attributes (list of AnalysisAttributes):
    Returns:
        (list of tuple) List of pairs. The first  is plotable. Second is name
            of plotalbe type in strig. Implemented plotables:
                MESH ... returns mesh grids for X,Y,Z
                HIST ... returns matrix with histogram intervals for x and y axis
    """
    plot_attributes = [
        (ana_attr.numbers,ana_attr.range)
        for ana_attr in analysis_attributes
    ]
    plotable_fnc = {
        'SMAP': dec_tupl(create_mesh_grid, 'MESH'),
        'PTS': dec_tupl(create_histogram, 'HIST')
    }
    return [
        plotable_fnc[stype](data,*plt_attr)
        for (data, stype), plt_attr in zip(ana_results,plot_attributes)
    ]

def create_mesh_grid(val_loc, linspace, frame=None, method='cubic'):
    """ Create mash grid by interpolating the values

    The function interpolates the values to a mesh grid. The position of values
    in a grid are specifiend in the locations. A value is assumed to be positive.

    Args:
        val_loc (np.array): Array of locations and their values. A row consists
            of three values. First two represent location and the third one
            is a value.
        linspace (pair of positive int): number of values for mesh grid along
            the axes. linspace[0] is for x axis, linspace[1] is for y axis.
        frame ([[xmin,xmax],[ymin,ymax)): If frame is None, than it is computed
            from locations, so that all the locations will fit in to the frame.
        method (str): method used for interpolation. Supported methods:
            nearest, linear, cubic. For the method information look at
                http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    Rerturns
        (np.array), (np.array), (np.array): mesh grid for x, y and z axis.
            Shape is equale to (linspace[1],linspace[0])
    Disclaimer:
        The axes follow Cartesian convention.
    """
    locations = val_loc[:,0:2]
    values = val_loc[:,2]
    if frame == None:
        frame = create_frame_for(locations)
    assert len(frame) == 2, "Precondition violation"
    assert len(frame[0]) == 2 and len(frame[1]) == 2, "Preconditio violation"
    assert frame[0][0] <= frame[0][1] and frame[1][0] <= frame[1][1], "Precondition violation"
    assert len(linspace) == 2, "Precondition violatio"
    assert linspace[0] > 0 and linspace[1] > 0, "Precondition violation"
    assert method in ['cubic','linear','nearest'], "Precondition violaiton"
    mesh_X, mesh_Y = np.meshgrid(
        np.linspace(frame[0][0],frame[0][1],linspace[0]),
        np.linspace(frame[1][0],frame[1][1],linspace[1]),
        indexing='xy'
    )
    mesh_Z = scipy.interpolate.griddata(
        locations, values, (mesh_X, mesh_Y), method=method, fill_value=0
    )
    return mesh_X, mesh_Y, mesh_Z

def create_frame_for(locations):
    """ Create fram for locations

    The function crates a frame so that all the locaitons will fit
    inside.

    Args:
        locations (np.array): locations for wich to create the frame
    Returns:
        (pair of positive float): size of frame. If the frame is centered
            at the origin, than all locations are covered.
    """
    assert locations.shape[1] == 2
    min_x, min_y = np.subtract(np.min(locations,axis=0),1)
    max_x, max_y = np.add(np.max(locations,axis=0),1)
    return [[min_x,max_x],[min_y,max_y]]

def create_histogram(points,nums,range=None):
    """ Create histogram for points in 2D

    Args:
        points (np.array): array of points. A point is stored in a row
        nums ([int,int]): the number of values along each dimension
        range ([[xmin,xmax],[ymin,ymax]]): number of bins along x and y axis.
    Returns:
        (np.array,np.array,np.array) Tuple of np.array. First array contains
        frequency distribution for locations. Second array conains number
        of bins for x axis. Third array contains number of bins for y axis.
    Disclaimer:
        Axes follows Cartesian convention, i.e. x is horizontal line (abscissa),
        and y is vertical line (ordinate).
    """
    if range != None:
        range=[range[1],range[0]]
    z, y, x = np.histogram2d(
        points[:,0],points[:,1],
        bins=np.subtract([nums[1],nums[0]],1), # convert nums to bins
        range=range
    )
#TODO: delete
#    print points.shape
#    print np.vstack({tuple(row) for row in points}).shape
    return z, x, y
    
### ANA SAVE [X] ###
#TODO test
def analysis_save(plotables,analysis_attributes,file_name):
    """ Save plotable matrices to .mat file

    Name of file is concatenation of file_name and name of plotable.

    Args:
        plotables (pair (list,list)): The pair contains plotable data and name of plotable
            type in string.
        analysis_attributes (list of AnalysisAttributes)
        file_name (string): output file name
    """
#TODO implement error handling
    for (plotable,plt_type), ana_attr in zip(plotables,analysis_attributes):
        scipy.io.savemat(
            "{0}_{1}".format(file_name,str(ana_attr)),
            { plt_type: plotable }
        )

def analysis_save_dm(analyzed_datas,plotables,analysis_attributes,file_name):
    """ Save data-mineable datas into file_name

    The functionn pics datas from analyzed_datas and plotables, wich could be
    used for datamining and save them into file_name as .npy file.
    Supported analysis:
        FSD - outputs SMAP

    Args:
        plotables (pair (list,list)): The pair contains plotable data and name of plotable
            type in string.
        analyzed_datas (list of pairs): List of analysis pair. The pair consit of
            analyzed data and name of data, e.g. SMAP, PTS ...
        analysis_attributes (list of AnalysisAttributes)
        file_name (string): output file name
    """
#TODO implement error handling
    save_dict = {}
    for i,ana_attr in enumerate(analysis_attributes):
        if ana_attr.method == 'FSD':
            save_dict['FSD_SMAP'] = analyzed_datas[i][0]
    np.save(file_name,save_dict)

### ANA PLOTS ###
def ana_plot_figures(plotables_dict,analysis_attributes):
    """ Plot figures

    Args:
        plotables_dict (dict): key is string. It is used as a title for the figure.
            A value is pair of plotable type with its name.
        analysis_attributes (list of AnalysisAttributes)
    """
    analysis_methods = [
        ana_attr.method for ana_attr in analysis_attributes
    ]
    for title, plotables in plotables_dict.viewitems():
        ana_plot_graphs(plotables,analysis_methods,title)
    plt.show()

def ana_plot_graphs(plotables,plotable_titles,figure_title=None,show=False):
    """ Plot plotables

    The functnion plots datas found in plotables. Type of plotables, wich are supported
    MESH and HIST.

    Args:
        plotables (list of tuple): List of pairs. First in a pair is a plotable.
            Second is a plotable type.
        plotable_titles (list of string): titles for plotables
        figure_title (string): title for figure window
    """
    axes = num_to_subplots_axes(len(plotables))
    fig = plt.figure()
    fig.suptitle(figure_title)
    for i, ((plotable,plot_type),ana_type) in enumerate(zip(plotables,plotable_titles)):
        if plot_type == 'MESH':
            #ax = plot_mesh_sub(fig, axes+(i+1,), *plotable)
            ax = plot_imshow_from_mesh_sub(fig, axes+(i+1,), *plotable)
            # Suplots indexing is from 1 => i+1
            ax.set_title(ana_type)
        elif plot_type == 'HIST':
            ax = plot_imshow_sub(
                fig, axes+(i+1,), plotable[0],
                (np.min(plotable[1]),np.max(plotable[1])),
                (np.min(plotable[2]),np.max(plotable[2]))
            )
            ax.set_title(ana_type)
        else:
            assert False, "Not implemented"
    if show:
        plt.show()
   
#TODO: joint with ploting in SHOW
def plot_mesh_sub(fig,sub,X,Y,Z):
#TODO description
    ax = fig.add_subplot(*sub,projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    ax.view_init(elev=90,azim=-90)
    ax.set_aspect('equal')
    fig.colorbar(surf,shrink=0.8)
    return ax

def plot_imshow_sub(fig,sub,V,x_int,y_int,vmin=None,vmax=None):
#TODO: description
    ax = fig.add_subplot(*sub)
    im = ax.imshow(V, extent=x_int+y_int, vmin=vmin, vmax=vmax,
        aspect='auto',origin='lower')
    fig.colorbar(im,shrink=0.8)
    return ax

def plot_imshow_from_mesh_sub(fig,sub,X,Y,Z,vmin=None,vmax=None):
    min_x, min_y = np.min(X), np.min(Y)
    max_x, max_y = np.max(X), np.max(Y)
    ax = fig.add_subplot(*sub)
    im = ax.imshow(
        Z, extent=(min_x,max_x,min_y,max_y), aspect='equal',
        vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(im,shrink=0.8)
    return ax

### ANA OTHER ###
def dec_tupl(fnc,*data):
    """ Function decorater

    Decorators returns tuple, where first value is fnc return value.

    Args:
        fnc (function)
        *data (vals): values to be filled into a tuple
    Returns:
        (function)
    """
    def wrapper(*args, **kwargs):
        return (fnc(*args,**kwargs),) + data
    return wrapper

def num_to_subplots_axes(num):
    """ Returns number of columns and rows for num

    Computes shape of matrix, wich could store num values.
    y axis is always wider.

    Args:
        num (int): number of values for potential matrix, for wich to
            compute number rows and columns.
    Return:
        ((int,int)) (rows,cols)
    """
    cols = int(math.ceil(math.sqrt(num)))
    rows = int(math.ceil(float(num) / cols))
    return rows, cols

#TODO delete after cleaning
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 
#END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: END NEW: 

### [G]RAPHICAL OUTPUT ###
#TODO: description
#TODO: allow beeter outputing, e.g. subplotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

def plot_mesh(X,Y,Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    ax.view_init(elev=90,azim=-90)
    ax.set_aspect('equal')
    plt.show()

def plot_imshow(X,xax,yax,vmin=None,vmax=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        X, extent=(np.min(xax),np.max(xax),np.min(yax),np.max(yax)),
        aspect='auto',vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(im)
    plt.show()

def plot_imshow_from_mesh(X,Y,Z,vmin=None,vmax=None):
    min_x, min_y = np.min(X), np.min(Y)
    max_x, max_y = np.max(X), np.max(Y)
    fig, ax = plt.subplots()
    im = ax.imshow(
        Z, extent=(min_x,max_x,min_y,max_y), aspect='equal',
        vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(im)
    plt.show()

### [P]ARAMETRS PARSER ###
import argparse

def parse_args():
    """ Parsing arguments

    The function parses arguments and stored them in to the Args.
    Top-level commands:
        run
    """
    global Args
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    pars_simulation(subparsers)
    pars_analyze(subparsers)
    Args = parser.parse_args()

def pars_simulation(subparsers):
    """ Parsing arguments for run command
    """

    def nonnegative(string):
        try:
            return str_to_nonnegative_float(string)
        except ValueError, Argument:
            raise argparse.ArgumentTypeError(Argument)

    def positive(string):
        try:
            return str_to_positive_int(string)
        except ValueError, Argument:
            raise argparse.ArgumentTypeError(Argument)

    def unit_interval(string):
        try:
            return str_to_unit_interval_float(string)
        except ValueError, Argument:
            raise argparse.ArgumentTypeError(Argument)

    def next_fixation_fnc(string):
        try:
            return str_to_next_fixation_fnc(string)
        except ValueError, Argument:
            raise argparse.ArgumentTypeError(Argument)
    #simulation
    parser_simulation = subparsers.add_parser('simulation',help="simulation part of the program")
    parser_simulation.set_defaults(function=simulation)
    parser_simulation.add_argument('-v', '--verbose', action='count',default=0,
        help="output informations about the simulation",dest='verbose')
    parser_simulation.add_argument('-c', '--conf-file', action='store',
        type=argparse.FileType('r'), help='configuration file',
        dest='conf_fin')
    parser_simulation.add_argument('-L', '--locations', action='store',
        type=str, help="file with target locations",
        metavar="LOCATIONS FILE", dest='senzory_map_file')
    group_dprime = parser_simulation.add_mutually_exclusive_group()
    group_dprime.add_argument('-D', '--dprime-map-file', action='store',
        type=argparse.FileType('r'), help="dprime map configuration file",
        metavar="DPRIME CONFIGURATION FILE", dest='dprime_map_fin')
    group_dprime.add_argument('-d', '--dprime-map', action='store',
        type=str, help="set dprime map as specified in the argument",
        metavar="SPECIFICATION", dest='dprime_map_str')
    group_dprime.add_argument('--dprime-map-fast', action='store',
        type=str, choices=Dprime_types,
        help="set dprime map on {} with default values".format(Dprime_types).translate(None,"'"),
        metavar="TYPE", dest='dprime_map_type')
#? -a obsolete. Delete?
    parser_simulation.add_argument('-a', '--algorithm', action='store',
        type=next_fixation_fnc,
        help="the algorithm used for choosing next fixation",
        dest='algorithm')
    parser_simulation.add_argument('-n', '--search-number', action='store',
        type=positive, help="the number of search simulations",
        dest='num_of_searches')
    parser_simulation.add_argument('-t', '--threshold', action='store',
        type=unit_interval, help="detection threshold for simulator",
        dest='threshold')
    simulation_subparsers = parser_simulation.add_subparsers()
    # run
    parser_run = simulation_subparsers.add_parser('run', help="run the simulation")
    parser_run.set_defaults(simulation_function=run)
    interactive_group = parser_run.add_mutually_exclusive_group()
    interactive_group.add_argument('-i', '--interactive', action='store_true',
        help="interactive mode")
    interactive_group.add_argument('-T', '--timer', action='store',
        nargs='?', default=0.0, type=nonnegative,
        help="the timer for output refresh")
    parser_run.add_argument('-o', '--out-file', action='store',
        type=str, help="output file name. File type will be .npz",
        metavar="FILE", dest='output_file')
    # show
    parser_show = simulation_subparsers.add_parser('show',
        help="show the similation internal variables")
    parser_show.set_defaults(simulation_function=show_data)
    show_subparsers = parser_show.add_subparsers()
    # dprime
    parser_dprime = show_subparsers.add_parser('dprime', help="show dprime")
    parser_dprime.set_defaults(data_to_show='dprime')
    group_dprime = parser_dprime.add_mutually_exclusive_group()
    group_dprime.add_argument('-U', '--upper-bound', action='store',
        type=nonnegative, help="upper bound for plot bar",
        metavar="UPPER BOUND", dest='upper_bound')
    group_dprime.add_argument('--out-mat', action='store',
        type=str, help="store as mesh grid into matlab type file",
        metavar="FILE", dest='mat_file_out')
    parser_dprime.add_argument('-S', '--size', action='store', nargs=2, default=(100,100),
        type=positive, help='size of grid', metavar=("x","y"), dest='grid_size')

def pars_analyze(subparsers):
    def analysis_attributes(s):
        try:
            return AnalysisAttributes(string.join(s.split(':')))
        except ValueError, Argument:
            raise argparse.ArgumentTypeError(
                "Error in {0}. {1}".format(s,Argument)
            )

    parser_analyze = subparsers.add_parser('analyze', help="analysis of simulation datas")
    parser_analyze.set_defaults(function=analyze)
    analyze_subparsers = parser_analyze.add_subparsers()
    # run
    parser_run = analyze_subparsers.add_parser('run', help="run analysis")
    parser_run.set_defaults(analyze_function=analyze_run)
    parser_run.add_argument('-f', '--file', nargs='+', type=str, required=True,
        help='files to analyze', metavar='FILE', dest='data_files')
    parser_run.add_argument('-m', '--method', nargs='+',
        type=analysis_attributes, required=True, help="analysis methods.",
        metavar="METHOD", dest='analysis_attributes')
    parser_run.add_argument('-o', '--out-mat', action='store', type=str,
        help="store analysis in .mat file", metavar="FILE",
        dest='mat_file_out')
    parser_run.add_argument('-j', '--join', action='store_true',
        help="analze datas in files together")
    parser_run.add_argument('-v', '--verbose', action='store_true', 
        help="output informations about the simulation")
    parser_run.add_argument('-p', '--plot-one', action='store_true',
        help="plot output into one graph")
    parser_run.add_argument('-d', '--out-data-mining', action='store',
        help="output data for datamining. Implemented only for FSD.", metavar="FILE",
        dest='dm_file_out')
    # show
    parser_show = analyze_subparsers.add_parser('show', help="show analyses results")
    parser_show.set_defaults(analyze_function=analyze_show)
    parser_show.add_argument('plotable_files', nargs='+', type=str,
        help='files to plot', metavar='FILE')

### MAIN [Z] ###
# global read: Args
if __name__ == "__main__":
    parse_args()
    Args.function()
