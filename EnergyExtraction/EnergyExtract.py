"""This module contains function to compute LJ and Coulomb potential
        from an xtc trajectory
"""
from openmm.app import GromacsTopFile
from openmm.app import PDBFile, PME, HBonds, CutoffPeriodic
from openmm import unit, NonbondedForce, CustomNonbondedForce, LangevinMiddleIntegrator
from scipy.special import erfcinv
from openmm.app import ForceField, Simulation
from scipy.special import erfc
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS
import numpy as np
import pandas as pd
from numba import njit
from math import erfc as matherfc
from math import erf as matherf
from math import sqrt
from numba import prange
import time


@njit(cache=True, parallel=True)
def get_EperResidue_numba( 
                    positions, 
                    resids,
                    indices, 
                    nbindices, 
                    Acoef, 
                    Bcoef, 
                    charges,
                    neigh_res,
                    cutoff,
                    beta,                    
                    exc_begin,
                    exc_i,
                    exc_j,
                    exc_qprod,
                    exc_aij,
                    exc_bij):
    """Function to compute per-residue Lennard-Jones and Coulomb energies using
    Numba for optimization. This function uses a neighbor list for atom pairs, 
    and then computes the energies for all atom pairs within those residues.

    Parameters
    ----------
    positions : np.ndarray(n_at, 3)
        Array of atom positions. Must be the positions of all atoms in the system
    resids : np.ndarray(n_at)
        Array of residue indices for each atom.
    indices : np.ndarray(n_at)
        Array of atom indices. This allows for flexibility in selecting a subset of atoms for energy calculations.
    n_res :  int
        Total number of residues.
    nbindices : np.ndarray
        Array of lj nonbonded indices for each atom.
    Acoef : np.ndarray
        A coefficients for Lennard-Jones potential.
    Bcoef : np.ndarray
            B coefficients for Lennard-Jones potential.
    charges : np.ndarray(n_at)
        Array of charges for each atom.
    neigh_res : np.ndarray
        Array of neighboring atom pairs.
    cutoff : float
        Cutoff distance for nonbonded interactions.
    beta : float
        Parameter for the error function in Coulomb potential.

    Returns
    -------
    _type_
        _description_
    """
    cutoff_2 = cutoff*cutoff
    cutoff_6 = cutoff_2*cutoff_2*cutoff_2
    inv_cutt6 = 1.0/cutoff_6 

    n_pairs = neigh_res.shape[0]
    out_lj = np.zeros(n_pairs, dtype=np.float32)
    out_coul = np.zeros(n_pairs, dtype=np.float32)


    for k in prange(neigh_res.shape[0]):
        i = neigh_res[k,0]
        j = neigh_res[k,1]
        mapped_i = indices[i]
        mapped_j = indices[j]
        if resids[i] != resids[j]:
            dx = positions[i,0] - positions[j,0]
            dy = positions[i,1] - positions[j,1]
            dz = positions[i,2] - positions[j,2]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < cutoff_2: #May delete
                r = sqrt(r2)
                inv_r = 1.0/r
                inv_r2 = inv_r*inv_r
                inv_r6 = inv_r2*inv_r2*inv_r2

                aij = Acoef[nbindices[mapped_i], nbindices[mapped_j]]
                bij = Bcoef[nbindices[mapped_i], nbindices[mapped_j]]
                qprod = charges[mapped_i] * charges[mapped_j]

                Coul = 138.935456 * qprod * (inv_r) * matherfc(beta * r) #-1/cutoff)  # Coulomb's constant in kJ·nm/(mol·e²)
                if mapped_i > mapped_j:
                    ii = mapped_j
                    jj = mapped_i
                else:  
                    ii = mapped_i
                    jj = mapped_j
                # Lookup in the encoded list
                start_idx = exc_begin[ii]
                final_idx = exc_begin[ii+1]
                # Lookup in the elements of the encoded list
                for s in range(start_idx,final_idx):
                    if exc_j[s] == jj and exc_i[s] == ii:
                        aij = 0#exc_aij[s] # Gromacs excludes all the interactions 1-2, 1-3, 1-4
                        bij = 0#exc_bij[s] # Gromacs excludes all the interactions 1-2, 1-3, 1-4 for LJ potentials
                        
                        #Coul = 138.935456 * qprod * (inv_r) * matherfc(beta * r)
                        Coul = -138.935456 * qprod * (inv_r) * matherf(beta * r)  # Coulomb correction for PME (May remove eventually) 
                        break

                out_lj[k] = ((inv_r6 * aij)**2 - inv_r6 * bij) - ((aij*inv_cutt6)**2 - (bij*inv_cutt6))  # Lennard-Jones potential with cutoff
                out_coul[k] = Coul

    return out_lj, out_coul



# Running with numba and residue optimization
# Warning, this potentally only work is the residues and numbered from 1-n_res without any jump
@njit(cache=True, parallel=True)
def get_EperResidue_numba_res( 
                    positions, 
                    resids,
                    n_res,
                    res_limits, 
                    nbindices, 
                    Acoef, 
                    Bcoef, 
                    charges,
                    neigh_res,
                    cutoff,
                    beta,
                    exc_begin,
                    exc_i,
                    exc_j,
                    exc_qprod,
                    exc_aij,
                    exc_bij,): # Notice that exc arrays have not been used because I realized that when computing energy with 
                                # gromacs, those corrections are not added. The corresponding exceptions are just deleted 

    """Function to compute per-residue Lennard-Jones and Coulomb energies using 
    Numba for optimization. This function uses a neighbor list only 
    for residues, and then computes the energies for all atom pairs within those residues.
    In this way, we avoid computing energies for atom pairs that are not in neighboring residues,
    which can significantly reduce the number of computations for large systems.

    Parameters
    ----------
    positions : np.ndarray(n_at, 3)
        Array of atom positions. Must be the positions of all atoms in the system
    resids : np.ndarray(n_at)
        Array of residue indices for each atom. (Should match with resindices? (To check))
    n_res : int
        Total number of residues.
    res_limits : np.ndarray(n_res)
        Array of indices that mark the end of each residue in the positions array.
    nbindices : np.ndarray(n_at)
        Array of nonbonded indices for each atom.
    Acoef : np.ndarray
        Array of A coefficients for Lennard-Jones potential.
    Bcoef : np.ndarray  
         Array of B coefficients for Lennard-Jones potential.
    charges : np.ndarray(n_at)        print(i,j,"Residues to work on")
        Array of charges for each atom.
    neigh_res : np.ndarray
        Array of neighboring residue pairs.
    cutoff : float
        Cutoff distance for nonbonded interactions.
    beta : float
        Parameter for the error function in Coulomb potential.
    exc_begin : np.ndarray
        Array of indices that mark the beginning of the exceptions for each atom.
    exc_i : np.ndarray
        Array of atom indices for the first atom in each exception.
    exc_j : np.ndarray
        Array of atom indices for the second atom in each exception.
    exc_qprod : np.ndarray
        Array of charge products for each exception.        print(i,j,"Residues to work on")
    exc_aij : np.ndarray
        Array of A^2 coefficients for each exception.
    exc_bij : np.ndarray
        Array of B coefficients for each exception.




    Returns
    -------
    LJ_mat : np.ndarray(n_res, n_res)
        Matrix of Lennard-Jones energies between residues.
    Coul_mat : np.ndarray(n_res, n_res)
        Matrix of Coulomb energies between residues.
    """
    
    # May consired to ask this quantities as input
    cutoff_2 = cutoff*cutoff
    cutoff_6 = cutoff_2*cutoff_2*cutoff_2
    inv_cutt6 = 1.0/cutoff_6

    n_pairs = neigh_res.shape[0]

    out_lj = np.zeros(n_pairs, dtype=np.float32)
    out_coul = np.zeros(n_pairs, dtype=np.float32)

    for k in prange(n_pairs):
        i = neigh_res[k,0] # These will be residue indices, not atom indices
        j = neigh_res[k,1] # These will be residue indices, not atom indices


        
        # Get the ids of the atoms in the residues
        begin_i = res_limits[i-1] if i > 0 else 0
        begin_j = res_limits[j-1] if j > 0 else 0

        lj_acc = 0
        coul_acc = 0

        if resids[begin_i] != resids[begin_j]:  # Check if the residues are different
            for l in range(begin_i, res_limits[i]):
                
                for m in range(begin_j, res_limits[j]):

                    dx = positions[l,0] - positions[m,0]
                    dy = positions[l,1] - positions[m,1]
                    dz = positions[l,2] - positions[m,2]
                    r2 = (dx*dx + dy*dy + dz*dz)
                    if r2 < cutoff_2:
                        r = sqrt(r2)
                        #print(r)
                        inv_r = 1.0/r
                        inv_r6 = inv_r**6
                        aij = Acoef[nbindices[l], nbindices[m]]
                        bij = Bcoef[nbindices[l], nbindices[m]]
                        qprod = charges[l] * charges[m]

                        Coul = 138.935456 * qprod * (inv_r) * matherfc(beta * r)#-1/cutoff)  # Coulomb's constant in kJ·nm/(mol·e²)

                        # Only add bonded exceptions if the residues are adjacent

                        if l > m:
                            ii = m
                            jj = l
                        else:  
                            ii = l
                            jj = m
                        # Lookup in the encoded list
                        start_idx = exc_begin[ii]
                        final_idx = exc_begin[ii+1]
                        # Lookup in the elements of the encoded list
                        for s in range(start_idx,final_idx):
                            if exc_j[s] == jj and exc_i[s] == ii:
                                aij = 0#exc_aij[s] # Gromacs excludes all the interactions 1-2, 1-3, 1-4
                                bij = 0#exc_bij[s] # Gromacs excludes all the interactions 1-2, 1-3, 1-4 for LJ potentials
                                #Coul = 138.935456 * qprod * (inv_r) * matherfc(beta * r)
                                Coul = -138.935456 * qprod * (inv_r) * matherf(beta * r)  # Coulomb correction for PME (May remove eventually) 
                                break
                        
                        

                        LJ_val = ((inv_r6 * aij)**2 - inv_r6 * bij) - ((aij*inv_cutt6)**2 - (bij*inv_cutt6))  # Lennard-Jones potential with cutoff


                        # New store approach:


                        lj_acc += LJ_val
                        coul_acc += Coul
            out_lj[k] = lj_acc
            out_coul[k] = coul_acc


    return out_lj, out_coul











class Simul():
    def __init__(self, 
                 pdb_file, 
                 xtc_file, 
                 tpr_file = None,
                 verbose = False):

        self.xtc_file = xtc_file
        self.tpr_file = tpr_file if tpr_file is not None else None
        self.pdb_file = pdb_file
        self.cutoff = 1.2
        self.resid_cutoff = 3
        self.emtol = 1e-5
        self.beta = erfcinv(self.emtol)/self.cutoff
        if not tpr_file:
            self.universe = mda.Universe(self.pdb_file, self.xtc_file)
        else:
            self.universe = mda.Universe(self.tpr_file, self.xtc_file)
        # Store numbre of atoms
        self.n_atoms = len(self.universe.atoms)

        # Store the resindices (Unique and starting from 0)        
        self.resids = self.universe.atoms.resindices

        # Store the number of residues
        self.n_res = len(self.universe.residues.resindices)

        # Turns on/off printing
        self.verbose = verbose

        # Get the limits of the residues in the atom list, to be used in the energy calculation
        resindices = self.universe.atoms.resids # Change to resindices for next versions
        res_limits = np.flatnonzero(np.diff(resindices)) + 1
        res_limits = np.append(res_limits, len(resindices))

        self.resid_limits = np.array(res_limits)

        # Create an inverse map of the original residues with the resindices
        # Notice that this map is not a function since original resids can have identical resids
        self.original_resids = self.universe.residues.resids 
        self.map_resids = {int(new_resid): int(resid_or)  for resid_or, new_resid in zip(self.original_resids, self.universe.residues.resindices)}

        

        

    def IncludeTopology(self, 
                        path_top = None, 
                        include_dir = None, 
                        openmm_param = False,
                        simulation = False):
        """Extract topology information from the GROMACS files or parametrize them
        with OpenMM ForceField under the CHARMM36 force field. Then, include the 
        topology information into the MDAnalysis Universe.

        Parameters
        ----------
        path_top : str, optional
            Path to the topology file (.top), by default None
        include_dir : str, optional
            Path to the attachment in the .top file, by default None
        openmm_param : bool, optional
            If true add parameters from charmm36 forcefield using openmm, by default False
        """
        
        if openmm_param: # Parametrizes the topology using openmm
            pdb = PDBFile(self.pdb_file)
            forcefield = ForceField("charmm36.xml")
            self.system = forcefield.createSystem(pdb.topology, 
                                                  nonbondedMethod=CutoffPeriodic, 
                                                  constraints=HBonds, 
                                                  nonbondedCutoff=1.2*unit.nanometer)
            count = 0
            for atom in pdb.topology.atoms():
                count += 1
                if count > 100:
                    break
                print(atom.index, atom.name, atom.residue.name, atom.residue.index, atom.element.symbol)

            
        else:  # uses gromacs files to build the topology/parametrization
            pdb = PDBFile(self.pdb_file)
            top = GromacsTopFile(path_top, 
                                 periodicBoxVectors=pdb.topology.getPeriodicBoxVectors(), 
                                 includeDir = include_dir)
            self.system = top.createSystem(nonbondedMethod=PME, 
                                       constraints=HBonds, 
                                       nonbondedCutoff=1.2*unit.nanometer)
        self.simulation_obj = None
        if simulation: # Create a simulation object needed to compute things wiht openmm
            integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.004*unit.picoseconds)
            self.simulation_obj = Simulation(pdb.topology, self.system, integrator)
        
        # Variable not used yet, still thinking if worth it or not
        self.optimization_methods = ["residue-optimized", "atom-optimized", "openmm", "python"]

        # Obtain topology information
        self.forces = self.system.getForces()
        self.custom_nb = None
        self.simulation = simulation if simulation else None

        # When using gromcas topology, openmm build the LJ parameters and exceptions in custom non bonded force
        for force in self.forces:
            if isinstance(force, NonbondedForce):
                self.nb = force
            if isinstance(force, CustomNonbondedForce):
                self.custom_nb = force

        # Get the information from the forcefield
        n = self.nb.getNumParticles()
        attributes = {"charges" : np.empty(n),
                      "radii" : np.empty(n),
                      "epsilons" : np.empty(n),
                      "nbindex" : np.empty(n, dtype=int)}
        
        # MAtrices to store the LJ parameters
        self.Acoef = None
        self.Bcoef = None
        if self.custom_nb is not None:
            acoef = self.custom_nb.getTabulatedFunction(0)
            bcoef = self.custom_nb.getTabulatedFunction(1)

            xsize, ysize, val_a = acoef.getFunctionParameters()
            xsize, ysize, val_b = bcoef.getFunctionParameters()
            self.Acoef = np.array(val_a).reshape((xsize, ysize))
            self.Bcoef = np.array(val_b).reshape((xsize, ysize))
        for i in range(n):
            q, s, e = self.nb.getParticleParameters(i)
            attributes["charges"][i] = q.value_in_unit(unit.elementary_charge)
            attributes["radii"][i] = s.value_in_unit(unit.nanometers)
            attributes["epsilons"][i] = e.value_in_unit(unit.kilojoule_per_mole)
            attributes["nbindex"][i] = self.custom_nb.getParticleParameters(i)[0]
        
        self.charges = attributes["charges"]
        self.radii = attributes["radii"]
        self.epsilons = attributes["epsilons"]
        self.nbindices = attributes["nbindex"]
        self.resindices = self.universe.atoms.resindices

        for attr in attributes:
            self.universe.add_TopologyAttr(attr, attributes[attr])

    



    # Compute exclusions for the nonbonded interactions
    # Still not implemented in energy calculation
    def get_exclusions(self):

        exclusions_residues = set()
        n_exceptions = self.nb.getNumExceptions()

        # Create arrays to hold the exception parameters
        # Machinery for adding exceptions
        id_i = np.empty(n_exceptions, dtype=np.int32)
        id_j = np.empty(n_exceptions, dtype=np.int32)
        sigmas = np.empty(n_exceptions)
        epsilons = np.empty(n_exceptions)
        qprod = np.empty(n_exceptions)

        exc = []


        for k in range(self.nb.getNumExceptions()):
            p1, p2, chargeprod, sigma, epsilon = self.nb.getExceptionParameters(k)
            id_i[k] = p1
            id_j[k] = p2
            exc.append([p1, p2])

            sigmas[k] = sigma.value_in_unit(unit.nanometer)
            epsilons[k] = epsilon.value_in_unit(unit.kilojoule_per_mole)
            qprod[k] = chargeprod.value_in_unit(unit.elementary_charge**2)

        # Lookup arrays are created but not used because of Gromacs missmatch
        self.exc = np.array(exc)
        order = np.lexsort((id_j, id_i))

        self.exc_i = id_i[order]
        self.exc_j = id_j[order]
        sigmas = sigmas[order]
        self.exc_sigma6 = sigmas*sigmas*sigmas*sigmas*sigmas*sigmas*sigmas

        self.exc_epsilons = epsilons[order]
        self.exc_qprod = qprod[order]
        self.exc_aij = np.sqrt(4*self.exc_epsilons*self.exc_sigma6*self.exc_sigma6)
        self.exc_bij = 4*self.exc_epsilons*self.exc_sigma6

        counts = np.bincount(id_i, minlength=self.n_atoms)
        begin = np.zeros(len(counts)+1, dtype=np.int32)
        begin[1:] = np.cumsum(counts)
        self.exc_begin = begin


        return exclusions_residues
    
        

    
    def get_Energy_perres(self, start = 0, stop = -1,
                           step = 1,
                            selection = "protein", 
                            ligand_selection = None):
        """Compute the energy of the system from the trajectory

        Parameters
        ----------
        start : int, optional
            Start frame, by default 0
        stop : int, optional
            Stop frame, by default -1 (last frame)
        step : int, optional
            Step size, by default 1
        selection: str or list of two elements, optional
            Atom selection string in MDAnalysis style, by default "protein"
            Exmples: "protein and chainID A" (This will compute the per residue energy of the selection)
                    ["chainID A", "chainID B"] (This will compute the per residue energy of the interaction chain A, chain B)
        ligand_selection: list, optional
            List containing the selection string for the ligand and the 
            atom name to be used for the energy calculation, by default None
            Example: ligand_selection = ["resname LIG", "C1"] (This will inlcude the ligand in the per residue energy
             calculation)

        Returns
        -------
        result_lj : np.ndarray
            Array of Lennard-Jones energies for each frame
        result_coul : np.ndarray
            Array of Coulomb energies for each frame
        """
        

        sel_string = selection if isinstance(selection, str) else " or ".join([f"({sel})" for sel in selection])
        if ligand_selection is not None:
            sel_string += f" or ({ligand_selection[0]})"

        

        sel_indices = self.universe.select_atoms(sel_string).residues.resindices
        

        select_indices = False
        if isinstance(selection, list):
            select_indices = [] 
            for select in selection:
                select_indices.append(self.universe.select_atoms(select).residues.resindices)
        



        
        # All atoms should be accounted, if not there may be problems with the energy 
        # calculation because the resindices/ indices wont match
        all_atoms = self.universe.select_atoms("all")

        # Selection of atoms we will work wiht including ligand
        subset = all_atoms.select_atoms(sel_string)
        resnames = list(set(subset.residues.resnames))
        water_string = ""

        # Add waters to the calculation if present in selection
        # This is needed because initial selectio of CA does not account for water representative. Notice that ligand represetative is in 
        # ligand_selection and ions should be included in case needed
        if "TIP3" in resnames:
            water_string = "or (resname TIP3 and name OH2)" 
        #ca_atoms = all_atoms.select_atoms(sel_string) # Only used for residue-optimized energy calculation
        
        # Select one representative atom for each residue (Needed to compute neighbor list optimization)
        ca_atoms = all_atoms.select_atoms(f"(({sel_string}) and name CA) or ({ligand_selection[0]} and name {ligand_selection[1]}) {water_string}") # Only used for residue-optimized energy calculation
        ca_atoms_ids = ca_atoms.indices


        Lj_data = []
        Coul_data = []
        for ts in self.universe.trajectory[start:stop:step]:
            if self.verbose:
                print(f"Frame {ts.frame}")

            # Get positions of all atoms in the system
            positions = all_atoms.positions/10 # Convert from Angstroms (MDAnalysis default units) to nanometers

            t1 = time.perf_counter()
            idx, idy, energies_per_res = self.call_Energy_resbased(positions=positions, 
                                                         ca_atoms_ids=ca_atoms_ids, 
                                                         selections = select_indices, 
                                                         selected_residues = sel_indices)
            t2 = time.perf_counter()
            print(f"Frame {ts.frame}: Energy calculation took {t2-t1:.4f} seconds####")
            res_lj_total = (
                np.bincount(idx, weights=energies_per_res[:, 0], minlength=self.n_res) 
                + np.bincount(idy, weights=energies_per_res[:, 0], minlength=self.n_res)
            )

            res_coul_total = (
                np.bincount(idx, weights=energies_per_res[:, 1], minlength=self.n_res) 
                + np.bincount(idy, weights=energies_per_res[:, 1], minlength=self.n_res)
            )

            Lj_data.append(res_lj_total) # Sum over all residues to get total energy for the frame
            Coul_data.append(res_coul_total) # Sum over all residues to get total energy for the frame

        result_lj = np.asarray(Lj_data, dtype = np.float32)
        
        result_coul = np.asarray(Coul_data, dtype = np.float32)
        
        return result_lj, result_coul




    def get_Energy_perres_atombased(self, start = 0, stop = -1,
                           step = 1,
                            selection = "protein", 
                            ):
        """Compute the energy of the system from the trajectory

        Parameters
        ----------
        start : int, optional
            Start frame, by default 0
        stop : int, optional
            Stop frame, by default -1 (last frame)
        step : int, optional
            Step size, by default 1
        selection: str or list of two elements, optional
            Atom selection string in MDAnalysis style, by default "protein"
            Exmples: "protein and chainID A" (This will compute the per residue energy of the selection)
                    ["chainID A", "chainID B"] (This will compute the per residue energy of the interaction chain A, chain B)


        Returns
        -------
        result_lj : np.ndarray
            Array of Lennard-Jones energies for each frame
        result_coul : np.ndarray
            Array of Coulomb energies for each frame
        """
        

        sel_string = selection if isinstance(selection, str) else " or ".join([f"({sel})" for sel in selection])


        self.sel_string = sel_string



        
        # In atom based we can account only for the atoms in sel_string but we have to make sure that 
        # Those atoms are consiguous in the order of the original pdb file. 
        # To be more specifically, the atoms 
        all_atoms = self.universe.select_atoms(sel_string)

        Lj_data = []
        Coul_data = []
        for ts in self.universe.trajectory[start:stop:step]:
            if self.verbose:
                print(f"Frame {ts.frame}")

            # Get positions of all atoms in the system
            #positions = all_atoms.positions/10 # Convert from Angstroms (MDAnalysis default units) to nanometers

            t1 = time.perf_counter()

            # Currently, it will compute for all the atoms/resids in the selection, later we can modify it to only incluse some by using selection
            idx, idy, energies_per_res = self.call_Energy_atombased(all_atoms)
            t2 = time.perf_counter()
            print(f"Frame {ts.frame}: Energy calculation took {t2-t1:.4f} seconds####")
            res_lj_total = (
                np.bincount(idx, weights=energies_per_res[:, 0], minlength=self.n_res) 
                + np.bincount(idy, weights=energies_per_res[:, 0], minlength=self.n_res)
            )

            res_coul_total = (
                np.bincount(idx, weights=energies_per_res[:, 1], minlength=self.n_res) 
                + np.bincount(idy, weights=energies_per_res[:, 1], minlength=self.n_res)
            )

            Lj_data.append(res_lj_total) # Sum over all residues to get total energy for the frame
            Coul_data.append(res_coul_total) # Sum over all residues to get total energy for the frame

        result_lj = np.asarray(Lj_data, dtype = np.float32)
        
        result_coul = np.asarray(Coul_data, dtype = np.float32)
        
        return result_lj, result_coul
    
    

    def get_Energy_pairwise(self, start = 0, stop = -1, step = 1, selection = "protein"):
        """Compute the energy of the system from the trajectory

        Parameters
        ----------
        start : int, optional
            Start frame, by default 0
        stop : int, optional
            Stop frame, by default -1 (last frame)
        step : int, optional
            Step size, by default 1
        selection: str, optional
            Atom selection string for MDAnalysis, by default "protein"

        Returns
        -------
        result_lj : np.ndarray
            Array of Lennard-Jones energies for each frame
        result_coul : np.ndarray
            Array of Coulomb energies for each frame
        """
        
        print("started_perres")
        sel_string = selection if isinstance(selection, str) else " or ".join([f"({sel})" for sel in selection])
        sel_indices = self.universe.select_atoms(sel_string).residues.resindices


        select_indices = False
        if isinstance(selection, list):
            select_indices = [] 
            for select in selection:
                select_indices.append(self.universe.select_atoms(select).residues.resindices)
        


        all_atoms = self.universe.select_atoms("all") 
        ca_atoms = all_atoms.select_atoms(sel_string) # Only used for residue-optimized energy calculation
        ca_atoms = all_atoms.select_atoms(f"({sel_string}) and name CA") # Only used for residue-optimized energy calculation
        ca_atoms_ids = ca_atoms.indices

        frames = []
        idxs = []
        idys = []
        Lj_data = []
        Coul_data = []
        for ts in self.universe.trajectory[start:stop:step]:
            if self.verbose:
                print(f"Frame {ts.frame}")

            # Get positions of all atoms in the system
            positions = all_atoms.positions/10 # Convert from Angstroms (MDAnalysis default units) to nanometers

            t1 = time.perf_counter()
            idx, idy, energies_per_res = self.call_Energy_resbased(positions=positions, 
                                                         ca_atoms_ids=ca_atoms_ids, 
                                                         selections = select_indices, 
                                                         selected_residues = sel_indices)
            t2 = time.perf_counter()
            print(f"Frame {ts.frame}: Energy calculation took {t2-t1:.4f} seconds####")

            frames.append(np.full(idx.shape, ts.frame, dtype=np.int32))
            idxs.append(idx)
            idys.append(idy)
            Lj_data.append(energies_per_res[:, 0]) # Store LJ energies for each residue pair
            Coul_data.append(energies_per_res[:, 1]) # Store Coulomb energies for each residue pair

        frames = np.concatenate(frames)
        idxs = np.concatenate(idxs)
        idys = np.concatenate(idys)
        Lj_data = np.concatenate(Lj_data)
        Coul_data = np.concatenate(Coul_data)


        
        return frames, idxs, idys, Lj_data, Coul_data
    

    def call_Energy_atombased(self,atoms):

        

        neigh_res = FastNS(self.cutoff, positions, box = self.universe.dimensions)


        positions = atoms.positions/10 # Convert from Angstroms (MDAnalysis default units) to nanometers
        resids = atoms.resindices
        indices = atoms.indices


        # Get neighboring residue pairs
        neigh_res = neigh_res.self_search()
        neigh_res = neigh_res.get_pairs()
        output_data = np.zeros((neigh_res.shape[0], 2), dtype=np.float32) # Store value for each residue pair, 0: res1, 1: res2, 2: LJ, 3: Coulomb

        output_data[:,0], output_data[:,1] = get_EperResidue_numba(positions,
                        resids,
                        indices, 
                        self.nbindices, 
                        self.Acoef, 
                        self.Bcoef, 
                        self.charges,
                        neigh_res,
                        self.cutoff,
                        self.beta,                    
                        self.exc_begin,
                        self.exc_i,
                        self.exc_j,
                        self.exc_qprod,
                        self.exc_aij,
                        self.exc_bij)
        
        return neigh_res[:,0], neigh_res[:,1], output_data


    
    def call_Energy_resbased(self,positions, ca_atoms_ids, selections = False, selected_residues = None):
        """Calculate residue-based Lennard-Jones and Coulomb energies using a neighbor list for residues.

        Parameters
        ----------
        positions : np.ndarray
            positions of all atoms in the system
        ca_atoms_ids : np.ndarray
            indices of all the CA atoms to be computed (Limited by the selections)
        selections : bool, optional
            If True receive a list of 2, whith the resindices of the groups (0based index), by default False
        selected_residues : list, optional
            List containing all the residues selected (merged groups, 0 based index), by default None # I may be able to ask only one of those

        Returns
        -------
        _type_
            _description_
        """

        neigh_res = FastNS(self.resid_cutoff, positions[ca_atoms_ids], box = self.universe.dimensions)
        
        # Get neighboring residue pairs
        neigh_res = neigh_res.self_search()
        neigh_res = neigh_res.get_pairs()
        if selections:
            # Filter intra residue interactions and interactions between residue groups
            group_mask = np.zeros(len(selected_residues), dtype=np.uint8)
            idx1 = np.searchsorted(selected_residues, selections[0])
            idx2 = np.searchsorted(selected_residues, selections[1])
            group_mask[idx1] |= 1
            group_mask[idx2] |= 2
            g0 = group_mask[neigh_res[:,0]]
            g1 = group_mask[neigh_res[:,1]]
            mask = (
                    (((g0 & 1) != 0) & ((g1 & 2) != 0)) |
                    (((g0 & 2) != 0) & ((g1 & 1) != 0))
                    )
            
            neigh_res = neigh_res[mask]

        # This will map the residue indices from the neighbor search to the original residue indices in the system
        neigh_res[:,0] = selected_residues[neigh_res[:,0]]
        neigh_res[:,1] = selected_residues[neigh_res[:,1]]

        output_data = np.zeros((neigh_res.shape[0], 2), dtype=np.float32) # Store value for each residue pair, 0: res1, 1: res2, 2: LJ, 3: Coulomb

        output_data[:,0], output_data[:,1] = get_EperResidue_numba_res(positions,
                                        self.resids,
                                        self.n_res, 
                                        self.resid_limits, 
                                        self.nbindices, 
                                        self.Acoef, 
                                        self.Bcoef, 
                                        self.charges,
                                        neigh_res,
                                        self.cutoff,
                                        self.beta,
                self.exc_begin,
                self.exc_i,
                self.exc_j,
                self.exc_qprod,
                self.exc_aij,
                self.exc_bij,)        
        

        return neigh_res[:,0], neigh_res[:,1], output_data
    
    def call_Energy_oneres(self, universe, resindex, begin, exc_i, exc_j): # The last three are exclusions encoded

        ca_atoms = universe.select_atoms(f"all and name CA") # Get all CA atoms in the system
        ca_interest = ca_atoms.select_atoms(f"resindex {resindex} and name CA") # Get the CA atom of the residue of interest

        positions = universe.atoms.positions/10 # Convert from Angstroms (MDAnalysis default units) to nanometers
        distances = np.linalg.norm(ca_atoms.positions/10 - ca_interest.positions/10, axis=1) # Compute distances from the residue of interest to all CA atoms

        # Get the resindices of the neighboring residues within the cutoff distance

        neighbor_indices = np.where(distances < self.resid_cutoff)[0]

        neigh_res = [[resindex, neigh]for neigh in neighbor_indices if resindex != neigh] # Create neighbor list for only resindex of interest

        neigh_res = np.array(neigh_res, dtype=np.int32)

        resindices = universe.atoms.resindices
        n_res = len(set(universe.atoms.residues.resindices))
        charges = universe.atoms.charges
        nbindices = universe.atoms.nbindices

        all_at_resindices = universe.atoms.resindices
        res_limits = np.flatnonzero(np.diff(all_at_resindices)) + 1

        res_limits = np.append(res_limits, len(all_at_resindices))
        resid_limits = np.array(res_limits)
        energy = np.zeros(n_res)


        lj, coul = get_EperResidue_numba_res( 
                        positions, 
                        resindices,
                        n_res, 
                        resid_limits,
                    nbindices, 
                    self.Acoef, # This is in termns of reduced lj parameters, no the new indexes does not affect
                    self.Bcoef, # since this is already mapped with the repurposed 
                    charges,
                    neigh_res,
                    self.cutoff,
                    self.beta,                    
                    begin,
                    exc_i,
                    exc_j,
                    None,
                    None,
                    None) # To reproduce gromacs results we dont need to consider exclusions, just zero them, that why none is fine

        #print(neigh_res[:,1], "neigh_res")
        energy[neigh_res[:,1]] =  lj+coul

        return energy

        
        
    

    def get_EperResidue(self, frame, positions, resids, nbindices):
        """First attemp to compute per-residue Lennard-Jones and Coulomb energies using MDAnalysis and FastNS for neighbor search.


        Parameters
        ----------
        frame : int
            Frame number
        positions : np.ndarray
            Array of atom positions. Must be the positions of all atoms in the system
        resids : np.ndarray
            Array of residue indices for each atom.
        nbindices : np.ndarray
            Array of lj values fro each atom pair.

        Returns
        -------
        LJ_mat : np.ndarray
            Matrix of Lennard-Jones energies between residues.
        Coul_mat : np.ndarray
            Matrix of Coulomb energies between residues.
        """
        #print(self.cutoff)
        neigh = FastNS(self.cutoff, positions/10, box = self.universe.dimensions)
        neigh_res = neigh.self_search()
        #print(type(neigh_res.get_pairs()))

        n_res = len(set(resids))

        LJ_mat = np.zeros((n_res, n_res))
        Coul_mat = np.zeros((n_res, n_res))


        for i,j in neigh_res.get_pairs():
            #print(i,j,resids[i], resids[j])
            if resids[i] == resids[j]:
                continue
            else:
                r = np.linalg.norm(positions[i] - positions[j])/10
                inv_r = 1.0/r
                inv_r6 = inv_r**6

                aij = self.Acoef[nbindices[i], nbindices[j]]
                bij = self.Bcoef[nbindices[i], nbindices[j]]

                qprod = self.universe.atoms.charges[i] * self.universe.atoms.charges[j]
                LJ_val = ((inv_r6 * aij)**2 - inv_r6 * bij) - ((aij/self.cutoff**6)**2 - (bij/self.cutoff**6))  # Lennard-Jones potential with cutoff
                LJ_mat[resids[i]-1, resids[j]-1] += LJ_val
                LJ_mat[resids[j]-1, resids[i]-1] += LJ_val

                Coul = 138.935456 * qprod * (inv_r) * erfc(self.beta * r)#-1/cutoff)  # Coulomb's constant in kJ·nm/(mol·e²)
                Coul_mat[resids[i]-1, resids[j]-1] += Coul
                Coul_mat[resids[j]-1, resids[i]-1] += Coul

        return LJ_mat, Coul_mat
        





    def get_Energy_opmm(self, start = 0, stop = -1, step = 1):
        """Compute the energy of the system from the trajectory using OpenMM

        Parameters
        ----------
        start : int, optional
            Start frame, by default 0
        stop : int, optional
            Stop frame, by default -1 (last frame)
        step : int, optional
            Step size, by default 1

        Returns
        -------
        dict
            Dictionary with the energies for each frame
        """
        energies = {"frame" : [], "LJ" : [], "Coulomb" : []}

        
        

        for i, force in  enumerate(self.forces):
            print(force)
            try:
                print(force.getEnergyFunction())
            except:
                continue
            if isinstance(force, CustomNonbondedForce):
                print(i,force)
                cforce = force
        print(cforce.getEnergyFunction())        
        
        cforce.addInteractionGroup({100},{107})









         
