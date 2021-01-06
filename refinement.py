# will use FastRelax routine to refine structure
import os
import json
import mdtraj
# installation instructs in readme
# TODO: relocate this to readme: http://www.pyrosetta.org/dow
import pyrosetta

def downloadPDB(name, route):
    """ Downloads a PDB entry from the RCSB PDB. 
        Inputs:
        * name: str. the PDB entry id. 4 characters
        * route: str. route of the destin file. usually ".pdb" extension
        Output: route of destin file
    """
    os.system("curl https://files.rcsb.org/download/{0}.pdb > {1}".format(name, route))
    return route

def clean_pdb(name, route=None, chain_num=None):
    """ Cleans the structure to only leave the important part.
        Inputs: 
        * name: str. route of the input .pdb file
        * route: str. route of the output. will overwrite input if not provided
        * chain_num: int. index of chain to select (1-indexed as pdb files)
        Output: route of destin file.
    """
    destin = route if route is not None else name
    # read input
    raw_prot = mdtraj.load_pdb(name)
    # iterate over prot and select the specified chains
    idxs = []
    for chain in raw_prot.topology.chains:
        # if arg passed, only select that chain
        if chain_num is not None:
            if chain_num != chain.index:
                continue
        # select indexes of chain
        chain_idxs = raw_prot.topology.select("chainid == {0}".format(chain.index))
        idxs.extend( chain_idxs.tolist() )
    # sort: topology and xyz selection are ordered
    idxs = sorted(idxs)
    # get new trajectory from the sleected subset of indexes and save
    prot = mdtraj.Trajectory(xyz=raw_prot.xyz[:, idxs], 
                             topology=raw_prot.topology.subset(idxs))
    prot.save(destin)
    return destin

def custom2pdb(coords, proteinnet_id, route):
    """ Takes a custom representation and turns into a .pdb file. 
        Inputs:
        * coords: array/tensor of shape (3 x N) or (N x 3). in Angstroms.
                  same order as in the proteinnnet is assumed (same as raw pdb file)
        * proteinnet_id: str. proteinnet id format (<class>#<pdb_id>_<chain_number>_<chain_id>)
                         see: https://github.com/aqlaboratory/proteinnet/
        * route: str. destin route.
        Output: tuple of routes: (original, generated) for the structures. 
    """
    # convert to numpy
    if isinstance(coords, torch.Tensor):
    	coords = coords.detach().cpu().numpy()
    # ensure (1, N, 3)
    if coords.shape[1] == 3:
    	coords = coords.T
    coords = np.newaxis(coords, axis=0)
    # get pdb id and chain num
    pdb_name, china_num = proteinnet_id.split("#")[-1].split("_")[:-1]
    pdb_destin = "/".join(route.split("/")[:-1])+"/"+pdb_name+".pdb"
    # download pdb file and select appropiate 
    downloadPDB(pdb_name, pdb_destin)
    clean_pdb(pdb_destin, chain_num=chain_num)
    # load trajectory scaffold and replace coordinates - assumes same order
    scaffold = mdtraj.load_pdb(pdb_destin)
    scaffold.xyz = coords
    scaffold.save(route)
    return pdb_destin, route


#####################
### ROSETTA STUFF ###
#####################


def pdb2rosetta(route):
    """ Takes pdb file route(s) as input and returns rosetta pose(s). 
        Input:
        * route: list or string.
        Output: list of 1 or many according to input
       """
       if isinstance(route, str):
        return [pyrosetta.io.pose_from_pdb(route)]
    else:
        return list(pyrosetta.io.poses_from_files(route))

def rosetta2pdb(pose, route, verbose=True):
    """ Takes pose(s) as input and saves pdb(s) to disk.
        Input:
        * pose: list or string. rosetta poses object(s).
        * route: list or string. destin filenames to be written.
        * verbose: bool. warns if lengths dont match and @ every write.
        Inspo:
        * https://www.rosettacommons.org/demos/latest/tutorials/input_and_output/input_and_output#controlling-output_common-structure-output-files_pdb-file
        * https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.rosetta.core.io.pdb.html#pyrosetta.rosetta.core.io.pdb.dump_pdb
    """
    # convert to list
    pose  = [pose] if isinstance(pose, str) else pose
    route = [route] if isinstance(route, str) else route
    # check lengths and warn if necessary
    if verbose and ( len(pose) != len(route) ):
        print("Length of pose and route are not the same. Will stop at the minimum.")
    # convert and save
    for i,pos in enumerate(pose):
        pyrosetta.rosetta.core.io.pdb.dump_pdb(pos, route[i])
        if verbose:
            print("Saved structure @ "+route)
    return

def run_fast_relax(config_route, pose):
    """ Runs the Fast-Relax pipeline.
        * config_route: route to json file with config
        * pose: rosetta pose to run the pipeline on
        Output: rosetta pose
    """
    config = json.load(config_route)
    raise NotImplementedError("Last step. Not implemented yet.")

