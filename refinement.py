# will use FastRelax routine to refine structure
import json
# installation instructs in readme
# TODO: relocate this to readme: http://www.pyrosetta.org/dow
import pyrosetta


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

