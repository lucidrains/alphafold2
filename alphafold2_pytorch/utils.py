# utils for working with 3d-protein structures
import os
import re
import numpy as np
import torch
import contextlib
from functools import wraps
from einops import rearrange, repeat
# import torch_sparse # only needed for sparse nth_deg adj calculation

# bio 
from Bio import SeqIO
import itertools
import string

# sidechainnet

from sidechainnet.utils.sequence import ProteinVocabulary, ONE_TO_THREE_LETTER_MAP
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, BB_BUILD_INFO, SC_BUILD_INFO
from sidechainnet.structure.StructureBuilder import _get_residue_build_iter

# custom
import mp_nerf

# build vocabulary

VOCAB = ProteinVocabulary()

# constants

import alphafold2_pytorch.constants as constants

# helpers

def exists(val):
    return val is not None

# constants: same as in alphafold2.py

DISTANCE_THRESHOLDS = torch.linspace(2, 20, steps = constants.DISTOGRAM_BUCKETS)

# distance binning function

def get_bucketed_distance_matrix(coords, mask, num_buckets = constants.DISTOGRAM_BUCKETS, ignore_index = -100):
    distances = torch.cdist(coords, coords, p=2)
    boundaries = torch.linspace(2, 20, steps = num_buckets, device = coords.device)
    discretized_distances = torch.bucketize(distances, boundaries[:-1])
    discretized_distances.masked_fill_(~(mask[..., None] & mask[..., None, :]), ignore_index)
    return discretized_distances

# decorators

def set_backend_kwarg(fn):
    @wraps(fn)
    def inner(*args, backend = 'auto', **kwargs):
        if backend == 'auto':
            backend = 'torch' if isinstance(args[0], torch.Tensor) else 'numpy'
        kwargs.update(backend = backend)
        return fn(*args, **kwargs)
    return inner

def expand_dims_to(t, length = 3):
    if length == 0:
        return t
    return t.reshape(*((1,) * length), *t.shape) # will work with both torch and numpy

def expand_arg_dims(dim_len = 3):
    """ pack here for reuse. 
        turns input into (B x D x N)
    """
    def outer(fn):
        @wraps(fn)
        def inner(x, y, **kwargs):
            assert len(x.shape) == len(y.shape), "Shapes of A and B must match."
            remaining_len = dim_len - len(x.shape)
            x = expand_dims_to(x, length = remaining_len)
            y = expand_dims_to(y, length = remaining_len)
            return fn(x, y, **kwargs)
        return inner
    return outer

def invoke_torch_or_numpy(torch_fn, numpy_fn):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            backend = kwargs.pop('backend')
            passed_args = fn(*args, **kwargs)
            passed_args = list(passed_args)
            if isinstance(passed_args[-1], dict):
                passed_kwargs = passed_args.pop()
            else:
                passed_kwargs = {}
            backend_fn = torch_fn if backend == 'torch' else numpy_fn
            return backend_fn(*passed_args, **passed_kwargs)
        return inner
    return outer

@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# preprocess data

def get_atom_ids_dict():
    """ Get's a dict mapping each atom to a token. """
    ids = set(["", "N", "CA", "C", "O"])

    for k,v in SC_BUILD_INFO.items():
        for name in v["atom-names"]:
            ids.add(name)
            
    return {k: i for i,k in enumerate(sorted(ids))}

def make_cloud_mask(aa):
    """ relevent points will be 1. paddings will be 0. """
    mask = np.zeros(constants.NUM_COORDS_PER_RES)
    # early stop if padding token
    if aa == "_":
        return mask
    # get num of atoms in aa
    n_atoms = 4+len( SC_BUILD_INFO[ ONE_TO_THREE_LETTER_MAP[aa] ]["atom-names"] )
    mask[:n_atoms] = 1
    return mask

def make_atom_id_embedds(aa, atom_ids):
    """ Return the tokens for each atom in the aa. """
    mask = np.zeros(constants.NUM_COORDS_PER_RES)
    # early stop if padding token
    if aa == "_":
        return mask
    # get atom id
    atom_list = ["N", "CA", "C", "O"] + SC_BUILD_INFO[ ONE_TO_THREE_LETTER_MAP[aa] ]["atom-names"]
    for i,atom in enumerate(atom_list):
        mask[i] = ATOM_IDS[atom]
    return mask


ATOM_IDS = get_atom_ids_dict()
CUSTOM_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                   "atom_id_embedd": make_atom_id_embedds(k, atom_ids=ATOM_IDS),
                  } for k in "ARNDCQEGHILKMFPSTWYV_"}

# common utils

# parsing to pdb for easier visualization - other example from sidechainnet is:
# https://github.com/jonathanking/sidechainnet/tree/master/sidechainnet/structure

def download_pdb(name, route):
    """ Downloads a PDB entry from the RCSB PDB. 
        Inputs:
        * name: str. the PDB entry id. 4 characters, capitalized.
        * route: str. route of the destin file. usually ".pdb" extension
        Output: route of destin file
    """
    os.system(f"curl https://files.rcsb.org/download/{name}.pdb > {route}")
    return route

def clean_pdb(name, route=None, chain_num=None):
    """ Cleans the structure to only leave the important part.
        Inputs: 
        * name: str. route of the input .pdb file
        * route: str. route of the output. will overwrite input if not provided
        * chain_num: int. index of chain to select (1-indexed as pdb files)
        Output: route of destin file.
    """
    import mdtraj
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
        chain_idxs = raw_prot.topology.select(f"chainid == {str(chain.index)}")
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
    import mdtraj
    # convert to numpy
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    # ensure (1, N, 3)
    if coords.shape[1] == 3:
        coords = coords.T
    coords = np.newaxis(coords, axis=0)
    # get pdb id and chain num
    pdb_name, chain_num = proteinnet_id.split("#")[-1].split("_")[:-1]
    pdb_destin = "/".join(route.split("/")[:-1])+"/"+pdb_name+".pdb"
    # download pdb file and select appropiate 
    download_pdb(pdb_name, pdb_destin)
    clean_pdb(pdb_destin, chain_num=chain_num)
    # load trajectory scaffold and replace coordinates - assumes same order
    scaffold = mdtraj.load_pdb(pdb_destin)
    scaffold.xyz = coords
    scaffold.save(route)
    return pdb_destin, route


def coords2pdb(seq, coords, cloud_mask, prefix="", name="af2_struct.pdb"):
    """ Turns coordinates into PDB files ready to be visualized. 
        Inputs:
        * seq: (L,) tensor of ints (sidechainnet aa-key pairs)
        * coords: (3, N) coords of atoms
        * cloud_mask: (L, C) boolean mask of occupied spaces in scn format
        * prefix: str. directory to save files.
        * name: str. name of destin file (ex: pred1.pdb)
    """
    scaffold = torch.zeros( cloud_mask.shape, 3 )
    scaffold[cloud_mask] = coords.cpu().float()
    # build structures and save
    pred = scn.StructureBuilder( seq, crd=scaffold ) 
    pred.to_pdb(prefix+name)


# adapted from https://github.com/facebookresearch/esm

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int):
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


# sidechainnet / MSA / other data utils

def ids_to_embed_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(ids_to_embed_input(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')

    if all(map(lambda c: isinstance(c, str), out)):
        return (None, ''.join(out))

    return out

def ids_to_prottran_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for ids in x:
        chars = ' '.join([id2aa[i] for i in ids])
        chars = re.sub(r"[UZOB]", "X", chars)
        out.append(chars)

    return out

def get_prottran_embedd(seq, model, tokenizer, device = None):
    from transformers import pipeline

    fe = pipeline('feature-extraction', model = model, tokenizer = tokenizer, device = (-1 if not exists(device) else device.index))

    max_seq_len = seq.shape[1]
    embedd_inputs = ids_to_prottran_input(seq.cpu().tolist())

    embedding = fe(embedd_inputs)
    embedding = torch.tensor(embedding, device = device)

    return embedding[:, 1:(max_seq_len + 1)]

def get_msa_embedd(msa, embedd_model, batch_converter, device = None):
    """ Returns the MSA_tr embeddings for a protein.
        Inputs: 
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: MSA_tr model (see train_end2end.py for an example)
        * batch_converter: MSA_tr batch converter (see train_end2end.py for an example)
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA
            * embedd_dim: number of embedding dimensions. 768 for MSA_Transformer
    """
    # use MSA transformer
    REPR_LAYER_NUM = 12
    device = seq.device
    max_seq_len = msa.shape[-1]
    embedd_inputs = ids_to_embed_input(msa.cpu().tolist())

    msa_batch_labels, msa_batch_strs, msa_batch_tokens = batch_converter(embedd_inputs)
    with torch.no_grad():
        results = embedd_model(msa_batch_tokens.to(device), repr_layers=[REPR_LAYER_NUM], return_contacts=False)
    # index 0 is for start token. so take from 1 one
    token_reps = results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :]
    return token_reps

def get_esm_embedd(seq, embedd_model, batch_converter, msa_data=None):
    """ Returns the ESM embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: ESM model (see train_end2end.py for an example)
        * batch_converter: ESM batch converter (see train_end2end.py for an example)
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for ESM-1b
            * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
    """
    # use ESM transformer
    device = seq.device
    REPR_LAYER_NUM = 33
    max_seq_len = seq.shape[-1]
    embedd_inputs = ids_to_embed_input(seq.cpu().tolist())

    batch_labels, batch_strs, batch_tokens = batch_converter(embedd_inputs)
    with torch.no_grad():
        results = embedd_model(batch_tokens.to(device), repr_layers=[REPR_LAYER_NUM], return_contacts=False)
    # index 0 is for start token. so take from 1 one
    token_reps = results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :].unsqueeze(dim=1)
    return token_reps


def get_t5_embedd(seq, tokenizer, encoder, msa_data=None, device=None):
    """ Returns the ProtT5-XL-U50 embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * tokenizer:  tokenizer model: T5Tokenizer
        * encoder: encoder model: T5EncoderModel
                 ex: from transformers import T5EncoderModel, T5Tokenizer
                     model_name = "Rostlab/prot_t5_xl_uniref50"
                     tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
                     model = T5EncoderModel.from_pretrained(model_name)
                     # prepare model 
                     model = model.to(device)
                     model = model.eval()
                     if torch.cuda.is_available():
                         model = model.half()
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for T5 models
            * embedd_dim: number of embedding dimensions. 1024 for T5 models
    """
    # get params and prepare
    device = seq.device if device is None else device
    embedd_inputs = ids_to_prottran_input(seq.cpu().tolist())
    
    # embedd - https://huggingface.co/Rostlab/prot_t5_xl_uniref50
    inputs_embedding = []
    shift_left, shift_right = 0, -1
    ids = tokenizer.batch_encode_plus(embedd_inputs, add_special_tokens=True,
                                                     padding=True, 
                                                     return_tensors="pt")
    with torch.no_grad():
        embedding = encoder(input_ids=torch.tensor(ids['input_ids']).to(device), 
                            attention_mask=torch.tensor(ids["attention_mask"]).to(device))
    # return (batch, seq_len, embedd_dim)
    token_reps = embedding.last_hidden_state[:, shift_left:shift_right].to(device)
    token_reps = expand_dims_to(token_reps, 4-len(token_reps.shape))
    return token_reps.float()


def get_all_protein_ids(dataloader, verbose=False):
    """ Given a sidechainnet dataloader for a CASP version, 
        Returns all the ids belonging to proteins.
        Inputs: 
        * dataloader: a sidechainnet dataloader for a CASP version
        Outputs: a set containing the ids for all protein entries. 
    """
    # store ids here
    ids = set([])
    # iterate for all batches
    for i,batch in tqdm(enumerate(dataloaders['train'])):
        # for breaking from 2 loops at once
        try:
            for i in range(batch.int_seqs.shape[0]):
                # check if all fragments are : 4_LETTER_PDB + NUM + CHAIN
                max_len_10 = len(batch.pids[i]) < 10 
                fragments  = [len(x) <= 4 for x in batch.pids[i].split("_")] 
                fragments_under_4 = sum(fragments) == len(fragments) # AND CONDITION
                # record id 
                if max_len_10 and fragments_under_4:
                    ids.add(batch.pids[i])
                else: 
                    if verbose:
                        print("skip:", batch.pids[i], "under 4", fragments)
        except StopIteration:
            break
    # returns set of ids
    return ids
    

def scn_cloud_mask(scn_seq, boolean=True, coords=None):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * scn_seq: (batch, length) sequence as provided by Sidechainnet package
        * boolean: whether to return as array of idxs or boolean values
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        Outputs: (batch, length, NUM_COORDS_PER_RES) boolean mask 
    """

    scn_seq = expand_dims_to(scn_seq, 2 - len(scn_seq.shape))
    # early check for coords mask
    if coords is not None: 
        batch_mask = ( rearrange(coords, '... (l c) d -> ... l c d', c=constants.NUM_COORDS_PER_RES) == 0 ).sum(dim=-1) < coords.shape[-1]
        if boolean:
            return batch_mask.bool()
        else: 
            return batch_mask.nonzero()

    # do loop in cpu
    device = scn_seq.device
    batch_mask = []
    scn_seq = scn_seq.cpu().tolist()
    for i, seq in enumerate(scn_seq):
        # get masks for each prot (points for each aa)
        batch_mask.append( torch.tensor([CUSTOM_INFO[VOCAB._int2char[aa]]['cloud_mask'] \
                                         for aa in seq]).bool().to(device) )
    # concat in last dim
    batch_mask = torch.stack(batch_mask, dim=0)
    # return mask (boolean or indexes)
    if boolean:
        return batch_mask.bool()
    else: 
        return batch_mask.nonzero()

    

def scn_backbone_mask(scn_seq, boolean=True, n_aa=3):
    """ Gets the boolean mask for N and CA positions. 
        Inputs: 
        * scn_seq: sequence(s) as provided by Sidechainnet package (int tensor/s)
        * n_aa: number of atoms in a backbone. (may include cbeta as 4th pos)
        * bool: whether to return as array of idxs or boolean values
        Outputs: (N_mask, CA_mask, C_mask)
    """
    wrapper = torch.zeros(*scn_seq.shape, n_aa).to(scn_seq.device)
    # N is the first atom in every AA. CA is the 2nd.
    wrapper[..., 0] = 1
    wrapper[..., 1] = 2
    wrapper[..., 2] = 3
    wrapper = rearrange(wrapper, '... l c -> ... (l c)')
    # find idxs
    N_mask  = wrapper == 1
    CA_mask = wrapper == 2
    C_mask  = wrapper == 3 
    if boolean:
        return N_mask, CA_mask, C_mask
    return torch.nonzero(N_mask), torch.nonzero(CA_mask), torch.nonzero(C_mask)

def scn_atom_embedd(scn_seq):
    """ Returns the token for each atom in the aa. 
        Inputs: 
        * scn_seq: sequence(s) as provided by Sidechainnet package (int tensor/s)
    """
    device = scn_seq.device
    batch_tokens = []
    # do loop in cpu
    scn_seq = scn_seq.cpu().tolist()
    for i,seq in enumerate(scn_seq):
        batch_tokens.append( torch.tensor([CUSTOM_INFO[VOCAB.int2char(aa)]["atom_id_embedd"] \
                                           for aa in seq]) )
    batch_tokens = torch.stack(batch_tokens, dim=0).long().to(device)
    return batch_tokens

def mat_input_to_masked(x, x_mask=None, edges_mat=None, edges=None, 
                          edge_mask=None, edge_attr_mat=None, 
                          edge_attr=None): 
    """ Turns the padded input and edges + mask into the
        non-padded inputs and edges.
        At least one of (edges_mat, edges) must be provided. 
        The same format for edges and edge_attr must be provided 
        (either adj matrix form or flattened form).
        Inputs: 
        * x: ((batch), N, D) a tensor of N nodes and D dims for each one
        * x_mask: ((batch), N,) boolean mask for x
        * edges: (2, E) optional. indices of the corresponding adjancecy matrix. 
        * edges_mat: ((batch), N, N) optional. adjacency matrix for x
        * edge_mask: optional. boolean mask of the same shape of either "edge_mat" or "edges".
        * edge_attr: (E, D_edge) optional. edge attributes of D_edge dims.
        * edge_attr_mat: ((batch), N, N) optional. adjacency matrix with features 
        Outputs: 
        * x: (N_, D) the masked node features
        * edge_index: (2, E_) the masked x-indices for the edges
        * edge_attr: (E_, D_edge) the masked edge attributes 
        * batch: (N_,) the corresponding index in the batch for each node 
    """
    # collapse batch dimension
    if len(x.shape) == 3:
        batch_dim = x.shape[1] 
        # collapse for x and its mask
        x = rearrange(x, 'b n d ... -> (b n) d ...')
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b n ... -> (b n) ...')
        else: 
            x_mask = torch.ones_like(x[..., 0]).bool()

        # collapse for edge indexes and attributes if needed
        if edges_mat is not None and edges is None:
            edges = torch.nonzero(edges_mat, as_tuple=False).t()
            edges = edges[1:] + edges[:1]*batch_dim
        # get the batch identifier for each node
        batch = (torch.arange(x.shape[0], device=x.device) // batch_dim)[x_mask]
    else:
        # edges to indices format
        if edges_mat is not None and edges is None:
            edges = torch.nonzero(edges_mat, as_tuple=False).t()
        # get the batch identifier for each node
        batch = torch.zeros(x.shape[0], device=x.device).to(x.device)

    # adapt edge attrs if provided
    if edge_attr_mat is not None and edge_attr is None: 
            edge_attr = edge_attr[edges_mat.bool()]
    # gen edge_mask if not provided
    if edge_mask is None:
        edge_mask = torch.ones_like(edges[-1]).bool()

    # begin applying masks
    x = x[x_mask]
    # process edge indexes: get square mat and remove all non-coding atoms
    max_num = edges.max().item()+1
    wrapper = torch.zeros(max_num, max_num).to(x.device)
    wrapper[edges[0][edge_mask], edges[1][edge_mask]] = 1
    wrapper = wrapper[x_mask, :][:, x_mask]
    edge_index = torch.nonzero(wrapper, as_tuple=False).t()
    # process edge attr
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    
    return x, edge_index, edge_attr, batch



def nth_deg_adjacency(adj_mat, n=1, sparse=False):
    """ Calculates the n-th degree adjacency matrix.
        Performs mm of adj_mat and adds the newly added.
        Default is dense. Mods for sparse version are done when needed.
        Inputs: 
        * adj_mat: (N, N) adjacency tensor
        * n: int. degree of the output adjacency
        * sparse: bool. whether to use torch-sparse module
        Outputs: 
        * edge_idxs: ij positions of the adjacency matrix
        * edge_attrs: degree of connectivity (1 for neighs, 2 for neighs^2, ... )
    """
    adj_mat = adj_mat.float()
    attr_mat = torch.zeros_like(adj_mat)
    new_adj_mat = adj_mat.clone()
        
    for i in range(n):
        if i == 0:
            attr_mat += adj_mat
            continue

        if i == 1 and sparse: 
            idxs = adj_mat.nonzero().t()
            vals = adj_mat[idxs[0], idxs[1]]
            new_idxs = idxs.clone()
            new_vals = vals.clone() 
            m, k, n = 3 * [adj_mat.shape[0]] # (m, n) * (n, k) , but adj_mats are squared: m=n=k            

        if sparse:
            new_idxs, new_vals = torch_sparse.spspmm(new_idxs, new_vals, idxs, vals, m=m, k=k, n=n)
            new_vals = new_vals.bool().float()
            # fill by indexes bc it's faster in sparse mode - will need an intersection function
            previous = attr_mat[new_idxs[0], new_idxs[1]].bool().float()
            attr_mat[new_idxs[0], new_idxs[1]] = (1 - previous)*(i+1)
        else:
            new_adj_mat = (new_adj_mat @ adj_mat).bool().float() 
            attr_mat.masked_fill( (new_adj_mat - attr_mat.bool().float()).bool(), i+1 )

    return new_adj_mat, attr_mat

def prot_covalent_bond(seqs, adj_degree=1, cloud_mask=None, mat=True, sparse=False):
    """ Returns the idxs of covalent bonds for a protein.
        Inputs 
        * seq: (b, n) torch long.
        * adj_degree: int. adjacency degree
        * cloud_mask: mask selecting the present atoms.
        * mat: whether to return as indexes of only atoms (PyG version)
               or matrices of masked atoms (for batched training). 
               for indexes, only 1 seq is supported.
        * sparse: bool. whether to use torch_sparse for adj_mat calc
        Outputs: edge_idxs, edge_types (degree of adjacency). 
    """
    device = seqs.device
    # set up container adj_mat (will get trimmed - less than 14)
    next_aa = NUM_COORDS_PER_RES
    adj_mat = torch.zeros(seqs.shape[0], *[seqs.shape[1]*NUM_COORDS_PER_RES]*2)
    # not needed to device since it's only for indices
    seq_list = seqs.cpu().tolist()
    for s,seq in enumerate(seq_list): 
        next_idx = 0
        for i,idx in enumerate(seq):
            aa_bonds = constants.AA_DATA[VOCAB._int2char[idx]]['bonds']
            # if no edges -> padding token -> finish bond creation for this seq
            if len(aa_bonds) == 0: 
                break
            # correct next position. for indexes functionality
            next_aa = max(aa_bonds, key=lambda x: max(x))[-1]
            # offset by pos in chain ( intra-aa bonds + with next aa )
            bonds = next_idx + torch.tensor( aa_bonds + [[2, next_aa]] ).t()
            next_idx += next_aa
            # delete link with next if final AA in seq
            if i == seqs.shape[1] - 1:
                bonds = bonds[:, :-1]
            # modify adj mat
            adj_mat[s, bonds[0], bonds[1]] = 1
        # convert to undirected
        adj_mat[s] = adj_mat[s] + adj_mat[s].t()
        # do N_th degree adjacency
        adj_mat, attr_mat = nth_deg_adjacency(adj_mat, n=adj_degree, sparse=sparse)

    if mat: 
        # return the full matrix/tensor
        return attr_mat.bool().to(seqs.device), attr_mat.to(device)
    else:
        edge_idxs = attr_mat[0].nonzero().t().long()
        edge_types = attr_mat[0, edge_idxs[0], edge_idxs[1]]
        return edge_idxs.to(seqs.device), edge_types.to(seqs.device)


def sidechain_container(seqs, backbones, atom_mask, cloud_mask=None, padding_tok=20):
    """ Gets a backbone of the protein, returns the whole coordinates
        with sidechains (same format as sidechainnet). Keeps differentiability.
        Inputs: 
        * seqs: (batch, L) either tensor or list
        * backbones: (batch, L*n_aa, 3): assume batch=1 (could be extended (?not tested)).
                     Coords for (N-term, C-alpha, C-term, (c_beta)) of every aa.
        * atom_mask: (14,). int or bool tensor specifying which atoms are passed.
        * cloud_mask: (batch, l, c). optional. cloud mask from scn_cloud_mask`.
                      sets point outside of mask to 0. if passed, else c_alpha
        * padding: int. padding token. same as in sidechainnet: 20
        Outputs: whole coordinates of shape (batch, L, 14, 3)
    """
    atom_mask = atom_mask.bool().cpu().detach()
    cum_atom_mask = atom_mask.cumsum(dim=-1).tolist()

    device = backbones.device
    batch, length = backbones.shape[0], backbones.shape[1] // cum_atom_mask[-1]
    predicted  = rearrange(backbones, 'b (l back) d -> b l back d', l=length)

    # early check if whole chain is already pred
    if cum_atom_mask[-1] == 14:
        return predicted

    # build scaffold from (N, CA, C, CB) - do in cpu
    new_coords = torch.zeros(batch, length, constants.NUM_COORDS_PER_RES, 3)
    predicted  = predicted.cpu() if predicted.is_cuda else predicted

    # fill atoms if they have been passed
    for i,atom in enumerate(atom_mask.tolist()):
        if atom:
            new_coords[:, :, i] = predicted[:, :, cum_atom_mask[i]-1]

    # generate sidechain if not passed
    for s,seq in enumerate(seqs): 
        # format seq accordingly
        if isinstance(seq, torch.Tensor):
            padding = (seq == padding_tok).sum().item()
            seq_str = ''.join([VOCAB._int2char[aa] for aa in seq.cpu().numpy()[:-padding or None]])
        elif isinstance(seq, str):
            padding = 0
            seq_str = seq
        # get scaffolds - will overwrite oxygen since its position is fully determined by N-C-CA
        scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq_str, angles=None, device="cpu")
        coords, _ = mp_nerf.proteins.sidechain_fold(wrapper = new_coords[s, :-padding or None].detach(),
                                                    **scaffolds, c_beta = cum_atom_mask[4]==5)
        # add detached scn
        for i,atom in enumerate(atom_mask.tolist()):
            if not atom:
                new_coords[:, :-padding or None, i] = coords[:, i]

    new_coords = new_coords.to(device)
    if cloud_mask is not None:
        new_coords[torch.logical_not(cloud_mask)] = 0.

    # replace any nan-s with previous point location (or N if pos is 13th of AA)
    nan_mask = list(torch.nonzero(new_coords!=new_coords, as_tuple=True))
    new_coords[nan_mask[0], nan_mask[1], nan_mask[2]] = new_coords[nan_mask[0], 
                                                                   nan_mask[1],
                                                                   (nan_mask[-2]+1) % new_coords.shape[-1]] 
    return new_coords.to(device)


# distance utils (distogram to dist mat + masking)

def center_distogram_torch(distogram, bins=DISTANCE_THRESHOLDS, min_t=1., center="mean", wide="std"):
    """ Returns the central estimate of a distogram. Median for now.
        Inputs:
        * distogram: (batch, N, N, B) where B is the number of buckets.
        * bins: (B,) containing the cutoffs for the different buckets
        * min_t: float. lower bound for distances.
        Outputs:
        * central: (batch, N, N)
        * dispersion: (batch, N, N)
        * weights: (batch, N, N)
    """
    shape, device = distogram.shape, distogram.device
    # threshold to weights and find mean value of each bin
    n_bins = ( bins - 0.5 * (bins[2] - bins[1]) ).to(device)
    n_bins[0]  = 1.5
    n_bins[-1] = 1.33*bins[-1] # above last threshold is ignored
    max_bin_allowed = torch.tensor(n_bins.shape[0]-1).to(device).long()
    # calculate measures of centrality and dispersion - 
    magnitudes = distogram.sum(dim=-1)
    if center == "median":
        cum_dist = torch.cumsum(distogram, dim=-1)
        medium   = 0.5 * cum_dist[..., -1:]
        central  = torch.searchsorted(cum_dist, medium).squeeze()
        central  = n_bins[ torch.min(central, max_bin_allowed) ]
    elif center == "mean":
        central  = (distogram * n_bins).sum(dim=-1) / magnitudes
    # create mask for last class - (IGNORE_INDEX)   
    mask = (central <= bins[-2].item()).float()
    # mask diagonal to 0 dist - don't do masked filling to avoid inplace errors
    diag_idxs = np.arange(shape[-2])
    central   = expand_dims_to(central, 3 - len(central.shape))
    central[:, diag_idxs, diag_idxs]  *= 0.
    # provide weights
    if wide == "var":
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes
    elif wide == "std":
        dispersion = ((distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes).sqrt()
    else:
        dispersion = torch.zeros_like(central, device=device)
    # rescale to 0-1. lower std / var  --> weight=1. set potential nan's to 0
    weights = mask / (1+dispersion)
    weights[weights != weights] *= 0.
    weights[:, diag_idxs, diag_idxs] *= 0.
    return central, weights


# distance matrix to 3d coords: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_mds.py#L279

def mds_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distogram is (N x N) and symmetric
        Outs: 
        * best_3d_coords: (batch x 3 x N)
        * historic_stresses: (batch x steps)
    """
    device, dtype = pre_dist_mat.device, pre_dist_mat.type()
    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    diag_idxs = np.arange(N)
    his = [torch.tensor([np.inf]*batch, device=device)]

    # initialize by eigendecomposition: https://www.lptmc.jussieu.fr/user/lesne/bioinformatics.pdf
    # follow : https://www.biorxiv.org/content/10.1101/2020.11.27.401232v1.full.pdf
    D = pre_dist_mat**2
    M =  0.5 * (D[:, :1, :] + D[:, :, :1] - D) 
    # do loop svd bc it's faster: (2-3x in CPU and 1-2x in GPU)
    # https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336
    svds = [torch.svd_lowrank(mi) for mi in M]
    u = torch.stack([svd[0] for svd in svds], dim=0)
    s = torch.stack([svd[1] for svd in svds], dim=0)
    v = torch.stack([svd[2] for svd in svds], dim=0)
    best_3d_coords = torch.bmm(u, torch.diag_embed(s).abs().sqrt())[..., :3]

    # only eigen - way faster but not weights
    if weights is None and eigen==True:
        return torch.transpose( best_3d_coords, -1, -2), torch.zeros_like(torch.stack(his, dim=0))
    elif eigen==True:
        if verbose:
            print("Can't use eigen flag if weights are active. Fallback to iterative")

    # continue the iterative way
    if weights is None:
        weights = torch.ones_like(pre_dist_mat)

    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        best_3d_coords = best_3d_coords.contiguous()
        dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2).clone()

        stress   = ( weights * (dist_mat - pre_dist_mat)**2 ).sum(dim=(-1,-2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[ dist_mat <= 0 ] += 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio
        B[:, diag_idxs, diag_idxs] += ratio.sum(dim=-1)

        # update
        coords = (1. / N * torch.matmul(B, best_3d_coords))
        dis = torch.norm(coords, dim=(-1, -2))

        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (his[-1] - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        his.append( stress / dis )

    return torch.transpose(best_3d_coords, -1,-2), torch.stack(his, dim=0)

def mds_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distrogram is (N x N) and symmetric
        Out:
        * best_3d_coords: (3 x N)
        * historic_stress 
    """
    if weights is None:
        weights = np.ones_like(pre_dist_mat)

    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    his = [np.inf]
    # init random coords
    best_stress = np.inf * np.ones(batch)
    best_3d_coords = 2*np.random.rand(batch, 3, N) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = np.linalg.norm(best_3d_coords[:, :, :, None] - best_3d_coords[:, :, None, :], axis=-3)
        stress   = (( weights * (dist_mat - pre_dist_mat) )**2).sum(axis=(-1, -2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio 
        B[:, np.arange(N), np.arange(N)] += ratio.sum(axis=-1)
        # update - double transpose. TODO: consider fix
        coords = (1. / N * np.matmul(best_3d_coords, B))
        dis = np.linalg.norm(coords, axis=(-1, -2))
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (best_stress - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        best_stress = stress / dis
        his.append(best_stress)

    return best_3d_coords, np.array(his)

def get_dihedral_torch(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Can't use torch.dot bc it does not broadcast
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) ) 


def get_dihedral_numpy(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return np.arctan2( ( (np.linalg.norm(u2, axis=-1, keepdims=True) * u1) * np.cross(u2,u3, axis=-1)).sum(axis=-1),  
                       ( np.cross(u1,u2, axis=-1) * np.cross(u2, u3, axis=-1) ).sum(axis=-1) ) 

def calc_phis_torch(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (batch, N) boolean mask for N-term positions
        * CA_mask: (batch, N) boolean mask for C-alpha positions
        * C_mask: (batch, N) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
        Note: use [0] since all prots in batch have same backbone
    """ 
    # detach gradients for angle calculation - mirror selection
    pred_coords_ = torch.transpose(pred_coords.detach(), -1 , -2).cpu()
    # ensure dims
    N_mask = expand_dims_to( N_mask, 2-len(N_mask.shape) )
    CA_mask = expand_dims_to( CA_mask, 2-len(CA_mask.shape) )
    if C_mask is not None: 
        C_mask = expand_dims_to( C_mask, 2-len(C_mask.shape) )
    else:
        C_mask = torch.logical_not(torch.logical_or(N_mask,CA_mask))
    # select points
    n_terms  = pred_coords_[:, N_mask[0].squeeze()]
    c_alphas = pred_coords_[:, CA_mask[0].squeeze()]
    c_terms  = pred_coords_[:, C_mask[0].squeeze()]
    # compute phis for every pritein in the batch
    phis = [get_dihedral_torch(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # return percentage of lower than 0
    if prop: 
        return torch.stack([(x<0).float().mean() for x in phis], dim=0 ) 
    return phis


def calc_phis_numpy(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (N, ) boolean mask for N-term positions
        * CA_mask: (N, ) boolean mask for C-alpha positions
        * C_mask: (N, ) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
    """ 
    # detach gradients for angle calculation - mirror selection
    pred_coords_ = np.transpose(pred_coords, (0, 2, 1))
    n_terms  = pred_coords_[:, N_mask.squeeze()]
    c_alphas = pred_coords_[:, CA_mask.squeeze()]
    # select c_term auto if not passed
    if C_mask is not None: 
        c_terms = pred_coords_[:, C_mask]
    else:
        c_terms  = pred_coords_[:, (np.ones_like(N_mask)-N_mask-CA_mask).squeeze().astype(bool) ]
    # compute phis for every pritein in the batch
    phis = [get_dihedral_numpy(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # return percentage of lower than 0
    if prop: 
        return np.array( [(x<0).mean() for x in phis] ) 
    return phis


# alignment by centering + rotation to compute optimal RMSD
# adapted from : https://github.com/charnley/rmsd/

def kabsch_torch(X, Y, cpu=True):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    device = X.device
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t()).detach()
    if cpu: 
        C = C.cpu()
    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = W.t()
    else: 
        V, S, W = torch.linalg.svd(C)
    
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W).to(device)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_

def kabsch_numpy(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    # center X and Y to the origin
    X_ = X - X.mean(axis=-1, keepdims=True)
    Y_ = Y - Y.mean(axis=-1, keepdims=True)
    # calculate convariance matrix (for each prot in the batch)
    C = np.dot(X_, Y_.transpose())
    # Optimal rotation matrix via SVD
    V, S, W = np.linalg.svd(C)
    # determinant sign for direction correction
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = np.dot(V, W)
    # calculate rotations
    X_ = np.dot(X_.T, U).T
    # return centered and aligned
    return X_, Y_


# metrics - more formulas here: http://predictioncenter.org/casp12/doc/help.html

def distmat_loss_torch(X=None, Y=None, X_mat=None, Y_mat=None, p=2, q=2,
                       custom=None, distmat_mask=None, clamp=None):
    """ Calculates a loss on the distance matrix - no need to align structs.
        Inputs: 
        * X: (N, d) tensor. the predicted structure. One of (X, X_mat) is needed.
        * X_mat: (N, N) tensor. the predicted distance matrix. Optional ()
        * Y: (N, d) tensor. the true structure. One of (Y, Y_mat) is needed.
        * Y_mat: (N, N) tensor. the predicted distance matrix. Optional ()
        * p: int. power for the distance calculation (2 for euclidean)
        * q: float. power for the scaling of the loss (2 for MSE, 1 for MAE, etc)
        * custom: func or None. custom loss over distance matrices. 
                  ex: lambda x,y: 1 - 1/ (1 + ((x-y))**2) (1 is very bad. 0 is good)
        * distmat_mask: (N, N) mask (boolean or weights for each ij pos). optional.
        * clamp: tuple of (min,max) values for clipping distance matrices. ex: (0,150)
    """
    assert (X is not None or X_mat is not None) and \
           (Y is not None or Y_mat is not None), "The true and predicted coords or dist mats must be provided"
    # calculate distance matrices
    if X_mat is None: 
        X = X.squeeze()
        if clamp is not None:
            X = torch.clamp(X, *clamp)
        X_mat = torch.cdist(X, X, p=p)
    if Y_mat is None: 
        Y = Y.squeeze()
        if clamp is not None:
            Y = torch.clamp(Y, *clamp)
        Y_mat = torch.cdist(Y, Y, p=p)
    if distmat_mask is None:
        distmat_mask = torch.ones_like(Y_mat).bool()

    # do custom expression if passed
    if custom is not None:
        return custom(X_mat.squeeze(), Y_mat.squeeze()).mean()
    # **2 ensures always positive. Later scale back to desired power
    else:
        loss = ( X_mat - Y_mat )**2 
        if q != 2:
            loss = loss**(q/2)
        return loss[distmat_mask].mean()

def rmsd_torch(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )

def rmsd_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    return np.sqrt( np.mean((X - Y)**2, axis=(-1, -2)) )

def gdt_torch(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    device = X.device
    if weights is None:
        weights = torch.ones(1,len(cutoffs))
    else:
        weights = torch.tensor([weights]).to(device)
    # set zeros and fill with values
    GDT = torch.zeros(X.shape[0], len(cutoffs), device=device)
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).float().mean(dim=-1)
    # weighted mean
    return (GDT*weights).mean(-1)

def gdt_numpy(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    if weights is None:
        weights = np.ones( (1,len(cutoffs)) )
    else:
        weights = np.array([weights])
    # set zeros and fill with values
    GDT = np.zeros( (X.shape[0], len(cutoffs)) )
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).mean(axis=-1)
    # weighted mean
    return (GDT*weights).mean(-1)

def tmscore_torch(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = max(15, X.shape[-1])
    d0 = 1.24 * (L - 15)**(1/3) - 1.8
    # get distance
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # formula (see wrapper for source): 
    return (1 / (1 + (dist/d0)**2)).mean(dim=-1)

def tmscore_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = max(15, X.shape[-1])
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    # get distance
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # formula (see wrapper for source): 
    return (1 / (1 + (dist/d0)**2)).mean(axis=-1)


def mdscaling_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, 
                    eigen=False, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # batched mds for full parallel 
    preds, stresses = mds_torch(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, eigen=eigen, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    # no need to caculate multiple mirrors - just correct Z axis
    phi_ratios = calc_phis_torch(preds, N_mask, CA_mask, C_mask, prop=True)
    to_correct = torch.nonzero( (phi_ratios < 0.5)).view(-1)
    # fix mirrors by (-1)*Z if more (+) than (-) phi angles
    preds[to_correct, -1] = (-1)*preds[to_correct, -1]
    if verbose == 2:
        print("Corrected mirror idxs:", to_correct)
            
    return preds, stresses


def mdscaling_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # batched mds for full parallel 
    preds, stresses = mds_numpy(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    # no need to caculate multiple mirrors - just correct Z axis
    phi_ratios = calc_phis_numpy(preds, N_mask, CA_mask, C_mask, prop=True)
    for i,pred in enumerate(preds):
        # fix mirrors by (-1)*Z if more (+) than (-) phi angles
        if phi_ratios < 0.5:
            preds[i, -1] = (-1)*preds[i, -1]
            if verbose == 2:
                print("Corrected mirror in struct no.", i)

    return preds, stresses


def lddt_ca_torch(true_coords, pred_coords, cloud_mask, r_0=15.):
    """ Computes the lddt score for each C_alpha.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * true_coords: (b, l, c, d) in sidechainnet format.
        * pred_coords: (b, l, c, d) in sidechainnet format.
        * cloud_mask : (b, l, c) adapted for scn format.
        * r_0: float. maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt for c_alpha scores (ranging between 0 and 1)
        See wrapper below.
    """
    device, dtype = true_coords.device, true_coords.type()
    thresholds = torch.tensor([0.5, 1, 2, 4], device=device).type(dtype)
    # adapt masks
    cloud_mask = cloud_mask.bool().cpu()
    c_alpha_mask  = torch.zeros(cloud_mask.shape[1:], device=device).bool() # doesn't have batch dim
    c_alpha_mask[..., 1] = True
    # container for c_alpha scores (between 0,1)
    wrapper = torch.zeros(true_coords.shape[:2], device=device).type(dtype)

    for bi, seq in enumerate(true_coords):
        # select atoms for study
        c_alphas = cloud_mask[bi]*c_alpha_mask # only pick c_alpha positions
        selected_pred = pred_coords[bi, c_alphas, :] 
        selected_target = true_coords[bi, c_alphas, :]
        # get number under distance
        dist_mat_pred = torch.cdist(selected_pred, selected_pred, p=2)
        dist_mat_target = torch.cdist(selected_target, selected_target, p=2) 
        under_r0_target = dist_mat_target < r_0
        compare_dists = torch.abs(dist_mat_pred - dist_mat_target)[under_r0_target]
        # measure diff below threshold
        score = torch.zeros_like(under_r0_target).float()
        max_score = torch.zeros_like(under_r0_target).float()
        max_score[under_r0_target] = 4.
        # measure under how many thresholds
        score[under_r0_target] = thresholds.shape[0] - \
                                 torch.bucketize( compare_dists, boundaries=thresholds ).float()
        # dont include diagonal
        l_mask = c_alphas.float().sum(dim=-1).bool()
        wrapper[bi, l_mask] = ( score.sum(dim=-1) - thresholds.shape[0] ) / \
                              ( max_score.sum(dim=-1) - thresholds.shape[0] )

    return wrapper


################
### WRAPPERS ###
################

@set_backend_kwarg
@invoke_torch_or_numpy(mdscaling_torch, mdscaling_numpy)
def MDScaling(pre_dist_mat, **kwargs):
    """ Gets distance matrix (-ces). Outputs 3d.  
        Assumes (for now) distrogram is (N x N) and symmetric.
        For support of ditograms: see `center_distogram_torch()`
        Inputs:
        * pre_dist_mat: (1, N, N) distance matrix.
        * weights: optional. (N x N) pairwise relative weights .
        * iters: number of iterations to run the algorithm on
        * tol: relative tolerance at which to stop the algorithm if no better
               improvement is achieved
        * backend: one of ["numpy", "torch", "auto"] for backend choice
        * fix_mirror: int. number of iterations to run the 3d generation and
                      pick the best mirror (highest number of negative phis)
        * N_mask: indexing array/tensor for indices of backbone N.
                  Only used if fix_mirror > 0.
        * CA_mask: indexing array/tensor for indices of backbone C_alpha.
                   Only used if fix_mirror > 0.
        * verbose: whether to print logs
        Outputs:
        * best_3d_coords: (3 x N)
        * historic_stress: (timesteps, )
    """
    pre_dist_mat = expand_dims_to(pre_dist_mat, 3 - len(pre_dist_mat.shape))
    return pre_dist_mat, kwargs

@expand_arg_dims(dim_len = 2)
@set_backend_kwarg
@invoke_torch_or_numpy(kabsch_torch, kabsch_numpy)
def Kabsch(A, B):
    """ Returns Kabsch-rotated matrices resulting
        from aligning A into B.
        Adapted from: https://github.com/charnley/rmsd/
        * Inputs: 
            * A,B are (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of shape (3 x N)
    """
    # run calcs - pick the 0th bc an additional dim was created
    return A, B

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(rmsd_torch, rmsd_numpy)
def RMSD(A, B):
    """ Returns RMSD score as defined here (lower is better):
        https://en.wikipedia.org/wiki/
        Root-mean-square_deviation_of_atomic_positions
        * Inputs: 
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
    return A, B

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(gdt_torch, gdt_numpy)
def GDT(A, B, *, mode="TS", cutoffs=[1,2,4,8], weights=None):
    """ Returns GDT score as defined here (highre is better):
        Supports both TS and HA
        http://predictioncenter.org/casp12/doc/help.html
        * Inputs:
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * cutoffs: defines thresholds for gdt
            * weights: list containing the weights
            * mode: one of ["numpy", "torch", "auto"] for backend
        * Outputs: tensor/array of size (B,)
    """
    # define cutoffs for each type of gdt and weights
    cutoffs = [0.5,1,2,4] if mode in ["HA", "ha"] else [1,2,4,8]
    # calculate GDT
    return A, B, cutoffs, {'weights': weights}

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(tmscore_torch, tmscore_numpy)
def TMscore(A, B):
    """ Returns TMscore as defined here (higher is better):
        >0.5 (likely) >0.6 (highly likely) same folding. 
        = 0.2. https://en.wikipedia.org/wiki/Template_modeling_score
        Warning! It's not exactly the code in:
        https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp
        but will suffice for now. 
        Inputs: 
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * mode: one of ["numpy", "torch", "auto"] for backend
        Outputs: tensor/array of size (B,)
    """
    return A, B
