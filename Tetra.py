import numpy as np
import alphashape, trimesh
from itertools import permutations, combinations

from chimerax.core.commands import CmdDesc
from chimerax.core.models import Model
from chimerax.surface import calculate_vertex_normals
from chimerax.core.commands import ListOf, SetOf, TupleOf, Or, RepeatOf, BoolArg, IntArg

ht3 = 0.81649658092772603273242802490196379732198249355222337614423086
ht2 = 0.86602540378443864676372317075293618347140262690519031402790349


# Class to represent all the required properties of newly created amino objects from pdb data.
# TODO: Validate the RMSD calculation methods
class Amino:
    def __init__(self, coords, obj):
        self.nh, self.co, self.c_beta, self.h, self.c_alpha = coords
        self._model_coords = [self.nh, self.nh, self.nh, self.co, self.co, self.co, self.c_beta, self.c_beta, self.c_beta, self.h, self.h, self.h]
        self._rmsd_calpha, self._rmsd = None, None
        self._e_len_og, _e_len = None, None
        self.obj, self.coords = obj, coords

    @property
    # To get model coordinates of amino acid in new tetrahedron model
    def model_coords(self):
        return self._model_coords
    
    @model_coords.setter
    def model_coords(self, coords):
        self._model_coords = coords

    @property
    # To get the rmsd of newly created residue tetrahedron c_alpha form original c_alpha pdb coordinates of the same residue
    def rmsd_calpha(self):
        if not self._rmsd_calpha: 
            self._rmsd_calpha = np.sqrt(((np.linalg.norm(self.obj.atoms[1].coord - self.coords[1])) ** 2).mean())

        return self._rmsd_calpha

    @rmsd_calpha.setter
    def rmsd_calpha(self, val):
        self._rmsd_calpha = val

    @property
    # To get the rmsd of newly created residue tetrahedron form original pdb coordinates of the same residue
    def rmsd(self):
        if not self._rmsd:
            og_coords = [self.obj.atoms[x].coord for x in [0, 1, 2]]
            if (len(self.obj.atoms) <= 4):
                og_coords.append(self.c_beta)
            else:
                og_coords.append(self.obj.atoms[4].coord)

            og_coords.append(self.h)
            self._rmsd = np.sqrt((np.array([np.linalg.norm(p1 - p2) ** 2 for (p1, p2) in zip(og_coords, self.coords)])).mean())

        return self._rmsd

    @rmsd.setter
    def rmsd(self, val):
        self._rmsd = val

    @property
    # To get the averaged edge-length for current residue if we forms a tetraheron directly from original coordinates
    def e_len_og(self):
        if not self._e_len_og:
            og_coords = [self.obj.atoms[x].coord for x in [0, 1]]
            if (len(self.obj.atoms) <= 4):
                og_coords.append(self.c_beta)
            else:
                og_coords.append(self.obj.atoms[4].coord)

            og_coords.append(self.h)
            x = itertools.combinations(og_coords, 2)
            self._e_len_og = np.array([np.linalg.norm(p1 - p2) for (p1, p2) in x]).mean()

        return self._e_len_og

    @e_len_og.setter
    def e_len_og(self, val):
        self._e_len_og = val

    @property
    # To get the averaged edge-length for current residue if we forms a tetraheron directly from modified coordinates
    def e_len(self):
        if not self._e_len:
            x = itertools.combinations(self.coords[:1] + self.coords[2:], 2)
            self._e_len = np.array([np.linalg.norm(p1 - p2) for (p1, p2) in x]).mean()

        return self._e_len

    @e_len.setter
    def e_len(self, val):
        self._e_len = val

class Tetra:

    def __init__(self, session, models = None):

        # model_list will store all the model objects in current session, protein stores all the chain_elements
        self.model_list, self.protein = {}, {}
        # session will store the current session and edge_length will be the final edge size we use for tetrahedrons
        self.session, self.edge_length = session, None
        # all_edge_lengths will store edge lengths for all residues and chain_elements will store all the Amino objects in current chain
        self.all_edge_lengths, self.chain_elements = [], []
        # Custom created models of tetrahedron model and massing_model
        self.tetrahedron_model, self.massing_model = Model('Tetrahedrons', self.session), Model("Massing", self.session)

        # Populating the model_list. The pseudo-models are rejcted.
        if models is None:
            models = self.session.models.list()
        for model in models:
            try:
                model.chains
            except AttributeError:
                print("TODO: Handle Pseudo-Models !")
            else:
                self.model_list[model.id] = model

    # Function that will average out all the N-CO distances for each aminos after they have been moved to represent the mid-points of a peptide bonds
    # This averaged out values will be stored in edge_length and used as unit edge length for our uniform tetrahedrons in model.
    def regularize_egde_length(self, chain, res_index):

        mid_N_point = chain.residues[res_index].atoms[0].coord
        mid_CO_point = chain.residues[res_index].atoms[2].coord

        # If the amino is not the first or last in chain then move it to represent the mid points in a peptide bond
        if res_index != 0 and chain.residues[res_index - 1] is not None:
            mid_N_point = (mid_N_point + chain.residues[res_index - 1].atoms[2].coord) * 0.5

        if res_index != len(chain.residues) - 1 and chain.residues[res_index + 1] is not None:
            mid_CO_point = (mid_CO_point + chain.residues[res_index + 1].atoms[0].coord) * 0.5

        # Average out
        self.all_edge_lengths.append(np.linalg.norm(mid_N_point - mid_CO_point))
        self.edge_length = sum(self.all_edge_lengths) / len(self.all_edge_lengths)

    # Function modifies the original coordiantes to the ones we will use in our tetrahedron model. This process gives a fairly low RMSD but can be improved alot.
    def process_coordinates(self, chain, res_index, is_continuous_chain):

        N_coordinate = chain.residues[res_index].atoms[0].coord
        CO_coordinate = chain.residues[res_index].atoms[2].coord

        # If the residue is not the first or last then move the N and CO points to represent the mid point of adjacent peptide bonds.
        if res_index != 0 and chain.residues[res_index - 1]:
            if is_continuous_chain:
                N_coordinate = self.chain_elements[-1].co
            else:
                N_coordinate = (N_coordinate + chain.residues[res_index - 1].atoms[2].coord) * 0.5

        if res_index < len(chain.residues) - 1 and chain.residues[res_index + 1]:
            CO_coordinate = (CO_coordinate + chain.residues[res_index + 1].atoms[0].coord) * 0.5

        # Regualrize the CO-N edge length to the one we calculated in edge_length
        CO_coordinate = N_coordinate - (N_coordinate - CO_coordinate) * self.edge_length / np.linalg.norm(N_coordinate - CO_coordinate)
        CA_coordinate = chain.residues[res_index].atoms[1].coord
        CO_N_vector = N_coordinate - CO_coordinate

        # Find the new CB point in plane of (N, CO, CB) that form a equilateral triangle with N-CO-CB. This will be the base for residue tetrahdron.
        # If no CB coordinate the choose any random coordinate for CB. This can take CA reference for further improvements to minimize devaitions.
        if len(chain.residues[res_index].atoms) <= 4:
            vector = N_coordinate - CO_coordinate
            move_vertical_CO_CB = np.array([-1 / vector[0], 1 / vector[1], 0])
        else:
            CB_coordinate = chain.residues[res_index].atoms[4].coord
            move_along_CO_CB = (0.5 * self.edge_length - (np.dot(CO_N_vector, (CB_coordinate - CO_coordinate)) / self.edge_length)) * (CO_N_vector / self.edge_length)
            move_vertical_CO_CB = CB_coordinate + move_along_CO_CB - (CO_coordinate + N_coordinate) * 0.5

        move_vertical_CO_CB *= ht2 * self.edge_length / np.linalg.norm(move_vertical_CO_CB)
        CB_coordinate = (CO_coordinate + N_coordinate) * 0.5 + move_vertical_CO_CB

        # TODO: Check for directionality in cross product
        # Assume H as a point forming the tetrahedron with N-CO-CB base in direction to CA
        H_direction = np.cross((N_coordinate - CO_coordinate), (CB_coordinate - CO_coordinate))
        H_vector = ht3 * self.edge_length * H_direction / np.linalg.norm(H_direction)
        H_coordinate = (CO_coordinate + CB_coordinate + N_coordinate) / 3 + H_vector

        # Created an Amino object to store in chain_element that further stores in protein for references to current residue
        vertices = [N_coordinate, CO_coordinate, CB_coordinate, H_coordinate, (N_coordinate + CO_coordinate + CB_coordinate + H_coordinate) / 4]
        self.chain_elements.append(Amino(vertices, chain.residues[res_index]))

    # Function that forms an iteration method across all the residues
    # All the residues will be checked here for calculations
    def iterate_aminos(self, execute=False):

        for model in self.model_list.values():
            for chain in model.chains:
                # To store a flag that weather current amino have an amino behind it in indexing. Used in process_coordinates
                is_continuous_chain = False
                for res_index in range(len(chain.residues)):
                    residue = chain.residues[res_index]
                    # Check for amino acids
                    if not residue or residue.polymer_type != residue.PT_AMINO or not residue.find_atom('CA'):
                        is_continuous_chain = False
                        continue

                    if execute:
                        self.process_coordinates(chain, res_index, is_continuous_chain)
                        is_continuous_chain = True
                    else:
                        self.regularize_egde_length(chain, res_index)

                if execute:
                    self.protein[chain.chain_id] = self.chain_elements
                    self.chain_elements = []

    # Function to grow massing over the provided coordinates
    def grow(self, ms, unit, alpha):

        # Created a mesh to define the bounday of all given coordinates
        mesh = alphashape.alphashape(ms, alpha * 0.1)
        inside = lambda ms: trimesh.proximity.ProximityQuery(ms).signed_distance
        el = self.edge_length * unit

        # First massing tetrahedron from given coordiantes. All are indexed systematically
        pt1 = ms[0]
        pt2 = ms[0] + (ms[1] - ms[0]) * el / np.linalg.norm(ms[1] - ms[0])
        pt3 = ms[0] + (ms[2] - ms[0]) * el / np.linalg.norm(ms[2] - ms[0])
        pt4 = ms[0] + (ms[3] - ms[0]) * el / np.linalg.norm(ms[3] - ms[0])
        centroid = (pt1 + pt2 + pt3 + pt4) / 4

        # Created a store to all the calculated massing vertices to grow massing
        massing_coords = [[pt1, pt1, pt1, pt2, pt2, pt2, pt3, pt3, pt3, pt4, pt4, pt4]]
        visited = {tuple((int(centroid[0]), int(centroid[1]), int(centroid[2])))}
        queue = [[tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)]]
        faces = [[0, 3, 6], [1, 7, 9], [2, 4, 10], [5, 8, 11]]

        tcnt, depth = 1, 10**5
        while queue:

            if depth < 0:
                raise Exception("DEPTH EXCEEDED: " + len(queue))
            depth -= 1

            # Pull out the last tetrahedron we created to generate adjacent to it to grow in iteration
            prev_tetra = list(queue.pop())
            combine = permutations(prev_tetra)

            for p in combine:
                pt1, pt2, pt3, pt4 = [np.array(x) for x in p]
                p1, p2, p3, p4 = (pt1 + pt2) / 2, pt4 + (pt2 - pt1) / 2, (pt1 + pt2) / 2 + (pt4 - pt3), pt4 + (pt1 - pt2) / 2
                centroid = (p1 + p2 + p3 + p4) / 4

                t = tuple((round(centroid[0], 1), round(centroid[1], 1), round(centroid[2], 1)))
                is_visited = t in visited

                # Check if this tetrahedron is already generated. Prevents infinite looping.
                if not is_visited and (inside(mesh)((centroid,)) > -3 * unit):
                    massing_coords.append([p1, p1, p1, p2, p2, p2, p3, p3, p3, p4, p4, p4])
                    faces.extend([[12*tcnt, 12*tcnt + 3, 12*tcnt + 6], [12*tcnt + 1, 12*tcnt + 7, 12*tcnt + 9], [12*tcnt + 2, 12*tcnt + 4, 12*tcnt + 10], [12*tcnt + 5, 12*tcnt + 8, 12*tcnt + 11]])
                    tcnt += 1

                    queue.append([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
                    visited.add(t)

        x = 0
        visited = set()
        # Refine the tetrahedrons as some overlapping may have been created. Remove those overalapping tetrahedrons.
        while x < len(massing_coords):
            p1, p2, p3, p4 = massing_coords[x][0], massing_coords[x][3], massing_coords[x][6], massing_coords[x][9]
            centroid = (p1 + p2 + p3 + p4) / 4

            def checking_coords(pt, a, b):
                return pt + el * (a - b) / (4 * np.linalg.norm((a - b))), pt - el * (a - b) / (4 * np.linalg.norm((a - b)));

            pt12, pt23, pt34, pt13, pt24, pt14 = (p1 + p2) / 2, (p2 + p3) / 2, (p3 + p4) / 2, (p1 + p3) / 2, (p2 + p4) / 2, (p1 + p4) / 2
            pt12_a, pt12_b = checking_coords(pt12, p1, p2)
            pt23_a, pt23_b = checking_coords(pt23, p2, p3)
            pt34_a, pt34_b = checking_coords(pt34, p3, p4)
            pt13_a, pt13_b = checking_coords(pt13, p1, p3)
            pt24_a, pt24_b = checking_coords(pt24, p2, p4)
            pt14_a, pt14_b = checking_coords(pt14, p1, p4)

            def convert_to_tpl(a, b):
                return tuple((round(pt12_a[0], 2), round(pt12_a[1], 2), round(pt12_a[2], 2))), tuple((round(pt12_b[0], 2), round(pt12_b[1], 2), round(pt12_b[2], 2)))

            # Points we keep check to define an overlapping. If two of generated tetrahdrons 
            # share any one of these points will be considered overlapping and the recent one will be deleted.
            t1_a, t1_b = convert_to_tpl(pt12_a, pt12_b)
            t2_a, t2_b = convert_to_tpl(pt23_a, pt23_b)
            t3_a, t3_b = convert_to_tpl(pt34_a, pt34_b)
            t4_a, t4_b = convert_to_tpl(pt13_a, pt13_b)
            t5_a, t5_b = convert_to_tpl(pt24_a, pt24_b)
            t6_a, t6_b = convert_to_tpl(pt14_a, pt14_b)

            condition = any([x in visited for x in [t1_a, t2_a, t3_a, t4_a, t5_a, t6_a, t1_b, t2_b, t3_b, t4_b, t5_b, t6_b]])
            if not condition and (inside(mesh)((centroid,)) > -0.8 * unit):
                x += 1
                visited.update([t1_a, t1_b, t2_a, t2_b, t3_a, t3_b, t4_a, t4_b, t5_a, t5_b, t6_a, t6_b])
            else:
                massing_coords.pop(x)

        return np.array(massing_coords), np.array(faces[:len(massing_coords) * 4], np.int32)

    # Function to generate the tetrahedron model
    def tetrahedron(self, sequence=False, chains=False, in_sequence=True, scan=True):
        if scan:
            # First do the iterations over all the residues to calculate the 
            # required unit size tetrahedron and all the required coordinates to form the model.
            self.iterate_aminos()
            self.iterate_aminos(execute=True)

        # Generate the model via given coordinates
        def add_to_sub_model(va, ta, color, cid):
            sub_model = Model("Chain " + cid, self.session)
            va = np.reshape(va, (va.shape[0] * va.shape[1], va.shape[2]))
            ta = np.array(ta, np.int32)

            sub_model.set_geometry(va, calculate_vertex_normals(va, ta), ta)
            sub_model.vertex_colors = color
            self.tetrahedron_model.add([sub_model])

        # If given the sequence information on which residue to transform into tetrahedron model
        # There are two cases, one where input sequence need to be converted to tetra model and
        # the other that everything else other than in input sequence to be converted to tetra model.
        # This will be defined via in_sequence.
        if sequence:
            for (ch_id, chain) in self.protein.items():
                # In case of in_sequence=False, all the chains not in sequence will be converted to tetra model
                if not (ch_id in sequence.keys() or in_sequence):
                    va, ta, color, x = [], [], [], 0
                    for am in chain:
                        ta.extend([[12*x, 12*x+3, 12*x+6], [12*x+1, 12*x+7, 12*x+9], [12*x+2, 12*x+4, 12*x+10], [12*x+5, 12*x+8, 12*x+11]])
                        color.extend([am.obj.ribbon_color]*12)
                        va.append(am.model_coords)
                        x += 1

                    va = np.array(va, np.float32)
                    color = np.array(color, np.int32)
                    if 0 not in va.shape:
                        add_to_sub_model(va, ta, color, ch_id)

            for (ids, seq_lst) in sequence.items():
                va, ta, color, x = [], [], [], 0
                for am in self.protein[ids]:
                    cond = any([seq[0] <= am.obj.number and am.obj.number < seq[1] for seq in seq_lst])

                    # Check weather the current residue to be converted to tetra model or not
                    if (in_sequence and cond) or not (in_sequence or cond):
                        ta.extend([[12*x, 12*x + 3, 12*x + 6], [12*x + 1, 12*x + 7, 12*x + 9], [12*x + 2, 12*x + 4, 12*x + 10], [12*x + 5, 12*x + 8, 12*x + 11]])
                        color.extend([am.obj.ribbon_color]*12)
                        va.append(am.model_coords)
                        x += 1

                va = np.array(va, np.float32)
                color = np.array(color, np.int32)
                if 0 not in va.shape:
                    add_to_sub_model(va, ta, color, ids)

        else:
            if not chains:
                # If there is not chains in input then convert everthing in tetra model by-deault.
                chains = self.protein.keys()

            for ids in self.protein.keys():
                # Check weather current chain needed to converted to tetra model
                if ids not in chains and in_sequence or ids in chains and not in_sequence:
                    continue

                va, ta, color, x = [], [], [], 0
                for am in self.protein[ids]:
                    va.append(am.model_coords)
                    color.extend([am.obj.ribbon_color]*12)
                    ta.extend([[12*x, 12*x + 3, 12*x + 6], [12*x + 1, 12*x + 7, 12*x + 9], [12*x + 2, 12*x + 4, 12*x + 10], [12*x + 5, 12*x + 8, 12*x + 11]])
                    x += 1

                va = np.array(va, np.float32)
                color = np.array(color, np.int32)
                if 0 not in va.shape:
                    add_to_sub_model(va, ta, color, ids)

        self.session.models.add([self.tetrahedron_model])

    def massing(self, sequence=False, chains=False, unit=1, alpha=2, tetra=False):

        # First do the iterations over all the residues to calculate the 
        # required unit size tetrahedron and all the required coordinates to form the model.
        # No need to do another scan while calling tetrahedron function.
        self.iterate_aminos()
        self.iterate_aminos(execute=True)

        # Generate the model via given coordinates
        def add_to_sub_model(ms, mass_id, color):
            massing_coords, faces = self.grow(ms, unit, alpha)
            if not np.all(massing_coords.shape):
                return

            sub_model = Model("Chain " + str(mass_id), self.session)
            massing_coords = np.reshape(massing_coords, (massing_coords.shape[0] * massing_coords.shape[1], massing_coords.shape[2]))
            sub_model.set_geometry(massing_coords, calculate_vertex_normals(massing_coords, faces), faces)
            sub_model.vertex_colors = np.array([np.average(color, axis = 0)] * massing_coords.size * 12)
            self.massing_model.add([sub_model])

        mass_id = 1
        # Generate massing via given sequence
        if sequence:
            # Input given sequence in tetra model with in_sequence=False. 
            # This way everything other than massing will be in tetra model.
            if tetra:
                self.tetrahedron(sequence=sequence, in_sequence=False, scan=False)
            for (ch_id, chain) in self.protein.items():
                for (ids, seq_lst) in sequence.items():
                    if ids != ch_id:
                        continue

                    for seq in seq_lst:
                        ms, color = [], []
                        for am in chain:
                            # Check weather the current residue to be included to massing model or not
                            if seq[0] <= am.obj.number and am.obj.number < seq[1]:
                                ms.extend(am.coords)
                                color.extend([am.obj.ribbon_color]*12)

                        if ms:
                            color = np.array(color, np.int32)
                            add_to_sub_model(ms, mass_id, color)
                            mass_id += 1

        else:
            # If chains are given then convert them into massing model and everthing else into tetra model
            if chains:
                if tetra:
                    self.tetrahedron(chains=chains, in_sequence=False, scan=False)
                for ids in chains:
                    ms = [v for x in self.protein[ids] for v in x.coords]
                    color = np.array([x.obj.ribbon_color for x in self.protein[ids] for i in range(12)], np.int32)

                    add_to_sub_model(ms, mass_id, color)
                    mass_id += 1

            # If not given chains the convert everything into tetra model by-default.
            else:
                self.massing(chains=self.protein.keys())

        self.session.models.add([self.massing_model])


## Need to create a commmand input system. Later will be done using UI intergrations.
# def tetrahedral_model(session, chains=False):
#     #from Tetra import Tetra
#     t = Tetra(session)
#     if chains:
#         chains = list(enumerate(chains))
#     t.tetrahedron(chains=chains)

def massing_model(session, residues=None, unit=1, alpha=2, tetra=False):
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    for structure, chain_residue_intervals in residue_intervals(residues):
        t = Tetra(session, models = [structure])
        t.massing(sequence = chain_residue_intervals, unit = unit, alpha = alpha, tetra = tetra)

def residue_intervals(residues):
    return [(structure, {chain_id:number_intervals(cres.numbers) for s, chain_id, cres in sres.by_chain})
            for structure, sres in residues.by_structure]

def number_intervals(numbers):            
    intervals = []
    start = end = None
    for num in numbers:
        if start is None:
            start = end = num
        elif num == end+1:
            end = num
        else:
            intervals.append((start,end))
            start = end = num
    intervals.append((start,end))
    return intervals
    
def register_command(session):
    from chimerax.core.commands import CmdDesc, register, FloatArg
    from chimerax.atomic import ResiduesArg

#     t_desc = CmdDesc(required = [],
#                      optional=[("chains", UniqueChainsArg)],
#                      synopsis = 'creates tetrahedral model')
#     register('tetra', t_desc, tetrahedral_model, logger=session.logger)

    m_desc = CmdDesc(required = [],
                     optional=[("residues", ResiduesArg)],
                     keyword=[("unit", FloatArg), ("alpha", FloatArg), ("tetra", BoolArg)],
                     synopsis = 'create tetrahedral massing model')
    register('massing', m_desc, massing_model, logger=session.logger)

register_command(session)
