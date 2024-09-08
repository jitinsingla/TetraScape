import numpy as np, pyvista as pv
import alphashape, trimesh, gc, logging, warnings
warnings.filterwarnings("ignore")
from itertools import permutations, combinations
from chimerax.core.models import Model
from chimerax.atomic import ResiduesArg
from chimerax.surface import calculate_vertex_normals
from chimerax.atomic import all_residues

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
            self._rmsd_calpha = np.linalg.norm(self.obj.atoms[1].coord - self.c_alpha)

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
        self.tetrahedron_model, self.massing_model = Model('TetraChain-'+str(models[0].name), self.session), Model("TetraMass", self.session)
        # contains individual meshes of each tetrahedron of massing
        self.massingMeshes = {}

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

        # Append to all edge lengths
        self.all_edge_lengths.append(np.linalg.norm(mid_N_point - mid_CO_point))

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

        # self.edge_length = sum(self.all_edge_lengths) / len(self.all_edge_lengths)

    # Function to grow massing over the provided coordinates
    def grow(self, ms, unit, alpha):
        # Created a mesh to define the bounday of all given coordinates
        mesh = alphashape.alphashape(ms, alpha)
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
        t = tuple((round(centroid[0], 1), round(centroid[1], 1), round(centroid[2], 1)))
        visited = {t}
        queue = [[tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)]]
        faces = [[0, 3, 6], [1, 7, 9], [2, 4, 10], [5, 8, 11]]
        tcnt = 1

        # create directional vectors
        p = queue[0]
        p1, p2, p3, p4 = [np.array(x) for x in p]
        aboveVector = (p1+p2)/2 - (p3+p4)/2
        belowVector = (p3+p4)/2 - (p1+p2)/2

        def add_tetrahedron_above(p_new):
            p1a, p2a, p3a, p4a = p_new[0] + aboveVector, p_new[1] + aboveVector, p_new[2] + aboveVector, p_new[3] + aboveVector
            add_tetrahedron([p1a, p2a, p3a, p4a])

        def add_tetrahedron_below(p_new):
            p1b, p2b, p3b, p4b = p_new[0] + belowVector, p_new[1] + belowVector, p_new[2] + belowVector, p_new[3] + belowVector
            add_tetrahedron([p1b, p2b, p3b, p4b])

        def add_tetrahedron(p_new):
            nonlocal massing_coords, faces, tcnt, queue, visited
            centroid = np.mean(p_new, axis=0)
            t = tuple((round(centroid[0], 1), round(centroid[1], 1), round(centroid[2], 1)))
            is_visited = t in visited
            # insideFlag = False
            # insideFlag = insideFlag and (inside(mesh)((p_new[0],)) > -3 * unit)
            # insideFlag = insideFlag and (inside(mesh)((p_new[1],)) > -3 * unit)
            # insideFlag = insideFlag and (inside(mesh)((p_new[2],)) > -3 * unit)
            # insideFlag = insideFlag and (inside(mesh)((p_new[3],)) > -3 * unit)
            if not is_visited and (inside(mesh)((centroid,)) > -0.05 * unit):
            # if not is_visited and insideFlag:
                massing_coords.append([p_new[0], p_new[0], p_new[0], p_new[1], p_new[1], p_new[1], p_new[2], p_new[2], p_new[2], p_new[3], p_new[3], p_new[3]])
                faces.extend([[12*tcnt, 12*tcnt + 6, 12*tcnt + 3], 
                          [12*tcnt + 1, 12*tcnt + 9, 12*tcnt + 7], 
                          [12*tcnt + 2, 12*tcnt + 4, 12*tcnt + 10], 
                          [12*tcnt + 5, 12*tcnt + 8, 12*tcnt + 11]])
                tcnt += 1
                queue.append([tuple(x) for x in p_new])
                visited.add(t)
                add_tetrahedron_above(p_new)
                add_tetrahedron_below(p_new)
        
        while queue:
            p = list(queue.pop(0))
            p1, p2, p3, p4 = [np.array(x) for x in p]
            centroid = (p1 + p2 + p3 + p4) / 4
            t = tuple((round(centroid[0], 1), round(centroid[1], 1), round(centroid[2], 1)))
            # print("Poped:", t)
            
            rightVector = p2-p1
            leftVector = p1-p2
            upVector = p3-p4
            downVector = p4-p3
            rightHalfUpHalfVector = (rightVector+upVector)/2
            leftHalfUpHalfVector = (leftVector+upVector)/2
            
            # 1 already in the list
            # 1 Above
            add_tetrahedron_above([p1, p2, p3, p4])
            add_tetrahedron_below([p1, p2, p3, p4])
            
            # 2 (Left)    
            p5, p6, p7, p8 = p2 + upVector/2, p2 + downVector/2, (p3+p4)/2 + rightVector, (p3+p4)/2
            add_tetrahedron([p5, p6, p7, p8])
            
            # 3 (Right)
            p5, p6, p7, p8 = p5 + leftVector, p6 + leftVector, p7 + leftVector, p8 + leftVector
            add_tetrahedron([p5, p6, p7, p8])
            
            # 4 (Up)
            p5, p6, p7, p8 = p5 + rightHalfUpHalfVector, p6 + rightHalfUpHalfVector, p7 + rightHalfUpHalfVector, p8 + rightHalfUpHalfVector
            add_tetrahedron([p5, p6, p7, p8])
            
            # 5 (Down)
            p5, p6, p7, p8 = p5 + downVector, p6 + downVector, p7 + downVector, p8 + downVector
            add_tetrahedron([p5, p6, p7, p8])
            
            # 6 (Right up)
            p5, p6, p7, p8 = p1 + rightHalfUpHalfVector, p2 + rightHalfUpHalfVector, p3 + rightHalfUpHalfVector, p4 + rightHalfUpHalfVector
            add_tetrahedron([p5, p6, p7, p8])
            
            # 7 (Right down)
            p5, p6, p7, p8 = p5 + downVector, p6 + downVector, p7 + downVector, p8 + downVector
            add_tetrahedron([p5, p6, p7, p8])

            # 8 (Left up)
            p5, p6, p7, p8 = p1 + leftHalfUpHalfVector, p2 + leftHalfUpHalfVector, p3 + leftHalfUpHalfVector, p4 + leftHalfUpHalfVector
            add_tetrahedron([p5, p6, p7, p8])
            
            # 9 (Left down)
            p5, p6, p7, p8 = p5 + downVector, p6 + downVector, p7 + downVector, p8 + downVector
            add_tetrahedron([p5, p6, p7, p8])

        del visited, queue, mesh
        gc.collect()
        return np.array(massing_coords), np.array(faces[:len(massing_coords) * 4], np.int32)

    # Function to generate the tetrahedron model
    def tetrahedron(self, sequence, in_sequence, scan=True):
        if scan:
            # First do the iterations over all the residues to calculate the 
            # required unit size tetrahedron and all the required coordinates to form the model.
            
            # Uncomment the bottom command if you want to compute the average edge-length instead of fixing 3 amstrong
            # self.iterate_aminos()
            self.edge_length = 3.0
            self.iterate_aminos(execute=True)

        # Generate the model via given coordinates
        def add_to_sub_model(va, ta, color, cid):
            sub_model = Model("Chain " + cid, self.session)
            va = np.reshape(va, (va.shape[0] * va.shape[1], va.shape[2]))
            ta = np.array(ta, np.int32)

            sub_model.set_geometry(va, calculate_vertex_normals(va, ta), ta)

            sub_model.vertex_colors = color
            self.tetrahedron_model.add([sub_model])
        # print("In sequence:", in_sequence)
        # print("Sequence:", sequence)
        for (ch_id, chain) in self.protein.items():
            # In case of in_sequence=False, all the chains not in sequence will be converted to tetra model
            if not (ch_id in sequence.keys() or in_sequence):
                va, ta, color, x = [], [], [], 0
                for am in chain:
                    ta.extend([[12*x, 12*x+3, 12*x+6], [12*x+1, 12*x+7, 12*x+9], [12*x+2, 12*x+4, 12*x+10], [12*x+5, 12*x+8, 12*x+11]])
                    color.extend([am.obj.ribbon_color for i in range(12)])
                    va.append(am.model_coords)
                    x += 1

                va = np.array(va, np.float32)
                color = np.array(color, np.float32)
                if 0 not in va.shape:
                    add_to_sub_model(va, ta, color, ch_id)

        for (ids, seq_lst) in sequence.items():
            va, ta, color, x = [], [], [], 0
            if ids not in self.protein.keys():
                continue
            for am in self.protein[ids]:
                cond = any([seq[0] <= am.obj.number and am.obj.number <= seq[1] for seq in seq_lst])
                # Check weather the current residue to be converted to tetra model or not
                if (in_sequence and cond) or not (in_sequence or cond): #xnor operation
                    ta.extend([[12*x, 12*x + 3, 12*x + 6], [12*x + 1, 12*x + 7, 12*x + 9], [12*x + 2, 12*x + 4, 12*x + 10], [12*x + 5, 12*x + 8, 12*x + 11]])
                    color.extend([am.obj.ribbon_color for i in range(12)])
                    va.append(am.model_coords)
                    x += 1

            va = np.array(va, np.float32)
            color = np.array(color, np.float32)
            if 0 not in va.shape:
                add_to_sub_model(va, ta, color, ids)

        self.session.models.add([self.tetrahedron_model])

    def massing(self, sequence, unit, alpha, checkOverlap, tetraChain, tetraChainResidues, tetraChainSequence):

        # First do the iterations over all the residues to calculate the 
        # required unit size tetrahedron and all the required coordinates to form the model.
        # No need to do another scan while calling tetrahedron function.
        # self.iterate_aminos()
        self.edge_length = 3.0
        self.iterate_aminos(execute=True)

        # Generate the model via given coordinates
        def add_to_sub_model(ms, chain_id, mass_id, color, checkOverlap):
            massing_coords, faces = self.grow(ms, unit, alpha)
            if not np.all(massing_coords.shape):
                return
            massingName = str(chain_id) + "-" + str(mass_id)
            sub_model = Model(massingName, self.session)
            massing_coords = np.reshape(massing_coords, (massing_coords.shape[0] * massing_coords.shape[1], massing_coords.shape[2]))
            numMassTetra = int(massing_coords.shape[0]/12)

            # individual meshes
            self.massingMeshes[massingName] = []
            for tetraNumber in range(numMassTetra):
                mesh = trimesh.Trimesh(vertices=massing_coords[tetraNumber*12:(tetraNumber+1)*12, :], faces=faces[:4, :])
                self.massingMeshes[massingName].append(pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces])))

            # print("checkOverlap:", checkOverlap)
            # print("Before:", massing_coords.shape, faces.shape)
            if checkOverlap and mass_id>1:
                # Check overlap with existing massings first and then add to the model list
                # Save the current logging level
                old_level = logging.getLogger().getEffectiveLevel()
                # Set the logging level to CRITICAL to suppress ERROR messages temporarily
                logging.getLogger().setLevel(logging.CRITICAL)
                massing_coords_new = []
                faces_new = []
                countNotIntersect = 0

                for i, meshCurr in enumerate(self.massingMeshes[massingName]):
                    intersectFlag = False
                    for massName in self.massingMeshes.keys():
                        if massingName==massName:
                            continue        
                        for j, meshOld in enumerate(self.massingMeshes[massName]):
                            intersection = meshOld.boolean_intersection(meshCurr)
                            if intersection.n_faces > 0:
                                intersectFlag = True
                                # print(i+1, "Intersect")
                                break
                    if intersectFlag is False:
                        #print(i+1, "Adding to new list")
                        massing_coords_new.append(massing_coords[i*12:(i+1)*12, :])
                        faces_new.extend(faces[:4, :] + 12*countNotIntersect)
                        countNotIntersect += 1

                massing_coords = np.array(massing_coords_new)
                faces = np.array(faces_new)
                if massing_coords.size > 0:
                    massing_coords = np.reshape(massing_coords, (massing_coords.shape[0] * massing_coords.shape[1], massing_coords.shape[2]))
                    numMassTetra = int(massing_coords.shape[0]/12)
                    sub_model = Model(massingName, self.session)
                    sub_model.set_geometry(massing_coords, calculate_vertex_normals(massing_coords, faces), faces)
                    sub_model.vertex_colors = np.array([np.average(color, axis = 0)] * massing_coords.size * 12)
                    self.massing_model.add([sub_model])

                    # Re-calculate the individual meshes
                    self.massingMeshes[massingName] = []
                    for tetraNumber in range(numMassTetra):
                        mesh = trimesh.Trimesh(vertices=massing_coords[tetraNumber*12:(tetraNumber+1)*12, :], faces=faces[:4, :])
                        self.massingMeshes[massingName].append(pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces])))
            
                # print("After:", massing_coords.shape, faces.shape)


            else:
                sub_model.set_geometry(massing_coords, calculate_vertex_normals(massing_coords, faces), faces)
                sub_model.vertex_colors = np.array([np.average(color, axis = 0)] * massing_coords.size * 12)
                self.massing_model.add([sub_model])

        # Input given sequence in tetra model with in_sequence=False. 
        # This way everything other than massing will be in tetra model.
        if tetraChain:
            if tetraChainResidues is None:
                self.tetrahedron(sequence=sequence, in_sequence=False, scan=False)
            else:
                self.tetrahedron(sequence=tetraChainSequence, in_sequence=True, scan=False)

        for (ch_id, chain) in self.protein.items():
            mass_id = 1
            for (ids, seq_lst) in sequence.items():
                if ids != ch_id:
                    continue

                for seq in seq_lst:
                    # print("--------------------------------------------------------")
                    # print("Mass ID:", mass_id, "Chain ID:", ch_id, "sequence:", seq)
                    ms, color = [], []
                    for am in chain:
                        # Check weather the current residue to be included to massing model or not
                        if seq[0] <= am.obj.number and am.obj.number <= seq[1]:
                            ms.extend(am.coords)
                            color.extend([am.obj.ribbon_color]*12)

                    if ms:
                        color = np.array(color, np.float32)
                        add_to_sub_model(ms, ch_id, mass_id, color, checkOverlap)
                        mass_id += 1

        self.session.models.add([self.massing_model])

    def rmsd_calpha(self):
        model_c_alpha_coords = []
        original_c_alpha_coords = []
        for chain_id in self.protein:
            for aminoAcid in self.protein[chain_id]:
                model_c_alpha_coords.append(aminoAcid.c_alpha)
                original_c_alpha_coords.append(aminoAcid.obj.atoms[1].coord)
        model_c_alpha_coords = np.array(model_c_alpha_coords)
        original_c_alpha_coords = np.array(original_c_alpha_coords)
        return np.sqrt(np.mean(np.square(model_c_alpha_coords - original_c_alpha_coords)))

    def rmsd_N_Calpha_CO(self):
        model_coords = []
        original_coords = []
        for chain_id in self.protein:
            for aminoAcid in self.protein[chain_id]:
                model_coords.extend([aminoAcid.nh, aminoAcid.c_alpha, aminoAcid.co])
                original_coords.extend([aminoAcid.obj.atoms[x].coord for x in [0, 1, 2]])
        model_coords = np.array(model_coords)
        original_coords = np.array(original_coords)
        return np.sqrt(np.mean(np.square(model_coords - original_coords)))

def massing_model(session, residues=None, unit=2, alpha=0.2, checkOverlap=True, tetraChain=True, tetraChainResidues=None):
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    if tetraChainResidues is not None:
        structure_Dict = {}
        residue_intervals_tetraChain = residue_intervals(tetraChainResidues)
        count = 0
        for structure, _ in residue_intervals_tetraChain:
            structure_Dict[structure] = count
            count += 1
    for structure, chain_residue_intervals in residue_intervals(residues):
        t = Tetra(session, models = [structure])
        print("Chain Residue Intervals:")
        print(chain_residue_intervals)
        if tetraChainResidues is not None:
            tetraChainSequence = residue_intervals_tetraChain[structure_Dict[structure]][1]
            print("Before overlap removal:")
            print(tetraChainSequence)
            tetraChainSequence = remove_overlap(tetraChainSequence, chain_residue_intervals)
            print("After overlap removal:")
            print(tetraChainSequence)
        else:
            tetraChainSequence = chain_residue_intervals
        # print("Tetra Chain Sequence:")
        # print(tetraChainSequence)
        t.massing(sequence = chain_residue_intervals, unit = unit, alpha = alpha, checkOverlap = checkOverlap, tetraChain = tetraChain, tetraChainResidues=tetraChainResidues, tetraChainSequence=tetraChainSequence)
        print("----------------------------------")
        print("RMSD of TetraChain part:")
        print("C_alpha only:", t.rmsd_calpha())
        print("N, C_alpha and CO:", t.rmsd_N_Calpha_CO())
        print("----------------------------------")

def tetrahedral_model(session, residues=None, revSeq=False):
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    for structure, chain_residue_intervals in residue_intervals(residues):
        t = Tetra(session, models = [structure])
        t.tetrahedron(sequence = chain_residue_intervals, in_sequence = not revSeq)
        print("----------------------------------")
        print("RMSD of TetraChain part:")
        print("C_alpha only:", t.rmsd_calpha())
        print("N, C_alpha and CO:", t.rmsd_N_Calpha_CO())
        print("----------------------------------")

def residue_intervals(residues):
    return [(structure, {chain_id:number_intervals(cres) for s, chain_id, cres in sres.by_chain})
            for structure, sres in residues.by_structure]

def number_intervals(cres):      
    intervals = []
    start = end = None
    # print(cres.numbers)
    for residue, num in zip(cres, cres.numbers):
        if not residue or residue.polymer_type != residue.PT_AMINO or not residue.find_atom('CA'):
            continue
        if start is None:
            start = end = num 
        elif num == end+1:
            end = num
        else:
            intervals.append((start,end))
            start = end = num
    intervals.append((start,end))
    return intervals

def remove_overlap(tetraChainSequence, chain_residue_intervals):
    def remove_overlap_from_interval(interval, forbidden_intervals):
        start, end = interval
        adjusted_intervals = []
        
        for forbidden_start, forbidden_end in forbidden_intervals:
            if end < forbidden_start or start > forbidden_end:
                # No overlap
                continue
            if start < forbidden_start and end > forbidden_end:
                # Split into two intervals
                adjusted_intervals.append((start, forbidden_start - 1))
                start = forbidden_end + 1
            elif start < forbidden_start:
                end = forbidden_start - 1
            elif end > forbidden_end:
                start = forbidden_end + 1
            else:
                # Fully overlapped
                start, end = -1, -1

        if start <= end and start != -1:
            adjusted_intervals.append((start, end))

        return adjusted_intervals

    result = {}
    for chain, intervals in tetraChainSequence.items():
        if chain in chain_residue_intervals:
            forbidden_intervals = chain_residue_intervals[chain]
            new_intervals = []
            for interval in intervals:
                adjusted_intervals = remove_overlap_from_interval(interval, forbidden_intervals)
                new_intervals.extend(adjusted_intervals)
            result[chain] = new_intervals
        else:
            result[chain] = intervals

    return result

def tetra_command(session, subcommand, residues=None, **kwargs):
    if subcommand == 'chain':
        tetrahedral_model(
            session,
            residues=residues,
            revSeq=kwargs.get('reverse', False)
        )
    elif subcommand == 'mass':
        massing_model(
            session, 
            residues=residues,
            unit=kwargs.get('unit', 2.0),
            alpha=kwargs.get('alpha', 0.2),
            checkOverlap=kwargs.get('overlap', True),
            tetraChain=kwargs.get('chain', True),
            tetraChainResidues=kwargs.get('chainResidues', None)
        )
    else:
        session.logger.warning(f"Unknown subcommand: {subcommand}")
