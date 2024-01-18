from . import cmd
from chimerax.atomic import ResiduesArg
from chimerax.core.toolshed import BundleAPI
from chimerax.core.commands import BoolArg, IntArg, CmdDesc, register, FloatArg

class _TetraAPI(BundleAPI):
	api_version = 1

	@staticmethod
	def register_command(bi, ci, logger):
	    t_desc = CmdDesc(required = [], 
	    				optional = [("residues", ResiduesArg), ("revSeq", BoolArg)], 
	    				keyword = [],
	    				synopsis = 'creates tetrahedral model')
	    register('tetra', t_desc, cmd.tetrahedral_model, logger=logger)

	    m_desc = CmdDesc(required = [], 
	    				optional = [("residues", ResiduesArg)], 
	    				keyword = [("unit", FloatArg), ("alpha", FloatArg), ("tetra", BoolArg)], 
	    				synopsis = 'create tetrahedral massing model')
	    register('massing', m_desc, cmd.massing_model, logger=logger)

bundle_api = _TetraAPI()