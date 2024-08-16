from . import TetraScape as TS
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
			     synopsis = 'Creates TetraChain model')
	    register('tetraChain', t_desc, TS.tetrahedral_model, logger=logger)

	    m_desc = CmdDesc(required = [],
			     optional = [("residues", ResiduesArg)],
			     keyword = [("unit", FloatArg), ("alpha", FloatArg), ("checkOverlap", BoolArg), ("tetraChain", BoolArg), ("tetraChainResidues", ResiduesArg)],
			     synopsis = 'Creates TetraMass model')
	    register('tetraMass', m_desc, TS.massing_model, logger=logger)

bundle_api = _TetraAPI()
