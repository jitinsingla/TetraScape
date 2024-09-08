from . import TetraScape as TS
from chimerax.core.toolshed import BundleAPI
from chimerax.atomic import ResiduesArg
from chimerax.core.commands import BoolArg, IntArg, CmdDesc, register, FloatArg, StringArg

class _TetraAPI(BundleAPI):
	api_version = 1

	@staticmethod
	def register_command(bi, ci, logger):
	    tetra_desc = CmdDesc(
	        required=[("subcommand", StringArg)],
	        optional=[("residues", ResiduesArg)],
	        keyword=[
	            ("reverse", BoolArg),
	            ("unit", FloatArg),
	            ("alpha", FloatArg),
	            ("overlap", BoolArg),
	            ("chain", BoolArg),
	            ("chainResidues", ResiduesArg)
	        ],
	        synopsis="Creates TetraChain or TetraMass model"
	    )
	    register('tetra', tetra_desc, TS.tetra_command, logger=logger)

bundle_api = _TetraAPI()
