from chimerax.core.toolshed import BundleAPI
from . import cmd

class _TetraAPI(BundleAPI):
	api_version = 1

	@staticmethod
	def register_command(bi, ci, logger):

		if ci.name == "massing":
			func = cmd.massing_model
			desc = cmd.m_desc
		else:
			raise ValueError("trying to register unknown command: %s" % ci.name)

		if desc.synopsis is None:
			desc.synopsis = ci.synopsis

		from chimerax.core.commands import register
		register(ci.name, desc, func)

bundle_api = _TetraAPI()