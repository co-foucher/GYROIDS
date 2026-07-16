"""
============================================================================
TPMS_CLASSES
Subpackage holding TPMSModel (the shared base class) and one module per
concrete TPMS surface (Gyroid, Schwartz P, Diamond, IWP, Neovius,
Fischer-Koch S, F-RD, Lidinoid, Split-P). Each surface module is a thin
subclass that only supplies its own implicit equation - see tpms_base.py
for the shared validation / field / mesh pipeline.
============================================================================
"""

from . import tpms_base
from . import tpms_gyroid
from . import tpms_schwartzp
from . import tpms_diamond
from . import tpms_iwp
from . import tpms_neovius
from . import tpms_fischerkochs
from . import tpms_frd
from . import tpms_lidinoid
from . import tpms_splitp
from . import tpms_custom

# Re-export the model classes and factory functions so callers can do
# `from gyroid_utils.TPMS_classes import GyroidModel` instead of reaching
# into the individual surface modules.
from .tpms_base import TPMSModel, create_a_tpms
from .tpms_gyroid import GyroidModel, create_a_gyroid
from .tpms_schwartzp import SchwartzPModel, create_a_schwartz_p
from .tpms_diamond import DiamondModel, create_a_diamond
from .tpms_iwp import IWPModel, create_a_iwp
from .tpms_neovius import NeoviusModel, create_a_neovius
from .tpms_fischerkochs import FischerKochSModel, create_a_fischer_koch_s
from .tpms_frd import FRDModel, create_a_frd
from .tpms_lidinoid import LidinoidModel, create_a_lidinoid
from .tpms_splitp import SplitPModel, create_a_split_p
from .tpms_custom import CustomTPMSModel, create_a_custom_tpms

#it defines what from gyroid_utils.TPMS_classes import * actually imports.
__all__ = [
    "tpms_base",
    "tpms_gyroid",
    "tpms_schwartzp",
    "tpms_diamond",
    "tpms_iwp",
    "tpms_neovius",
    "tpms_fischerkochs",
    "tpms_frd",
    "tpms_lidinoid",
    "tpms_splitp",
    "tpms_custom",
    "TPMSModel",
    "GyroidModel",
    "SchwartzPModel",
    "DiamondModel",
    "IWPModel",
    "NeoviusModel",
    "FischerKochSModel",
    "FRDModel",
    "LidinoidModel",
    "SplitPModel",
    "CustomTPMSModel",
    "create_a_tpms",
    "create_a_gyroid",
    "create_a_schwartz_p",
    "create_a_diamond",
    "create_a_iwp",
    "create_a_neovius",
    "create_a_fischer_koch_s",
    "create_a_frd",
    "create_a_lidinoid",
    "create_a_split_p",
    "create_a_custom_tpms",
]
