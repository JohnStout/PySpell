# pyspell - Spellman Lab Python Toolkit

"""
PySpell: Python tools for calcium imaging analysis.

This package provides tools for:
- Suite2p post-processing and deconvolution
- ThorLabs imaging file management
- NWB file format conversion
- Cell registration preparation

Example usage:
    from pyspell import postProcess, fast_suite2p
    
    # Run suite2p on your data
    fast_suite2p(imgpath="path/to/img.tif")
    
    # Post-process suite2p results
    pp = postProcess(s2ppath="path/to/suite2p/plane0")
    C, S, metrics = pp.cleanup_raw_traces_adaptive()
"""

__version__ = "0.1.0"
__author__ = "Spellman Lab"

# Core suite2p functions
from .s2pfuns import (
    postProcess,
    read_s2p,
    fast_suite2p,
    parse_fpath,
    baseline_corrected_F,
)

# ThorLabs imaging utilities
from .thorfuns import (
    RawToTif,
    importThorsync,
)

# Root/path utilities  
from .rootfun import (
    dropbox_root,
    list_all_subdirs,
)

# Session registration
from .sessreg import (
    suite2pToCellReg,
)
