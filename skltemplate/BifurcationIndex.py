import numpy as np


def bifurcation_index(PLV_condA, PLV_condB, PLV_condAB):
    """ Compute the bifurcation index between conditions A and B (from Busch et al., 2009, J Neuro)

    Parameters
    ----------
    PLV_condA: array sensors*frequencies*time
    Phase-locking value computed across trials for condition A

    PLV_condB: array sensors*frequencies*time
    Phase-locking value computed across trials for condition B

    PLV_condAB: array sensors*frequencies*time
    Phase-locking value computed across trials gathered across conditions A and B

    Returns
    ----------
    BI: array sensors*frequencies*time
    Bifurcation index

    """
    BI = (PLV_condA-PLV_condAB)*(PLV_condB-PLV_condAB)
    return BI
    
    
