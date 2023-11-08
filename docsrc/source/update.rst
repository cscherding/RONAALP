Update
================

The update procedure is done in 2 steps:

1. Tessalation of the new points using the ``update`` method of the K-Means class.

2. Retrain of the RBFs by inverting the augmented linear system using the Schur complement.
    
.. autofunction:: RONAALP.Ronaalp.update
    
