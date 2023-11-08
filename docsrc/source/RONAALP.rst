RONAALP Algorithm
-----------------

Description of the RONAALP module to create a reduced-order model of a high dimensional nonlinear functions with active learning procedure.

Mathematically, a RONAALP reduced-order model :math:`g` predicts the outputs of the function of interest :math:`f` such that :math:`\lVert g({\mathbf{z}}) - f(\mathbf{z}) \rVert_2` is minimized and the computational cost is significantly reduced compared to the original function. The strategy employed to derive the reduced-order model is presented in [1], [2].

.. toctree::
    train
    evaluate
    update
    
.. [1] Scherding, C., Rigas, G., Sipp, D., Schmid, P. J., & Sayadi, T. (2023). Data-driven framework for input/output lookup tables reduction: Application to hypersonic flows in chemical nonequilibrium. Physical Review Fluids, 8(2), 023201.
.. [2] Scherding, C. (2023). Predictive modeling of hypersonic flows in thermochemical nonequilibrium: physics-based and data-driven approaches. PhD Thesis, Sorbone University.

.. autofunction:: RONAALP.Ronaalp
