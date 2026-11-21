# ApexID Hierarchical Loss

`ApexIDHierarchicalHead` implements equations 4–11 of the paper. It adds global Sub-center SphereFace2-A identity discrimination to category-weighted ArcFace cross-entropy. The category proxy weights use inverse square roots of the number of training identities per category. The default category loss coefficient is 0.50.

`sphereface2.py` implements the standalone SphereFace2 baseline and provides the shared identity-margin computation and bias initialization. The training heads are discarded for evaluation.
