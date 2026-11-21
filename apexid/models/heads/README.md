# Training heads

The paper compares ArcFace, Dynamic ArcFace, SphereFace2-A, and ApexID Hierarchical Loss. ArcFace and Dynamic ArcFace return margin logits for cross-entropy. SphereFace2 and the [hierarchical head](../../losses/hierarchical.py) compute their own objectives. Evaluation uses only the image encoder.
