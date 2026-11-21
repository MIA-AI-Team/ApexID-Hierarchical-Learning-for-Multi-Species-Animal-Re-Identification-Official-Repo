# Training engine

`Trainer` optimizes embeddings and the selected training head. `Validator` runs internal training validation for checkpoint selection. `EarlyStopping` records the best encoder and full model, and `checkpoint.py` stores optimizer, scheduler, random state, and loader generator state for resume.

External paper benchmarks use separate frozen evaluation manifests and saved encoders.
