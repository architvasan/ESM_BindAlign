# ESM_BindAlign

## Usage
Generate embeddings:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m protein_search.distributed_inference --config smiles_embedding.yaml
CUDA_VISIBLE_DEVICES=0,1 python -m protein_search.distributed_inference --config protein_embedding.yaml
```
