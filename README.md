# Locality in Long Range Arena

## Running experiments
A configuration file needs to be created for each model and task (see the `configs` directory for examples). Then the experiment can be run with the following command:

```
python scripts/train.py --config configs/<config-file>
```

## Config possible values
- Tasks:
    - `"text-classification"`
- Models:
    - `"conv-transformer"`
    - `"transformer-base"`


## Experiments

### Pathfinder

```bash
micromamba run -n long-transformers-exp python scripts/train.py  \
    --config configs/pathfinder/rotary_transformer_augment_pretrain_2.json \
    --pretrain-config configs/pathfinder/rotary_transformer_augment_pretrain_1.json

micromamba run -n long-transformers-exp python scripts/train.py  \
    --config configs/pathfinder/rotary_transformer_noaugment_pretrain_2.json \
    --pretrain-config configs/pathfinder/rotary_transformer_noaugment_pretrain_1.json

micromamba run -n long-transformers-exp python scripts/train.py  \
    --config configs/pathfinder/rotary_transformer_augment_nopretrain.json

micromamba run -n long-transformers-exp python scripts/train.py  \
    --config configs/pathfinder/rotary_transformer_noaugment_nopretrain.json 
```