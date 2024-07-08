# Locality in Long Range Arena

## Running experiments
A configuration file needs to be created for each model and task (see the `configs` directory for examples). Then the experiment can be run with the following command:

```
python scripts/train.py --config configs/<config-file>
```

## Experiments

### RQ1

```bash
micromamba run -n long-transformers python scripts/train.py --config configs/cifar10/rotary_transformer_augment.json
micromamba run -n long-transformers python scripts/train.py --config configs/cifar10/rotary_transformer_no_augment.json
micromamba run -n long-transformers python scripts/train.py --config configs/cifar10/mega_augment.json
micromamba run -n long-transformers python scripts/train.py --config configs/cifar10/mega_no_augment.json

micromamba run -n long-transformers python scripts/train.py --config configs/pathfinder/rotary_transformer_noaugment_nopretrain.json
micromamba run -n long-transformers python scripts/train.py --config configs/pathfinder/rotary_transformer_augment_nopretrain.json
micromamba run -n long-transformers python scripts/train.py --config configs/pathfinder/mega_noaugment_nopretrain.json
micromamba run -n long-transformers python scripts/train.py --config configs/pathfinder/mega_augment_nopretrain.json
```

### RQ2

```bash
micromamba run -n long-transformers python scripts/train.py --config configs/text_retrieval/rotary_transformer_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_retrieval/rotary_transformer_wout_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_retrieval/mega_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_retrieval/mega_wout_mlm.json

micromamba run -n long-transformers python scripts/train.py --config configs/text_classification/rotary_transformer_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_classification/rotary_transformer_wout_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_classification/mega_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_classification/mega_wout_mlm.json
```

### RQ3

```bash
micromamba run -n long-transformers python scripts/train.py --config configs/listops/rotary_transformer_augment_nopretrain.json
micromamba run -n long-transformers python scripts/train.py --config configs/listops/rotary_transformer_noaugment_nopretrain.json

micromamba run -n long-transformers python scripts/train.py \
    --config configs/listops/rotary_transformer_noaugment_pretrain_2.json \
    --pretrain-config configs/listops/rotary_transformer_augment_pretrain_1.json
micromamba run -n long-transformers python scripts/train.py \
    --config configs/listops/rotary_transformer_noaugment_pretrain_2.json \
    --pretrain-config configs/listops/rotary_transformer_noaugment_pretrain_1.json

micromamba run -n long-transformers python scripts/train.py --config configs/listops/mega_augment_nopretrain.json
micromamba run -n long-transformers python scripts/train.py --config configs/listops/mega_noaugment_nopretrain.json

micromamba run -n long-transformers python scripts/train.py \
    --config configs/listops/mega_noaugment_pretrain_2.json \
    --pretrain-config configs/listops/mega_augment_pretrain_1.json
micromamba run -n long-transformers python scripts/train.py \
    --config configs/listops/mega_noaugment_pretrain_2.json \
    --pretrain-config configs/listops/mega_noaugment_pretrain_1.json
```

### RQ4

```bash
micromamba run -n long-transformers python scripts/train.py --config configs/text_classification/linear_transformer_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/listops/linear_transformer_augment_nopretrain.json
micromamba run -n long-transformers python scripts/train.py --config configs/cifar10/linear_transformer_augment.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_retrieval/linear_transformer_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/pathfinder/linear_transformer_noaugment.json

micromamba run -n long-transformers python scripts/train.py --config configs/cifar10/transformer_augment.json
micromamba run -n long-transformers python scripts/train.py --config configs/listops/transformer_augment_nopretrain.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_classification/transformer_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_retrieval/transformer_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/pathfinder/transformer_noaugment.json
```

### RQ5

```bash
micromamba run -n long-transformers python scripts/train.py --config configs/listops/gmlp_w_augment.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_classification/gmlp_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/text_retrieval/gmlp_w_mlm.json
micromamba run -n long-transformers python scripts/train.py --config configs/cifar10/gmlp_w_augment.json
micromamba run -n long-transformers python scripts/train.py --config configs/pathfinder/gmlp.json
```