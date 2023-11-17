# Locality in Long Range Arena

## Running experiments
A configuration file needs to be created for each model and task (see the `configs` directory for examples). Then the experiment can be run with the following command:

```
python scripts/train.py --config configs/<config-file>
```

The script is meant to be run from the root dir of the repository. Otherwise, you need to modify the following line:
```diff
- sys.path.append(os.path.join(os.getcwd(), "src"))
+ sys.path.append(os.path.join(os.getcwd(), <RELATIVE_SRC_DIR>)) 
```

## Config possible values
- Tasks:
    - `"text-classification"`
- Models:
    - `"conv-transformer"`
    - `"transformer-base"`