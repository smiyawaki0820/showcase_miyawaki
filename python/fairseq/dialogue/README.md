# Set-up
```bash
$ bash scripts/setup.sh
```

# Preprocess
```bash
$ bash sh/preprocess.sh \
  -i [data_dir datasets/parallel]
```

# Train
```bash
$ bash sh/train.sh \
  -g [gpu 0] \
  -i [data_dir datasets/data-bin] \
  -o [dest_dir results]
```

# Interactive
```bash
$ bash sh/interactive.sh \
  -g [gpu 0] \
  -i [model results/models/checkpoint_last.pt]
```
