# preprocess
## making csv
```
cd /mnt/hdd/jeff/mdgen-piezo/model/mdgen2026
export PYTHONPATH="/mnt/hdd/jeff/mdgen-piezo/model/mdgen2026"

python scripts/prep-protein-csv.py \
  --input_dir data/pdbxtc \
  --output_csv data/proteins-mdgen2026.csv
```

## making npy
```
cd /mnt/hdd/jeff/mdgen-piezo/model/mdgen2026
export PYTHONPATH="/mnt/hdd/jeff/mdgen-piezo/model/mdgen2026"

python scripts/prep_sims.py \
  --split data/proteins-mdgen2026.csv \
  --atlas_dir data/pdbxtc \
  --outdir data/npy \
  --num_workers 4 \
  --stride 40 \
  --atlas
```


# train
```
cd /mnt/hdd/jeff/mdgen-piezo/model/mdgen2026
export PYTHONPATH="/mnt/hdd/jeff/mdgen-piezo/model/mdgen2026"

python train.py \
  --sim_condition \
  --train_split data/proteins-mdgen2026.csv \
  --val_split data/proteins-mdgen2026.csv \
  --data_dir data/npy \
  --model_dir ckpt \
  --batch_size 1 \
  --prepend_ipa \
  --crop 256 \
  --atlas \
  --run_name train-origin \
  --val_repeat 25 \
  --epochs 10000 \
  --ckpt_freq 10000 \
  --num_frames 250 \
  --grad_checkpointing
```

# inference
```
cd /mnt/hdd/jeff/MDGen-piezo-l1
export PYTHONPATH="/mnt/hdd/jeff/MDGen-piezo-l1/model/mdgen2026"

python model/mdgen2026/sim_inference.py \
--sim_ckpt ckpt/atlas.ckpt \
--data_dir data/atlas/npy \
--num_frames 250 \
--num_rollouts 1 \
--suffix _R1 \
--split data/atlas/manifests/split0924.csv \
--out_dir projects/mdgen2026/0924-main-module/inference-origin \
--xtc
```