# Frequency-decoupled Quality Assessment Network for RGBT Tracking

### Environment Installation
Create and activate a conda environment:
```
conda create -n fqan python=3.8
conda activate fqan
bash install_fqan.sh
```

### Data Preparation
Download the training datasets, It should look like:
```
$<PATH_of_Datasets>
    -- LasHeR/TrainingSet
        |-- 1boygo
        |-- 1handsth
        ...
```

### Path Setting
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### Training
Dowmload the pretrained [DropTrack]
and put it under ./pretrained/.
```
python tracking/train.py --script tbsi_track --config rgbt --save_dir ./output/rgbt --mode single
```

### Evaluation

#### For RGB-T benchmarks
```
python tracking/test.py tbsi_track rgbt --dataset_name lasher_test --threads 4 --num_gpus 1
```

We refer you to use [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to use [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.
You can also use `eval/eval_lasher.py` to evaluate the results on the LasHeR dataset.


## Acknowledgment
- This repo is based on [TBSI]([https://github.com/SparkTempest/BAT](https://github.com/RyanHTR/TBSI)) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) library.
