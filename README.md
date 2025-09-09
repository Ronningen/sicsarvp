# Data-Driven Uncertainty-Aware Forecasting of Sea Ice Conditions in the Gulf of Ob Based on Satellite Radar Imagery

## Citation
```
{ailuro2024sicsarvp,
      title={Data-Driven Uncertainty-Aware Forecasting of Sea Ice Conditions in the Gulf of Ob Based on Satellite Radar Imagery}, 
      author={Stefan Maria Ailuro and Anna Nedorubova and Timofey Grigoryev and Evgeny Burnaev and Vladimir Vanovskiy},
      year={2024},
      eprint={2410.19782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.19782}, 
}
```

## Training
First, download preprocessed [data](https://webdav.appliedai.tech/radars/)
The data directory should look like the following:
```
data3/
   |-- sentinel-HV_2014-10-25_2023-12-31.npy
   ...
weights3/
```

Second, run `train.ipynb` notebook
