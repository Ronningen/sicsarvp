# Data-Driven Uncertainty-Aware Forecasting of Sea Ice Conditions in the Gulf of Ob Based on Satellite Radar Imagery

## Citation
```
@Article{Ailuro2025,
      title={Data-driven uncertainty-aware forecasting of sea ice conditions in the gulf of Ob based on satellite radar imagery},
      author={Stefan Maria Ailuro and Anna Nedorubova and Timofey Grigoryev and Evgeny Burnaev and Vladimir Vanovskiy},
      journal={Scientific Reports},
      year={2025},
      abstract={The increase in Arctic marine activity due to rapid warming and significant sea ice loss necessitates highly reliable, short-term sea ice forecasts to ensure maritime safety and operational efficiency. In this work, we present a novel data-driven approach for sea ice condition forecasting in the Gulf of Ob, leveraging sequences of radar images from Sentinel-1, weather observations, and GLORYS forecasts. Our approach integrates advanced video prediction models, originally developed for vision tasks, with domain-specific data preprocessing and augmentation techniques tailored to the unique challenges of Arctic sea ice dynamics. Central to our methodology is the use of uncertainty quantification to assess the reliability of predictions, ensuring robust decision-making in safety-critical applications. Furthermore, we propose a confidence-based model mixture mechanism that enhances forecast accuracy and model robustness, crucial for reliable operations in volatile Arctic environments. Our results demonstrate substantial improvements over baseline approaches, underscoring the importance of uncertainty quantification and specialized data handling for effective and safe operations and reliable forecasting.},
      issn={2045-2322},
      doi={10.1038/s41598-025-16572-7},
      url={https://doi.org/10.1038/s41598-025-16572-7}
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
