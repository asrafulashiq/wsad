# Weakly Supervised Temporal Action Localization Using Deep Metric Learning

This repository contains Pytorch code for our paper **[Weakly Supervised Temporal Action Localization Using Deep Metric Learning](https://arxiv.org/pdf/2001.07793.pdf)** which has been accepted to WACV 2020.



## Installation

This package requires Pytorch 1.4. Other Pytorch versions should also work. You can create environment using conda:

```
conda env create -n wsad -f environment.yml
conda activate wsad
```


## Download Dataset

Download Thumos14-reduced dataset from [here](https://emailucr-my.sharepoint.com/personal/sujoy_paul_email_ucr_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsujoy%5Fpaul%5Femail%5Fucr%5Fedu%2FDocuments%2Fwtalc%2Dfeatures&originalPath=aHR0cHM6Ly9lbWFpbHVjci1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9zdWpveV9wYXVsX2VtYWlsX3Vjcl9lZHUvRXMxemJIUVk0UHhLaFVrZGd2V0h0VTBCSy1feXVnYVNqWEs4NGtXc0IwWEQwdz9ydGltZT1SM1FmR1FTaTEwZw).

## Training

```
python main.py
```

## Testing

```
python main.py --test --ckpt [pretrained weight]
```

## Citation

If you find this work useful, please cite our work:

```
@misc{islam2020weakly,
    title={Weakly Supervised Temporal Action Localization Using Deep Metric Learning},
    author={Ashraful Islam and Richard J. Radke},
    year={2020},
    eprint={2001.07793},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgments

[sujoyp/wtalc-pytorch](https://github.com/sujoyp/wtalc-pytorch)
