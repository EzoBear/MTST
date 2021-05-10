# MTST
This method is not used in the our system in latest version. There is a way to perform better with smaller models. We also use [this method](https://wandb.ai/jhartquist/fastaudio-esc-50/reports/Fine-Tuning-ResNet-18-for-Audio-Classification--VmlldzoyOTAyMzc). We have applied MTST to the above method, however, we found that it causes performance degradation. Therefore, it is no longer supported. As a reference, this method has been developed in January 2019. It is technically insufficient because the paper was published after long time spend for user experiments rather than method of a similar published date. Therefore, it is recommended to use the latest method rather than our methods.

## Project Structure

```
* MTST
  - ESC-50-master(Dataset)
  + weights
      + fold1
          - weight.pth
          + result.out
      + fold2
      ...
      + fold5
  + test.py
```

| Fold | Accuracy | Recall | Precision | Weighted F1 | MCC   | AUC   |
| ---  | ---      | ---    | ---       | ---         | ---   | ---   |
| 1    | 81.2%    | 81.2%  | 84.4%     | 81.1%       | 81.0% | 90.4% |
| 2    | 81.0%    | 81.0%  | 83.5%     | 80.4%       | 80.7% | 90.3% |
| 3    | 81.2%    | 81.2%  | 82.6%     | 80.7%       | 80.9% | 90.4% |
| 4    | 84.8%    | 84.8%  | 85.7%     | 84.2%       | 83.7% | 91.8% |
| 5    | 81.2%    | 81.2%  | 84.0%     | 81.1%       | 80.9% | 90.4% |
| avg. | 81.9%    | 81.9%  | 84.0%     | 81.5%       | 81.4% | 90.7% |

## Download
**[Download ESC-50 dataset](https://github.com/karolpiczak/ESC-50)** 

**[Download weights](https://drive.google.com/drive/folders/1erj81zy_5l4TJJbwz8afid2HuEEFZHHz?usp=sharing)**
