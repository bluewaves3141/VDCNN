
# VDCNN
Reproduction code for [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/pdf/1606.01781.pdf "VDCNN for Text Classification") by Alexis Conneau et al. 2017. The VDCNN model was implemented using PyTorch, and it can be trained on either [Yelp Review Polarity dataset ](https://www.kaggle.com/irustandi/yelp-review-polarity/version/1 "Yelp Review Polarity dataset") or [Yahoo Answers dataset](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset "Yahoo Answers")  .

# Usage

## Basic Usage

```
$ python3 VDCNN.py --num_depth=9 --num_class=2 --dataset_path='../../data/yelp_review_polarity_csv/'
```
Note: *num_class* must be 2 for the Yelp Review Polarity dataset and 10 for the Yahoo Answer dataset. Depths of the VDCNN model can be modified by changing *num_depth* to either 9, 17, or 29. Other depths are not implemented.

# Result

## Testing Error

The testing errors for Yelp Polarity Review dataset are provided below:
| Depths | Testing Error | Testing Error (paper) |
|--------|---------------|-----------------------|
| 9      | 5.64 %        | 4.88 %                |
| 17     | 5.48 %        | 4.50 %                |
| 29     | 5.55 %        | 4.28 %                |


The testing errors for Yahoo Answer dataset are provided below:
| Depths | Testing Error | Testing Error (paper) |
|--------|---------------|-----------------------|
| 9      | 30.01 %       | 27.60 %               |
| 17     | 29.32 %       | 27.51 %               |
| 29     | 30.01 %       | 26.57 %               |


## Computation Time

The computation time for each epoch is provided below:
| Depths | Yelp Polarity Review | Yahoo Answer  |
|--------|----------------------|---------------|
| 9      | 1429 seconds         | 4145 seconds  |
| 17     | 2454 seconds         | 6698 seconds  |
| 29     | 4652 seconds         | 12154 seconds |

All experiments were trained for a total of 15 epochs and performed on a single NVIDIA GK110 GPU.