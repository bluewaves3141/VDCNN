
# VDCNN
Reproduction code for [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/pdf/1606.01781.pdf "VDCNN for Text Classification") by Alexis Conneau et al. 2017. Our VDCNN model was implemented using PyTorch, and it can be trained on either [Yelp Review Polarity dataset ](https://www.kaggle.com/irustandi/yelp-review-polarity/version/1 "Yelp Review Polarity dataset") or [Yahoo Answers dataset](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset "Yahoo Answers")  .

# Prerequisites
- Python3
- Torch >= 0.4.0
- Numpy

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
| 9      | 5.96 %        | 4.88 %                |
| 17     | 5.40 %        | 4.50 %                |
| 29     | 5.43 %        | 4.28 %                |


The testing errors for Yahoo Answer dataset are provided below:
| Depths | Testing Error | Testing Error (paper) |
|--------|---------------|-----------------------|
| 9      | 42.69 %       | 27.60 %               |
| 17     | 42.23 %       | 27.51 %               |
| 29     | 43.14 %       | 26.57 %               |


## Computation Time

The computation time for each epoch is provided below:
| Depths | Yelp Polarity Review | Yahoo Answer  |
|--------|----------------------|---------------|
| 9      | 1429 seconds         | 4145 seconds  |
| 17     | 2454 seconds         | 6698 seconds  |
| 29     | 4652 seconds         | 12154 seconds |

The VDCNN model was trained with a total of 15 epochs for each depth, except Yahoo Answer dataset with depth 29 --- only 9 epochs were used.