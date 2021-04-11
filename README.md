
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