# [Stanford CS231n](http://cs231n.stanford.edu/2017/syllabus.html)를 듣고 정리한 내용

## Lecture 2 | Image Classification
가장 간단하게 생각할 수 있는 방법은 픽셀별로 L1 distance 비교하는 것인데, 이를 더 robust하게 만들어야 한다.
1. kNN: region의 label을 결정할 때, instead of copying label from nearest neighbor, take **majority vote** from K closest points (outlier 처리 가능)  
*L1/L2 distance 선택: individual vector가 의미가 있을 경우 L1 distance, 종합적인 비교가 필요할 경우 L2 distance. Data에 따라 차이가 있어 둘 다 사용하고 비교.  
*hyperparameter 선택: train, val, test set으로 나누어 validation set에서 accuracy가 높은 것으로 선택
(test dataset의 경우 new data에서 알고리즘이 어떻게 적용되는지 알 수 없고, unfair한 결과가 발생할 수 있다)  
(data를 여러 개의 fold로 나누어 train/val을 나누는 Cross-validation 방법도 존재)  
하지만 image에서 distance metrics are not informative  
2. linear classification: f(x, W)=Wx+b, data에 대한 정보를 **parameter W**에 저장. Score에 따라 결과값 예측. 하지만 one template만 사용하기 때문에 고차원 문제에 사용하기 어려움.
</br>

## Lecture 3 | Loss Functions and Optimization

