# [Stanford CS231n](http://cs231n.stanford.edu/2017/syllabus.html)

## Lecture 2 | Image Classification
가장 간단하게 생각할 수 있는 방법은 픽셀별로 L1 distance을 구해 비교하는 것인데, 이를 더 robust하게 만들어야 한다.
1. kNN: region의 label을 결정할 때, instead of copying label from nearest neighbor, take **majority vote** from K closest points (outlier 처리 가능)  
*L1/L2 distance 선택: individual vector가 의미가 있을 경우 L1 distance, 종합적인 비교가 필요할 경우 L2 distance. Data에 따라 차이가 있어 둘 다 사용하고 비교.  
*hyperparameter 선택: train, val, test set으로 나누어 validation set에서 accuracy가 높은 것으로 선택
(test dataset의 경우 new data에서 알고리즘이 어떻게 적용되는지 알 수 없고, unfair한 결과가 발생할 수 있다)  
(data를 여러 개의 fold로 나누어 train/val을 나누는 Cross-validation 방법도 존재)  
하지만 image에서 distance metrics are not informative  
2. linear classification: f(x, W)=Wx+b, data에 대한 정보를 **parameter W**에 저장. Score에 따라 결과값 예측. 하지만 one template만 사용하기 때문에 고차원 문제에 사용하기 어려움.
</br>

## Lecture 3 | Loss Functions and Optimization
#### loss function: Which W가 가장 좋은지 찾는 함수. 다른 score에 비해 margin을 두고 차이가 나야 좋은 결과.  
1) multiclass SVM loss: labeling이 incorrect 한 물체에 대한 score에 비해 correct한 물체 score가 일정 safety margin을 두고 더 높다면, loss가 0. 그렇지 않으면 score 차이에 따라 linear하게 차이가 나는 loss 함수. (min: 0, max: infinity)  
<img src="https://user-images.githubusercontent.com/59794238/92897754-6a8ac480-f458-11ea-8e9e-64534120ca1b.png" width="30%"></img>
<img src="https://user-images.githubusercontent.com/59794238/92896937-c1dc6500-f457-11ea-8c98-0dc17d04f2f8.png" width="30%"></img>  
*처음 시작할 때, incorrect class에 대해서만 1이 출력되기 때문에(s는 0에 가까움) SVM loss는 (number of classes) - 1  
*overfitting되지 않도록 data loss뿐만 아니라 Regularization loss 사용 (lambda를 이용해 test data의 영향을 조절)  
*주로 L2 regularization 사용, sparsity 있는 경우 L1
2) Softmax Classifier (Multinomial Logistic Regression): score를 exponential을 사용해 0~1 사이 확률로 나타냄. correct할 경우 1에 가깝도록 설정. 그 후, 확률에 -log를 취하여 margin을 키우고 부호를 올바르게 설정 (min: 0(이론상), max: infinity)  
<img src="https://user-images.githubusercontent.com/59794238/92897709-6363b680-f458-11ea-9045-6ed36a84bf60.png" width="20%"></img>  
*처음 시작할 때, -log(Class 개수)  
*차이점: SVM은 일정 Margin을 넘어가면 고려 X, Softmax는 제한 X  
#### optimization: loss가 작은 W 찾는 과정. 
gradient descent: slope를 따라 loss가 최소가 되는 W 찾는 방법  
- Stochastic Gradient Descent: 일정 갯수의 minibatch를 거칠 때마다 update하는 방식. 모든 데이터를 반복하면서 학습하면 시간이 오래 걸려대부분의 딥러닝 알고리즘에서 사용.  
*training 중 learning rate/step size parameter가 매우 중요함.  

(기타 feature 출력 연구: HoG 같이 edge 방향에 따라 물체를 구분하려는 연구, 색/빈도 수를 이용한 연구 등도 있었다)  
</br>

## Lecture 4 | Introduction to Neural Networks
#### computational graph: node를 사용하여 연산 과정을 순차적으로 나타내는 graph. backpropagation 가능  
- backpropagation: 연산의 역순으로 따져 gradient 찾는 과정.   
*연산 방향으로 따질 경우 gradient를 찾을 때 전과정을 거쳐서 비효율적인데, 이 방법을 사용하면 chain rule을 사용하여 gradient 찾기가 쉬워진다. (Upstream gradient에 local gradient를 곱해 gradient를 구할 수 있다)  
*layer마다 forward()/ backward() API를 사용하여 결과값/gradient를 출력.  
*벡터 연산 시, gradient shape should be same with the shape of variable  
<img src="https://user-images.githubusercontent.com/59794238/92945284-084db600-f490-11ea-88eb-1108e165ca72.png" width="40%"></img>  
#### neural network: more complex non-linear function을 만들기 위해 여러 layer를 stack한 구조

## Lecture 5 | Convolutional Neural Networks

