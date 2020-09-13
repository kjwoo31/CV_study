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
1) multiclass SVM loss: labeling이 incorrect 한 물체에 대한 score에 비해 correct한 물체 score가 일정 safety **margin**을 두고 더 높다면, loss가 0. 그렇지 않으면 score 차이에 따라 linear하게 차이가 나는 loss 함수. (min: 0, max: infinity)  
<img src="https://user-images.githubusercontent.com/59794238/92897754-6a8ac480-f458-11ea-8e9e-64534120ca1b.png" width="30%"></img>
<img src="https://user-images.githubusercontent.com/59794238/92896937-c1dc6500-f457-11ea-8c98-0dc17d04f2f8.png" width="30%"></img>  
*처음 시작할 때, incorrect class에 대해서만 1이 출력되기 때문에(s는 0에 가까움) SVM loss는 (number of classes) - 1  
*overfitting되지 않도록 data loss뿐만 아니라 Regularization loss 사용 (lambda를 이용해 test data의 영향을 조절)  
*주로 L2 regularization 사용, sparsity 있는 경우 L1
2) Softmax Classifier (Multinomial Logistic Regression): score를 exponential을 사용해 0~1 사이 **확률**로 나타냄. correct할 경우 1에 가깝도록 설정. 그 후, 확률에 **-log**를 취하여 margin을 키우고 부호를 올바르게 설정 (min: 0(이론상), max: infinity)  
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
- backpropagation: 연산의 역순으로 따져 **gradient 찾는 과정**.   
*연산 방향으로 따질 경우 gradient를 찾을 때 전과정을 거쳐서 비효율적인데, 이 방법을 사용하면 **chain rule**을 사용하여 gradient 찾기가 쉬워진다. (Upstream gradient에 local gradient를 곱해 gradient를 구할 수 있다)  
*layer마다 forward()/ backward() API를 사용하여 결과값/gradient를 출력.  
*벡터 연산 시, gradient shape should be same with the shape of variable  
<img src="https://user-images.githubusercontent.com/59794238/92945284-084db600-f490-11ea-88eb-1108e165ca72.png" width="40%"></img>  
#### neural network: more complex non-linear function을 만들기 위해 여러 layer를 stack한 구조
</br>

## Lecture 5 | Convolutional Neural Networks
#### Conv Layer: image보다 작은 크기의 spatial filter(w)를 image의 spatial location과 dot product. (convolve/slide한다 표현) filter 하나 당 한 개의 activation map을 생성하고 이를 stack하여 원하는 출력을 얻는 방식.  
*spatial dimension: 크기 N의 input, 크기 F의 filter가 있다면, filter 간 간격에 따라 Output size는 (N-F) / stride + 1 으로 결정된다. (Output size를 image size와 같게 하기 위해 padding 추가, (N+2xpad-F) / stride + 1)  
*parameter 수: (filter 크기+1)x(filter 수)  
*5x5 filter = 5x5 receptive field for each neuron  
*이전 fully connected layer는 모든 streched out input이 연결되어 있었지만 Conv layer에서는 local spatial region이 연결됨.  
#### Pooling layer: makes the represantations smaller and more manageable, (Downsampling)
*Max pooling: filter에 속한 내용 중 max 값을 output으로 하는 pooling 방법  
*common settings for pooling: F=2, S=2 / F=3, S=2  
#### FC Layer (Fully Connected Layer)
<img src="https://user-images.githubusercontent.com/59794238/92955272-d0e70580-f49f-11ea-983a-dba1df5d8dd5.png" width="40%"></img>  
</br>

## Lecture 6 | Training Neural Networks, part I
### 1. setup
#### activation function
*sigmoid: 1/(1+exp(x)). can kill gradient, output not zero-centered (gradient가 모두 같은 부호로 update되어 비효율적)  
*tanh(x): zero-centered, still can kill gradient  
*ReLU: max(0, x). 속도가 빠르고 AlexNet에 사용됨, 음수일 경우 kill gradient, not zero-centered. (초기화를 잘못하거나 learning rate이 높으면 ReLU가 Data Cloud로부터 update를 받지 못할 수 있음. 이를 피하기 위해 positive biases를 추가하기도 하는데, 이 방법이 효과가 있는지는 확실하지 않다.)  
*Leaky ReLU: max(0.01x, x). ReLU의 음수에서의 문제를 해결.  
(0.01을 parameter로 바꾼 PReLU, zero mean output을 얻기 위해 조절한 ELU 등 다른 ReLU, weight를 사용한 Maxout도 존재)  
#### preprocessing : zero-mean, normalization을 함. 이미지의 경우, channel별로 zero-mean 수행 후 학습  
#### weight initialization
*모두 0으로 initialize되면 모든 nueron이 같은 연산, 같은 update을 하게 됨.  
*Small random numbers: Small data에서는 잘 작동하지만 activation 함수를 사용한 deep network에서는 w가 너무 작은 겂이라 곱할수록 출력 값이 급격히 줄며, 수렴하게 된다. Backprop시에도 gradient 또한 엄청 작은 값이 되어 업데이트가 잘 일어나지 않는다.  
*making weight big: output either very negative or very positive, gradient is 0.  
*Xavier initialization: 적절한 가중치로 초기화하는 방법. W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in). (Standard gaussian으로 뽑은 값을 입력의 수로 scaling.) 이 식을 통해 input, output의 variance를 맞춰준다. Linear activation이 있다는 가정을 포함한다.  
*Xavier initialization in ReLU: ReLU는 출력의 절반을 0으로 만들기 때문에 입력의 수를 반으로 줄여야 한다.  
#### batch normalization: Layer의 출력을 unit Gaussian(mean μ = 0 and standard deviation σ = 1)으로 만들기 위해 batch 단위로 normalization하는 과정
*Conv Layer에서의 normalization은 같은 Activation Map의 같은 채널에 있는 요소들은 같이 Normalize 해주고, batch norm은 feature element별로(차원별로) Normalization한다. (CNN에서는 Spatial structure가 유지되기를 원하기 때문)  
*Batch Norm 연산은 FC(Fully-Connected Layer)나 Conv(Convolution Layer) 직후에 넣는다.  
*scaling, shifting factor를 추가하여 얼마나 saturation이 일어나면 좋을지 조절 가능. 네트워크 값들을 복구하고 싶다면 y=rx+b에서 r에 분산 값을, b에 평균값을 넣어주면 된다.  
*Batch Norm을 쓰면 learning rate를 키울 수 있고(데이터의 크기에 영향을 덜 ) 다양한 초기화 기법들도 사용해 볼 수 있으며 Regulation 효과를 준다.  
*Batch Norm에서 평균과 분산은 학습 데이터로부터 얻는다. Test할 때에 추가적인 계산을 하지는 않는다.  
<img src="https://user-images.githubusercontent.com/59794238/93011037-c88ce880-f5cd-11ea-958a-a71fae62db2d.png" width="40%"></img>  
</br>

### 2. training
#### babysitting the learning process: check the loss is reasonable, start with small portion of the training data (overfit해야 함)
#### hyperparameter optimization: Cross-validation strategy 사용. few epoch 동안 돌려보고, 좋은 parameter 찾기를 반복한다.
*Learning rate: 너무 작으면 update가 잘 이루어지지 않아 loss barely change. 너무 크면 loss explode, NaN 발생. (cost가 original cost의 3배 이상이면 learning rate이 너무 크다고 보면 됨.)  
<img src="https://user-images.githubusercontent.com/59794238/93001071-72855a00-f567-11ea-80e8-f7a3d747ce15.png" width="40%"></img>  
*parameter별로 일정 간격을 두고 찾는 grid search, 범위 내의 임의의 parameter를 사용하는 random search가 있다. Random search를 사용하면 important parameter에 대해 더 많은 경우의 수를 볼 수 있어 더 좋은 경우가 있다.  
*ratio of weight updates/weight magnitudes: 0.001 근처가 적당하다  
</br>

## Lecture 7 | Training Neural Networks, part II
#### fancier optimization
*SGD(Stochastic gradient descent)는 local minima, saddle point에서 멈추는 문제 발생   
-> 방향 유지. 속도, 마찰을 포함한 momentum 개념을 추가하여 해결 (Momentum, Nesterov Momentum)  
-> 보폭 조절. learning rate에 squared gradient를 더하여 점차 learning rate를 줄여감. RMSProp는 새로운 기울기의 정보만 반영하여 learning rate이 급격히 줄어드는 것을 방지하는 방법 (AdaGrad, RMSProp)  
-> 둘을 섞은 기법. first timestep에서 second moment을 0으로 초기화하였기 때문에 large step을 하게 된다. 이를 방지하기 위해 bias correction을 해준다. 주로 사용되는 optimization 방법이고 설정은 beta1=0.9, beta2=0.999, learning rate=1e-3 or 5e-4로 주로 한다. (Adam)  
*high learning rate, low learning rate의 장점을 모두 사용하기 위해 learning rate decay를 사용하기도 함.  
*second-order optimization를 사용한 방법도 있다 (Hessian Matrix를 사용한 방법은 matrix 크기가 N^2꼴이라 크기가 너무 커져 딥러닝에 사용하기 어렵고 대신 Quasi-Newton method를 활용한 BGFS가 사용됨. Style transfer와 같이 less stochasticity, fewer parameter가 있을 때 사용)  
#### model ensembles: 여러 개 모델을 동시에 사용하여 예측하고 결과값을 종합하는 방법. Maximal performance를 얻기 위해 사용
*tips: use multiple snapshots of a single model, use moving average of the parameter vector rather than actual parameter vector (polyak averaging)  
#### Regularization: single-model에서 overfitting을 방지하여 성능 향상
*Dropout: train 과정에서 반 정도의 뉴런을 비활성화하여 학습하는 방식. Test time에는 probability를 곱하여 randomness를 줄인다. (test time의 probability를 곱하는 과정을 train 과정에서 하는 inverted-dropout이 more common)  
*Data Augmentation: 뒤집기, 자르기 등을 통해 데이터 형태를 다양하게 함  
*DropConnect, Fractional Max Pooling, Stochastic Depth 등 common하진 않지만 다른 regularization 방법도 있다.  
#### Transfer Learning: ImageNet과 같은 큰 dataset으로 학습시킨 후, layer들을 freeze하고 small dataset만 reinitialize하여 학습하는 방법.
*학습시키려는 Dataset이 큰 경우, 더 많은 layer를 갖고 train. (finetuning)  
</br>

## Lecture 8 | Deep Learning Software
