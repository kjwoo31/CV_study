# [DeepLearning Course](http://introtodeeplearning.com/?fbclid=IwAR2wCGZ_DrzzdpU2OLZHmXjZy9H14NfEXwat0d9L4IdbN76LHkgoHXqlidc)

## Lecture 1 | Introduction to Deep Learning
  
### 1. Why Deep Learning?  
- Deep Learning: Extract patterns from data using neural networks
- Why now? : Big data, HW(GPU, parallelizable), SW

### 2. Perceptron (single neuron): The structural building block of deep learning  
<img src="https://user-images.githubusercontent.com/59794238/119324261-9a15f000-bcba-11eb-9478-e9584f64efa4.png" width="50%"></img>  
1) input, weight의 dot product
2) add bias
3) Non-Linearity g : activation function, linear decision -> arbitrarily complex functions
- Sigmoid Function: 0~1의 결과. 확률 관련 문제에 적합함.
- Rectified Linear Unit(ReLU): 음수면 0, 양수면 z. 단순해서 많이 사용.

### 3. Building Neural Network
- Multi Output Perceptron
	- **Dense** layer : Layer between input, output. Because all inputs are densely connected to all outputs.  
	<img src="https://user-images.githubusercontent.com/59794238/119324345-ac902980-bcba-11eb-955b-2ec2873e633a.png" width="40%"></img> <img src="https://user-images.githubusercontent.com/59794238/119324405-bca80900-bcba-11eb-984f-e9621af32d3e.png" width="40%"></img>  

- Deep Neural Network
	- **Hidden** layer : Unlike input and output layer, they're hidden to some extent  
	<img src="https://user-images.githubusercontent.com/59794238/119324744-21636380-bcbb-11eb-9fff-5df63ab4fc8a.png" width="40%"></img>  

### 4. Applying Neural Network
- Loss: The cost incurred from incorrect predictions. (Empirical Loss: Average of Loss)
	- Softmax Cross Entropy Loss: Useful in binary classification. Cross entropy between two probability distributions.  
	<img src="https://user-images.githubusercontent.com/59794238/119324898-48ba3080-bcbb-11eb-9dc1-36f534b8bfaf.png" width="50%"></img>  
	- Mean Squared Error Loss: Predicting binary outputs. 분산.  
	<img src="https://user-images.githubusercontent.com/59794238/119324935-54a5f280-bcbb-11eb-967e-a50bf2d59e76.png" width="50%"></img>  

### 5. Training Neural Network
- Loss가 최소인 weight를 찾는다.
- Gradient Descent  
	<img src="https://user-images.githubusercontent.com/59794238/119324969-5d96c400-bcbb-11eb-89f0-7c58968b6c87.png" width="50%"></img>  
	- Computing Gradients: Backpropagation (Use Chain Rule)  
	<img src="https://user-images.githubusercontent.com/59794238/119325004-67202c00-bcbb-11eb-9050-f8a45d1553c3.png" width="50%"></img>  
	- SGD: 전체 데이터 대신 batch of data points를 받아 compute gradient estimation
	- 그 외에도 Adam, Adadelta, Adagrad, RMSProp이 있다.  
	<img src="https://user-images.githubusercontent.com/59794238/119325162-959e0700-bcbb-11eb-9d9a-63b6a3b4a12d.png" width="50%"></img>

### 6. Optimization
- Setting the Learning Rate: 작으면 local minima에 갇히고 크면 overshoot. 따라서, 학습 과정에 따라 적응하는 Adaptive Learning Rate 사용.
- Regularization: Model이 너무 복잡해지는 것을 막는 과정. 모델의 일반화, overfitting 방지.
	- Dropout: During training, randomly set some activations to 0.  
	<img src="https://user-images.githubusercontent.com/59794238/119325451-d5fd8500-bcbb-11eb-9fe8-65023d8e4c84.png" width="40%"></img>  
	- Early Stopping: Stop training before we have a chance to overfit.  
	<img src="https://user-images.githubusercontent.com/59794238/119325046-6f786700-bcbb-11eb-87be-8bf70afe3c49.png" width="40%"></img>  

</br>

## Lecture 2 | Recurrent Neural Networks

### 1. Sequence Modeling Applications - 데이터 간 연관성 존재. Add time component.  
<img src="https://user-images.githubusercontent.com/59794238/119547009-87d9a600-bdcf-11eb-878d-ad0edd911bbd.png" width="30%"></img>  

### 2. Neurons with recurrence  
<img src="https://user-images.githubusercontent.com/59794238/119545633-003f6780-bdce-11eb-8202-9440e78ce1a3.PNG" width="30%"></img>  
1) Make Feed-Forward Network for each time step.
2) Apply recurrence relation to pass the past memory. (Connect Hidden States)

### 3. Recurrent Neural Network (RNN)  
<img src="https://user-images.githubusercontent.com/59794238/119545699-0fbeb080-bdce-11eb-8f2c-d0c01753b63a.PNG" width="40%"></img> <img src="https://user-images.githubusercontent.com/59794238/119545881-409ee580-bdce-11eb-8d0f-13e84dea0adc.PNG" width="30%"></img>  
- Unfolding RNNs : Re-use the **same weight matrices** at every time step. Sum all losses.  
<img src="https://user-images.githubusercontent.com/59794238/119547156-b9527180-bdcf-11eb-8b80-9ae94408ce7f.png" width="40%"></img>  
- Use Call function to make a forward pass (tf.kears.layers.simpleRNN(rnn_units))  
<img src="https://user-images.githubusercontent.com/59794238/119545939-501e2e80-bdce-11eb-87c3-09050f084218.PNG" width="40%"></img>  

### 4. Sequence Modeling: Design Criteria
- Word prediction example: Encoding Language for a Neural Network (word -> vector)
1. Handle Variable Sequence Lengths
- Feed forward networks are not able to do this becuase they have inputs of fixed dimensionality.
- But in RNN, differences in sequence lengths are just differences in the number of time steps.
2. Long-Term Dependencies
- We need information from the distant past to accurately predict the correct word.
3. Capture Differences in Sequence Order : 순서가 중요함

### 5. Backpropagation Through Time (BPTT)  
<img src="https://user-images.githubusercontent.com/59794238/119546000-62986800-bdce-11eb-86a0-f9cf40892929.PNG" width="40%"></img>  
각 timestep에 대해 backpropagation을 한 후 최근->처음으로 pass
- Gradient Issues : During backpropagation, we repeat gradient computation! (W_hh backpropagation 반복)
	- Many values > 1: exploding gradients -> Gradient clipping (threshold 설정)
	- Many values < 1: vanishing gradients. 최종값은 Bias에 의지하고 Long-Term Dependencies 고려 X.
		- Use ReLU : x>0에서 미분값이 항상 1. Prevents gradient shrinking.
		- Parameter Initialization: Initialize weights, biases to zero.
		- Gated Cells: Use a more **complex recurrent unit with gates** (LSTM)

### 6. Long Short Term Memory (LSTM) Networks  
<img src="https://user-images.githubusercontent.com/59794238/119546752-3af5cf80-bdcf-11eb-963b-facdb1167ce9.PNG" width="40%"></img>  
- Information is added or removed through structures called gates.
- Forget -> Store -> Update -> Output (Sigmoid gate로 조절)

### 7. RNN Applications
1) Music Generation : Generate new composition.
2) Sentiment Classification : Use cross entropy about the output of sequence of words.
3) Machine Translation : Vector로 바꾸는 Encoder, 다른 언어로 바꾸는 Decoder 사용  
	<img src="https://user-images.githubusercontent.com/59794238/119546781-434e0a80-bdcf-11eb-9988-53d308b439b8.PNG" width="30%"></img>  
	- 데이터 양이 많아 발생하는 문제를 Attention을 사용하여 해결.  
	<img src="https://user-images.githubusercontent.com/59794238/119546809-4c3edc00-bdcf-11eb-9ba2-45d716e4af5c.PNG" width="30%"></img>  

</br>

## Lab 1 | Intro to TensorFlow; Music Generation
### 1. TensorFlow
1) 정의
- Shape는 차원의 크기, rank는 차원의 수
- tf.constant, tf.zeros 등으로 정의, 행렬과 같이 사용.
2) Computation
- tf.add, tf.matmul, tf.sigmoid 등 computation function 사용
3) Neural network
- __init__에는 model의 layer 정의, call에는 model의 forward pass 정의.
- Dense layer 정의: tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')
- **Layer에는 output space의 차원을 적는다.**
4) Automatic differentiation
- with tf.GradientTape() as tape: # Initiate the gradient tape, 미분할 변수 사이 관계 정의
- dy_dx = tape.gradient(y,x)

### 2. Music Generation with RNNs
1) Dependencies, Dataset - 817 song with 83 unique characters
2) Process the dataset
- Vectorize the text : char2idx = {u:i for i, u in enumerate(vocab)}
- Create training examples and targets : break text into chunks of 'seq_length+1' (batch: 한 번 학습할 때 사용하는 데이터 배열)
3) RNN  
<img src="https://raw.githubusercontent.com/aamini/introtodeeplearning/2019/lab1/img/lstm_unrolled-01-01.png" width="50%"></img>  
- Layer: Embedding(vector 변환), LSTM(RNN), Dense
- get batch, pred = model(x)
4) Training the Model: loss ant training operations
- Adam optimizer 이용, optimizer에 gradient 값들 apply, loss return
- batch를 골라 loss를 확인하고 update
5) Generate music using the RNN model  
<img src="https://raw.githubusercontent.com/aamini/introtodeeplearning/2019/lab1/img/lstm_inference.png" width="50%"></img>  
- batch_size=1인 학습된 RNN model을 재활용하여 예측

</br>

## Lecture 3 | Convolutional Neural Networks
### 1. Learning Visual Features
- Feature Extraction with Convolution : Apply filters to extract local features.  
<img src="https://user-images.githubusercontent.com/59794238/119630073-8f3d9580-be49-11eb-94a3-ce1a78cc6b2e.PNG" width="40%"></img>  

### 2. Convolutional Neural Networks (CNNs)  
<img src="https://user-images.githubusercontent.com/59794238/119630100-95cc0d00-be49-11eb-8657-cdb7fd6f885c.PNG" width="50%"></img>  
1) Convolution: Apply filters to generate feature maps.  
	<img src="https://user-images.githubusercontent.com/59794238/119630129-9c5a8480-be49-11eb-95f7-76a01203e810.PNG" width="40%"></img>  
	- 여러 filter 사용
	- Stride를 조절하고 input image에서의 feature 관계(Receptive Field)를 저장
2) Non-linearity: Apply after every convolution operation. Often ReLU.
3) Pooling: Downsampling operation on each feature map.
	- MaxPool: 최댓값 추출  
	<img src="https://user-images.githubusercontent.com/59794238/119630152-a2e8fc00-be49-11eb-9a50-5dd407056df4.PNG" width="40%"></img>  
4) Dense Network to use these features for classifying input image. (softmax classify)


### 3. Applications
- Object Detection
	- Select region and check if there is an object. 선택된 region의 양이 너무 많아지는 문제 발생.
		1) R-CNN: Manually find regions that we think have objects, use CNN
		2) Faster R-CNN: Use conv layer to find region. (Region Proposal Network)  
		<img src="https://user-images.githubusercontent.com/59794238/119630182-aa100a00-be49-11eb-9744-33b111f32b50.png" width="40%"></img>  
	- Semantic Segmentation: Fully Convolutional Networks  
	<img src="https://user-images.githubusercontent.com/59794238/119630200-b1371800-be49-11eb-9432-f71a1fdb9b5d.PNG" width="40%"></img>  
- End-to-End Framework for Autonomous Navigation  
<img src="https://user-images.githubusercontent.com/59794238/119630235-b85e2600-be49-11eb-877a-c9668b3fd06c.PNG" width="40%"></img>  

</br>

## Lecture 4 | Deep Generative Modeling
### 1. Introduction
- Generative modeling은 Unsupervised Learning. Learn the hidden or underlying structure of the data.
- Capable of uncovering **underlying features** in a dataset. 
	- Can make representative data set that is unbiased.
	- Can detect outliers.
- Latent variable: Data를 대표하는 underlying and hidden variable

### 2. Autoencoders : Automatically encoding data.  
<img src="https://user-images.githubusercontent.com/59794238/119799567-23296300-bf17-11eb-946b-4a4bfca4e71f.PNG" width="40%"></img>  
1) Encoder: Learning a **lower-dimensional** feature representation from unlabeled training data. Compress the data into a small latent vector.
2) Decoder: Learns mapping back from latent space to the original data.

### 3. Variational Autoencoders (VAEs) : Add stochastic or variational twist on the architecture to generate smooter represenations  
<img src="https://user-images.githubusercontent.com/59794238/119799590-2c1a3480-bf17-11eb-8d2d-94ccb97e4e07.PNG" width="40%"></img>  
- Loss has regularization term. This part enforces the latent variable to have a same centered mean and all their variances to be regularized. (기준점이 같아져 비교하기 쉬워진다.)  
<img src="https://user-images.githubusercontent.com/59794238/119799619-350b0600-bf17-11eb-96c6-d3b353486e0f.PNG" width="20%"></img> <img src="https://user-images.githubusercontent.com/59794238/119799648-3ccaaa80-bf17-11eb-9c75-e95dc6f4bc9f.PNG" width="20%"></img>   
- z가 확률적 분포를 가지면 backpropagation이 불가능. Fixed vector에 random constant를 더했다고 가정.  
<img src="https://user-images.githubusercontent.com/59794238/119799680-46eca900-bf17-11eb-8675-203a81bbff5b.PNG" width="40%"></img> <img src="https://user-images.githubusercontent.com/59794238/119799717-4eac4d80-bf17-11eb-8779-c799c1db92e7.PNG" width="40%"></img>  
- Regulation되는 정도가 클수록 서로 다른 요소가 최대한 uncorrelated 됨. (β-VAE)  
<img src="https://user-images.githubusercontent.com/59794238/119799743-57048880-bf17-11eb-98b9-527dfa2a256e.PNG" width="40%"></img>  

### 4. Generative Adversarial Networks (GAN) : Generate synthetic samples that were as faithful to a data distribution generally as possible.  
<img src="https://user-images.githubusercontent.com/59794238/119799779-608df080-bf17-11eb-83ae-9a0efeb41385.PNG" width="40%"></img>  
- Generator, Discriminator network competes against each other.
	- Generator tries to create imitations of data to trick the discriminator.
	- Discriminator tries to identify real data from fakes created by the generator.

### 5. GANs: Recent advances  
- Progressive GANs: Layer의 개수를 점점 늘리면서 훈련 반복. 높은 해상도의 이미지 생성.  
<img src="https://user-images.githubusercontent.com/59794238/119799822-684d9500-bf17-11eb-9bfe-e2ffc306196f.PNG" width="40%"></img>  
- StyleGAN: Style 요소를 추가. Age, facial structure 등의 특징을 반영 가능.  
<img src="https://user-images.githubusercontent.com/59794238/119799856-713e6680-bf17-11eb-8b59-484d63fb861e.PNG" width="40%"></img>  
- Conditional GANs: input을 넣으면 output을 도출하도록 label을 학습.  
<img src="https://user-images.githubusercontent.com/59794238/119799889-7a2f3800-bf17-11eb-80cb-5a4e0a042a07.PNG" width="40%"></img> <img src="https://user-images.githubusercontent.com/59794238/119799938-86b39080-bf17-11eb-9c78-e35c1a7a41e6.PNG" width="40%"></img>  
- CycleGAN: Unpaired data를 활용하여 다른 domain의 data로 변환. (Autoencoder처럼 2개의 Generator, Discriminator network를 사용하여 domain을 왔다갔다 하는 것 같다.)  
<img src="https://user-images.githubusercontent.com/59794238/119799990-916e2580-bf17-11eb-8b4d-089593303a98.PNG" width="40%"></img> <img src="https://user-images.githubusercontent.com/59794238/119800025-97fc9d00-bf17-11eb-93f7-56c94bc9e77f.PNG" width="20%"></img>  

</br>

## Lab 2 | Computer Vision
### 1. MNIST Digit Classification
<img src="https://raw.githubusercontent.com/aamini/introtodeeplearning/master/lab2/img/mnist_2layers_arch.png" width="50%"></img>  
1. Dataset : (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(), 차원 변환
2. Model : 중간 layer는 ReLU, 마지막 layer는 softmax
3. model.compile : optimizer (update 방식), loss, metrics (monitor steps)
4. model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
4-2. tf.GradientTape() 사용. 
	> grads = tape.gradient(loss_value, cnn_model.trainable_variables)  
	> optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))
5. test_loss, test_acc = model.evaluate(test_images, test_labels)
6. model.predict 이후 가장 높은 confidence를 갖는 argmax를 찾아 출력
#### Dense Network만 사용하면 overfitting 문제 발생. CNN으로 feature를 추출하여 분류.
- CNN model   
<img src="https://raw.githubusercontent.com/aamini/introtodeeplearning/master/lab2/img/convnet_fig.png" width="70%"></img>  
	- tf.keras.layers.Conv2D : filter 수, kernel_size (2D), activation function
	- tf.keras.layers.MaxPool2D: pool_size (2D)

### 2. Debiasing Facial Detection Systems
- Dataset 분포, training 방식에 따라 bias 발생.
1. skin tone, gender를 균일하게 만들기 위해 아래의 3개 Dataset 사용.
	- 유명인의 얼굴 사진이 있는 CelebA Dataset
	- non-human 사진이 있는 ImageNet
	- skin type 분류가 되어 있는 Fitzpatrick Scale
2. CNN for facial detection
- define our CNN model, and then train on the CelebA and ImageNet datasets
- test on Fitzpatrick Scale. Dark Male에 대한 표본이 적어 차이가 발생.  
<img src="https://user-images.githubusercontent.com/59794238/119994525-a3c78c80-c007-11eb-9515-0b3f8fc8e557.PNG" width="20%"></img>  
3. Mitigating algorithmic bias
- learn features in an unbiased, unsupervised manner, without the need for any annotation, and then train a classifier fairly with respect to these features.
4. Variational autoencoder (VAE) for learning latent structure
- loss function: vae_loss = kl_weight * latent_loss + reconstruction_loss  
<img src="https://user-images.githubusercontent.com/59794238/119994801-e9845500-c007-11eb-8bf1-0544cb6a65fa.PNG" width="60%"></img>  
- reparameterization: backpropagation이 가능하도록 z의 확률적 요소를 epsilon으로 빼낸다.  
<img src="https://user-images.githubusercontent.com/59794238/119994829-f1dc9000-c007-11eb-87b6-d62ad6d95394.PNG" width="20%"></img>  
5. Debiasing variational autoencoder (DB-VAE)  
<img src="https://raw.githubusercontent.com/aamini/introtodeeplearning/2019/lab2/img/DB-VAE.png" width="60%"></img>  
- Change the probability that a given image is used during training based on how often its latent features appear in the dataset. 귀한 feature일수록(like dark skin, sunglasses, or hats) 많이 sampling 된다.
- loss function: 얼굴 사진에 대해서는 VAE loss, Classification loss를 동시에 계산하고 non-얼굴 사진에 대해서는 Classification loss만 계산.  
<img src="https://user-images.githubusercontent.com/59794238/119995205-50097300-c008-11eb-8926-6dca198378ae.PNG" width="60%"></img>  
- Adaptive resampling for automated debiasing with DB-VAE: latent 분포를 확인하고 고르게 분포하도록 training sample probability 변화
- training loop : get_training_sample_probabilities -> get_batch -> train (loss, gradient descent)
- 결과: Biased data에 대해서도 잘 예측  
<img src="https://user-images.githubusercontent.com/59794238/119995236-58fa4480-c008-11eb-98d9-6b9a7b0d3bcb.PNG" width="40%"></img>  

</br>

## Lecture 5 | Reinforcement Learning
### 1. Introduction
- Classes of Learning Problems  
<img src="https://user-images.githubusercontent.com/59794238/120014237-c912c580-c01c-11eb-8db8-0a097cdc081c.PNG" width="40%"></img>  
- Reinforcement Learning (RL): Key Concepts  
<img src="https://user-images.githubusercontent.com/59794238/120014301-dc259580-c01c-11eb-97ec-a5e7ecf221cf.PNG" width="40%"></img>  

### 2. Value Learning : Find Q-function of each state, action.
- Q-function: 특정 state에서의 특정 action에 대한 기대 보상값을 정리하고 기대 보상값의 합이 최대가 되는 policy 설정.
- Deep Q Networks (DQN) : 현재 state가 주어지면 각 action에 대해 다음 state, action의 Q value를 예측하는 Network.  
<img src="https://user-images.githubusercontent.com/59794238/120014321-e182e000-c01c-11eb-9103-59a49c314b49.PNG" width="40%"></img>  
	- Useful in Atari Games.
	- Only handle discrete and small action space. (Complexity)
	- Cannot learn stochastic policies. (Flexibility)

### 3. Policy Learning : Find best policy.
- Policy Gradient (PG) : Q-function 대신 확률 분포로 나타냄.  
<img src="https://user-images.githubusercontent.com/59794238/120014345-e8a9ee00-c01c-11eb-8fe4-c01532bbd129.PNG" width="40%"></img>  
	- This can handle continuous action space.
- Training Policy Gradients  
<img src="https://user-images.githubusercontent.com/59794238/120014369-f0699280-c01c-11eb-8ae7-a3694d51932d.PNG" width="40%"></img>  
	- 실제 환경에서는 '2. Run a policy until termination'을 할 수 없어 대신 simulator 사용.
	- VISTA simulator: Use real data of the world to simulate self-driving.

### 4. RL and the game of Go
1) AlphaGo: learn from human data and RL by self play.  
<img src="https://user-images.githubusercontent.com/59794238/120014397-f65f7380-c01c-11eb-8eb6-9c61b61991be.PNG" width="50%"></img>  
2) AlphaZero: RL, Only self play.
3) MuZero: Learn the rule of the game. 다음 state에 대한 예측을 바탕으로 action.  
<img src="https://user-images.githubusercontent.com/59794238/120014419-fc555480-c01c-11eb-9b98-9f40c0e05ef1.PNG" width="50%"></img>  
