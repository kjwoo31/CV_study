# [DeepLearning Course](http://introtodeeplearning.com/?fbclid=IwAR2wCGZ_DrzzdpU2OLZHmXjZy9H14NfEXwat0d9L4IdbN76LHkgoHXqlidc)

## Lecture 1 | Introduction to Deep Learning
  
1. Why Deep Learning?  
- Deep Learning: Extract patterns from data using neural networks
- Why now? : Big data, HW(GPU, parallelizable), SW

2. Perceptron (single neuron): The structural building block of deep learning  
	<img src="https://user-images.githubusercontent.com/59794238/119324261-9a15f000-bcba-11eb-9478-e9584f64efa4.png" width="50%"></img>  
	1) input, weight의 dot product
	2) add bias
	3) Non-Linearity g : activation function, linear decision -> arbitrarily complex functions
- Sigmoid Function: 0~1의 결과. 확률 관련 문제에 적합함.
- Rectified Linear Unit(ReLU): 음수면 0, 양수면 z. 단순해서 많이 사용.

3. Building Neural Network
- Multi Output Perceptron
	- **Dense** layer : Layer between input, output. Because all inputs are densely connected to all outputs.  
	<img src="https://user-images.githubusercontent.com/59794238/119324345-ac902980-bcba-11eb-955b-2ec2873e633a.png" width="40%"></img> <img src="https://user-images.githubusercontent.com/59794238/119324405-bca80900-bcba-11eb-984f-e9621af32d3e.png" width="40%"></img>  

- Deep Neural Network
	- **Hidden** layer : Unlike input and output layer, they're hidden to some extent  
	<img src="https://user-images.githubusercontent.com/59794238/119324744-21636380-bcbb-11eb-9fff-5df63ab4fc8a.png" width="40%"></img>  

4. Applying Neural Network
- Loss: The cost incurred from incorrect predictions. (Empirical Loss: Average of Loss)
	- Softmax Cross Entropy Loss: Useful in binary classification. Cross entropy between two probability distributions.  
	<img src="https://user-images.githubusercontent.com/59794238/119324898-48ba3080-bcbb-11eb-9dc1-36f534b8bfaf.png" width="50%"></img>  
	- Mean Squared Error Loss: Predicting binary outputs. 분산.  
	<img src="https://user-images.githubusercontent.com/59794238/119324935-54a5f280-bcbb-11eb-967e-a50bf2d59e76.png" width="50%"></img>  

5. Training Neural Network
- Loss가 최소인 weight를 찾는다.
- Gradient Descent  
	<img src="https://user-images.githubusercontent.com/59794238/119324969-5d96c400-bcbb-11eb-89f0-7c58968b6c87.png" width="50%"></img>  
	- Computing Gradients: Backpropagation (Use Chain Rule)  
	<img src="https://user-images.githubusercontent.com/59794238/119325004-67202c00-bcbb-11eb-9050-f8a45d1553c3.png" width="50%"></img>  
	- SGD: 전체 데이터 대신 batch of data points를 받아 compute gradient estimation
	- 그 외에도 Adam, Adadelta, Adagrad, RMSProp이 있다.  
	<img src="https://user-images.githubusercontent.com/59794238/119325162-959e0700-bcbb-11eb-9d9a-63b6a3b4a12d.png" width="50%"></img>

6. Optimization
- Setting the Learning Rate: 작으면 local minima에 갇히고 크면 overshoot. 따라서, 학습 과정에 따라 적응하는 Adaptive Learning Rate 사용.
- Regularization: Model이 너무 복잡해지는 것을 막는 과정. 모델의 일반화, overfitting 방지.
	- Dropout: During training, randomly set some activations to 0.  
	<img src="https://user-images.githubusercontent.com/59794238/119325451-d5fd8500-bcbb-11eb-9fe8-65023d8e4c84.png" width="40%"></img>  
	- Early Stopping: Stop training before we have a chance to overfit.  
	<img src="https://user-images.githubusercontent.com/59794238/119325046-6f786700-bcbb-11eb-87be-8bf70afe3c49.png" width="40%"></img>  

</br>

## Lecture 2 | Recurrent Neural Networks

1. Sequence Modeling Applications - 데이터 간 연관성 존재. Add time component.  
<img src="https://user-images.githubusercontent.com/59794238/119547009-87d9a600-bdcf-11eb-878d-ad0edd911bbd.png" width="30%"></img>  

2. Neurons with recurrence  
<img src="https://user-images.githubusercontent.com/59794238/119545633-003f6780-bdce-11eb-8202-9440e78ce1a3.PNG" width="30%"></img>  
1) Make Feed-Forward Network for each time step.
2) Apply recurrence relation to pass the past memory. (Connect Hidden States)

3. Recurrent Neural Network (RNN)  
<img src="https://user-images.githubusercontent.com/59794238/119545699-0fbeb080-bdce-11eb-8f2c-d0c01753b63a.PNG" width="40%"></img> <img src="https://user-images.githubusercontent.com/59794238/119545881-409ee580-bdce-11eb-8d0f-13e84dea0adc.PNG" width="30%"></img>  
- Unfolding RNNs : Re-use the **same weight matrices** at every time step. Sum all losses.  
<img src="https://user-images.githubusercontent.com/59794238/119547156-b9527180-bdcf-11eb-8b80-9ae94408ce7f.png" width="40%"></img>  
- Use Call function to make a forward pass (tf.kears.layers.simpleRNN(rnn_units))  
<img src="https://user-images.githubusercontent.com/59794238/119545939-501e2e80-bdce-11eb-87c3-09050f084218.PNG" width="40%"></img>  

4. Sequence Modeling: Design Criteria
- Word prediction example: Encoding Language for a Neural Network (word -> vector)
1. Handle Variable Sequence Lengths
- Feed forward networks are not able to do this becuase they have inputs of fixed dimensionality.
- But in RNN, differences in sequence lengths are just differences in the number of time steps.
2. Long-Term Dependencies
- We need information from the distant past to accurately predict the correct word.
3. Capture Differences in Sequence Order : 순서가 중요함

5. Backpropagation Through Time (BPTT)  
<img src="https://user-images.githubusercontent.com/59794238/119546000-62986800-bdce-11eb-86a0-f9cf40892929.PNG" width="40%"></img>  
각 timestep에 대해 backpropagation을 한 후 최근->처음으로 pass
- Gradient Issues : During backpropagation, we repeat gradient computation! (W_hh backpropagation 반복)
	- Many values > 1: exploding gradients -> Gradient clipping (threshold 설정)
	- Many values < 1: vanishing gradients. 최종값은 Bias에 의지하고 Long-Term Dependencies 고려 X.
		- Use ReLU : x>0에서 미분값이 항상 1. Prevents gradient shrinking.
		- Parameter Initialization: Initialize weights, biases to zero.
		- Gated Cells: Use a more **complex recurrent unit with gates** (LSTM)

6. Long Short Term Memory (LSTM) Networks  
<img src="https://user-images.githubusercontent.com/59794238/119546752-3af5cf80-bdcf-11eb-963b-facdb1167ce9.PNG" width="40%"></img>  
- Information is added or removed through structures called gates.
- Forget -> Store -> Update -> Output (Sigmoid gate로 조절)

7. RNN Applications
	1) Music Generation : Generate new composition.
	2) Sentiment Classification : Use cross entropy about the output of sequence of words.
	3) Machine Translation : Vector로 바꾸는 Encoder, 다른 언어로 바꾸는 Decoder 사용  
		<img src="https://user-images.githubusercontent.com/59794238/119546781-434e0a80-bdcf-11eb-9988-53d308b439b8.PNG" width="30%"></img>  
		- 데이터 양이 많아 발생하는 문제를 Attention을 사용하여 해결.  
		<img src="https://user-images.githubusercontent.com/59794238/119546809-4c3edc00-bdcf-11eb-9ba2-45d716e4af5c.PNG" width="30%"></img>  
