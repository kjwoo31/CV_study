# [DeepLearning Course](http://introtodeeplearning.com/?fbclid=IwAR2wCGZ_DrzzdpU2OLZHmXjZy9H14NfEXwat0d9L4IdbN76LHkgoHXqlidc)

## Lecture 1 | Introduction to Deep Learning
  
1. Why Deep Learning?  
- Deep Learning: Extract patterns from data using neural networks
- Why now? : Big data, HW(GPU, parallelizable), SW

2. Perceptron (single neuron): The structural building block of deep learning  
<img src="https://user-images.githubusercontent.com/59794238/119324261-9a15f000-bcba-11eb-9478-e9584f64efa4.png" width="50%"></img>  
1. input, weight의 dot product
2. add bias
3. Non-Linearity g : activation function, linear decision -> arbitrarily complex functions
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
	<img src="https://user-images.githubusercontent.com/59794238/119325451-d5fd8500-bcbb-11eb-9fe8-65023d8e4c84.png" width="50%"></img>  
	- Early Stopping: Stop training before we have a chance to overfit.  
	<img src="https://user-images.githubusercontent.com/59794238/119325046-6f786700-bcbb-11eb-87be-8bf70afe3c49.png" width="50%"></img>  


</br>
