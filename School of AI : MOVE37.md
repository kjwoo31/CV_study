# [RL Course](https://www.edwith.org/move37/joinLectures/25196)

## Lecture 1 | Markov Decision Processes
  
1. **Markov Chain Model**: 현재 진행 중인 상태는 딱 한 단계 전의 상태에만 의존한다는 전제(**Markov Property**) 하에, 두 단계의 관계인 Transition Matrix를 찾는 것    
- 마르코브 결정 과정(MDP): Markov Property를 갖는 전이 확률로 의사결정하는 것. **(S, A, T, r, γ)**   
state(S): 현재 상태, action(A): 가능한 모든 결정들 (목표를 이루기 위한 일련의 행동들을 모아 policy라 함.), model(T): 어떤 상태에서의 행동의 영향을 알려줌, reward(r): 행동에 대한 반응, γ: 현재의 보상과 미래의 보상 사이의 상대적인 중요도.    
- Policy  
Deterministic policy: 상태 값에 따라 행동 결정  
Stochastic policy: 상태 s에서 확률적으로 a라는 행동을 취함 (a=π(s))  

2. **Bellman Equation**: 현재 state의 value는 즉각 보상에 할인율을 곱한 뒤따르는 보상을 더한 것과 같다.   
Bellman Expectation Equation : V(s)=R(s)+γV(s')  
Bellman Optimality Equation : V(s)=max_a(R(s,a)+γV(s'))  
유도)  
    1) state-value function : 누적 보상을 고려  
    ![image](https://user-images.githubusercontent.com/59794238/102096666-4e882c00-3e68-11eb-9036-6438c1378c9c.png)  
    2) action-value function (Q-function): action을 추가로 고려  
    ![image](https://user-images.githubusercontent.com/59794238/102097903-d6bb0100-3e69-11eb-9529-e9d7e9d5bae8.png)  
    3) 정책 고려  
    ![image](https://user-images.githubusercontent.com/59794238/102097230-fef63000-3e68-11eb-8dac-668fa7aa65fc.png)  
    ![image](https://user-images.githubusercontent.com/59794238/102098145-2ef20300-3e6a-11eb-82f2-e86112a085dc.png)  
최종 상태에서 recursive하게 reward를 계산하여 value를 표시하고, value가 높아지는 방향으로 이동  

과제) **OpenAI Gym**  
- step 함수 : 각 행동이 환경에 어떤 영향을 미치는지 알려줌. (observation, reward, done, info = env.step(action))   
observation: 환경에 대한 관찰 결과(카메라 위치, 로봇 각도, 속도, 보드 상태 등), reward: 이전의 행동으로 인한 보상, done: 환경 reset해야 하는지 알려줌, info: 디버깅, 진단을 위한 정보  
- Space: 가능한 행동들과 관찰값의 형식을 알려줌 (env.action_space, env.observation_space)  
Descrete: 고정된 범위의 음이 아닌 수만 허용. action의 경우, 0 혹은 1만 가능.  
Box: n차원의 숫자 배열. env.observation_space.high, .low로 Box의 경계값 확인 가능.  
- [Gym 환경 리스트](https://gym.openai.com/envs/#classic_control) (print(envs.registry.all()))

3. **[Google Dopamin](https://github.com/llSourcell/Google_Dopamine_LIVE/blob/master/Google_Dopamine_(LIVE)%20(1).ipynb)**: OpenAI Gym과 Tensorflow를 합친 프레임워크.   
- Deep Q: Atari 게임들을 마스터할 수 있는 딥마인드의 알고리즘이다.  
replay memory: 모든 상태, 행동, 보상을 학습하는 동안 큰 배열에 저장, 대규모 분산 학습, 분산 모델링 방법의 3가지로 구성됨  

</br>

## Lecture 2 | Dynamic Programming
1. **[Dynamic Programming](https://github.com/dennybritz/reinforcement-learning/tree/master/DP/)**: 완벽한 모델이 주어졌을 때, 최적 정책을 구하기 위해 사용할 수 있는 알고리즘의 집합   
- 큰 문제를 작은 문제로 쪼갠 후, 재귀적으로 풀어나가며 해결. 계산량이 많고 완벽한 모델 필요함    
- Policy Iteration: 정책 평가와 정책 향상 반복. 임의의 정책에서 시작하고 반복하여 각 칸의 가치표, 정책을 갱신. 정책과 가치표가 변동이 없을 때까지 반복한다.  
- Value Iteration: 가치표를 만든 후 정책 만듦. 각 칸의 value를 정확하게 계산하여 각 상태의 최적 정책을 구함. 더 많은 연산 비용이 든다. (최적화 원칙: 정책이 상태 s에서 최적 가치를 가지면 다음 상태도 s로부터 출발했을 때에만 최적 가치를 갖는다.)  

기타) [Kaggle 해보기](https://www.kaggle.com/c/two-sigma-financial-modeling/overview/description)

</br>

## Lecture 3 | Monte Carlo Methods
1. **Model Free RL**: action의 결과를 알 수 없을 때(transition model과 reward function 중 하나라도 모를 때)  
- **Monte Carlo Method**: 무작위로 샘플을 뽑아서 수치적인 결과를 얻는 방법 (경험적, Dynamic Planning은 가능한 모든 상태를 여러 번 순환.)   
fist visit Monte Carlo: 하나의 에피소드에서 같은 상태를 여러 번 방문할 경우, 첫 번째로 방문한 상태만 고려함.  
-  **Q Learning** (Q:action-state 쌍의 quality): Q(s,a)는 상태 s에서 행동 a를 취했을 때 value를 다음 상태의 Q 값을 이용(벨만 방정식)해 나타낸 것이다. 반복을 통해 state-action 쌍에 대한 reward가 최대인 Q 함수를 찾아 선택한다. (Value function의 max가 되야 하는 부분이 Q(= R(s,a)+γV(s')))  


2. **Exploration vs Exploitation**: RL할 때, 탐색을 많이 할 지, 기존 지식을 많이 활용할지 딜레마를 겪는다. 이 탐색과 이용의 균형을 찾을 때 가장 높은 보상을 얻는다.  
- ε-greedy: ε은 무작위한 행동을 취하는 확률로 ε(탐색)를 증가시키는 전략이다. 반대로, 기존 지식을 최대한 활용(이용)하는 전략을 greedy라고 한다.  

기타) TPU: 신경망의 행렬 연산을 위해 만들어진 반도체, 텐서 연산 ([활용](https://www.edwith.org/move37/lecture/59796/))  

</br>

## Lecture 4 | Model Free Learning
1. **시간차 학습**
- 파를로프 실험: 자극에 대한 조건 반응 (호루라기 소리를 들을 때마다 개는 음식을 기대하고 침을 흘림), 동물이 기대한 값과 실제로 받은 값의 차이를 예측 오차라고 한다.
=> 연상학습이론의 기초를 형성함.
- 강화학습에서 state = 자극, state를 통해 잠재적 reward를 예측한다. 실제 반환값에 학습률 α만큼 이전의 가치를 더함 (V(s)=(1-α)V(s)+α[r+γV(s')])
2. on policy, off policy
- on policy: 최근 policy에 기반하여 현재의 최선의 행동을 취하면 수렴
- off policy: 무작위로 행동을 취하여 기록된 데이터로 학습하여 수렴.  ex. Q learning

</br>

## Lecture 5 | RL in Continuous Spaces
1. **Augmented Random Search**(ARS)  
2. **Kalman Filter**: 간접적이고 불확실한 측정치로 시스템의 상태를 추정할 때 사용하는 추정 알고리즘이다.   
- 이전 상태의 측정치로부터 현재 상태를 예측한 값과 현재 상태의 측정치를 종합하여 현재 상태를 추정.  
- 자율주행차에서 센서 융합할 때, 각 센서에 대해 칼만 필터를 적용하여 예측 정확도 향상  

기타) [연속 행동 공간 관련 알고리즘](https://www.edwith.org/move37/lecture/59807/)

</br>

## Lecture 6 | Deep Reinforcement Learning
1. **Deep Q Learning**: Q table의 state-action 연관성을 신경망을 사용해 근사하는 방법.
- state-action 쌍이 너무 많아지면 이들을 저장하고 Q 함수를 **근사**하는 것이 어려워짐. (일반적인 함수로 표현 어려움) 따라서, 모든 state-action 쌍을 저장했던 방식을 **신경망**으로 state-action 쌍에서 Q 값을 근사하는 방법으로 변경. 
2. **Dueling DQN(DDQN)**
- DQN에서는 계산 흐름이 하나였지만 DDQN에서는 value, advantage(특정 행동이 다른 행동에 비해 얼마나 좋은지 알려주는 함수)를 각각 계산하여 마지막 층에서 합친다. (Q(s,a)=A(s,a)+V(s))  
3. 신경망
- 순방향 신경망: 다층 퍼셉트론, 네크워크를 연결된 노드의 그래프로 표현  
- 순환 신경망: RNN, hidden state에 과거 상태를 저장하고 입력값과 hidden state를 종합하여 출력  
- BPTT(Backpropagation Through Time): 경사도가 너무 크거나 작아질 수 있다. 이를 해결하기 위해 LSTM(hidden state뿐만 아니라 셀 상태도 저장), Gated Recurrent Unit 활용  

</br>

## Lecture 7 | Policy Based Methods
1. Meta Learning: 높은 단계의 AI가 낮은 단계의 AI 혹은 그들 여러 개를 최적화하는 것
- Neuroevolution: 다윈의 진화론처럼 여러 개의 알고리즘을 형성, 점수를 매겨 높은 점수의 멤버만 남기고 그 멤버의 특징을 갖는 다른 멤버를 번식한다. 이 과정을 반복하면 환경에 적합한 알고리즘만 남게 된다. ex. 이미지 분류 AmoebaNet
2. 정책 검색 알고리즘

</br>

## Lecture 8 | Policy Gradient Methods


