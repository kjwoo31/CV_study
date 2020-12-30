# [RL Course](https://www.edwith.org/move37/joinLectures/25196)

## Lecture 1 | Markov Decision Processes
  
1. **Markov Chain Model**: 현재 진행 중인 상태는 딱 한 단계 전의 상태에만 의존한다는 전제(**Markov Property**) 하에, 두 단계의 관계인 Transition Matrix를 찾는 것    
- 마르코브 결정 과정(MDP): Markov Property를 갖는 전이 확률로 의사결정하는 것. **(S, A, T, r, γ)**   
state(S): 현재 상태, action(A): 가능한 모든 결정들 (목표를 이루기 위한 일련의 행동들을 모아 policy라 함.), model(T): 어떤 상태에서의 행동의 영향을 알려줌, reward(r): 행동에 대한 반응, γ: 현재의 보상과 미래의 보상 사이의 상대적인 중요도.    
- Policy  
Deterministic policy: 상태 값에 따라 행동 결정  
Stochastic policy: 상태 s에서 확률적으로 a라는 행동을 취함 (a=π(s))  

2. **Bellman Equation**: 현재 state의 value는 즉각 보상에 할인율을 곱한 뒤따르는 보상을 더한 것과 같다.   
- Bellman Expectation Equation : V(s)=R(s)+γV(s')  
- Bellman Optimality Equation : V(s)=max_a(R(s,a)+γV(s'))  
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
replay memory(모든 상태, 행동, 보상을 학습하는 동안 큰 배열에 저장), 대규모 분산 학습, 분산 모델링 방법의 3가지로 구성됨  

</br>

## Lecture 2 | Dynamic Programming
1. **[Dynamic Programming](https://github.com/dennybritz/reinforcement-learning/tree/master/DP/)**: 완벽한 모델이 주어졌을 때, 최적 정책을 구하기 위해 사용할 수 있는 알고리즘의 집합   
- 큰 문제를 작은 문제로 쪼갠 후, 재귀적으로 풀어나가며 해결. 계산량이 많고 완벽한 모델 필요함    
- Policy Iteration: 정책 평가와 정책 향상 반복. 임의의 정책에서 시작하고 반복하여 Bellman Optimality Equation에 대입하였을 때 improvement가 발생하면 갱신. 정책과 가치표가 변동이 없을 때까지 반복한다.  
평가 방법)  
![image](https://user-images.githubusercontent.com/59794238/102102915-ae360580-3e6f-11eb-8fbd-8d34b7bb2a83.png)  
- Value Iteration: 모든 상태 s와 모든 행동 a에 대해 루프를 돈 후, Bellman Optimality Equation에 대입하여 최적의 policy를 찾는 과정. V(s)의 최대 변화값이 설정한 임계값보다 낮을 때까지 반복한다.  
![image](https://user-images.githubusercontent.com/59794238/102102876-a37b7080-3e6f-11eb-998e-8d67e0f6b6f6.png)  

기타) [Kaggle 해보기](https://www.kaggle.com/c/two-sigma-financial-modeling/overview/description)

</br>

## Lecture 3 | Monte Carlo Methods
1. **Model Free RL**: action의 결과를 알 수 없을 때(transition model과 reward function 중 하나라도 모를 때)  
2. **Monte Carlo Method**: 무작위로 state 집합 sample을 뽑아서 step별로 reward를 얻어 저장하고 terminal state에 도달하면 이를 바탕으로 state들의 value function을 찾는 방법  

경험적, Policy Iteration과 Value Iteration은 가능한 모든 상태를 순환한다는 점에서 다름   
식) 여러 episode를 진행하며 얻는 return 값들의 산술 평균은 true value func.으로 근사할 수 있다.  
![image](https://user-images.githubusercontent.com/59794238/102111638-040fab00-3e7a-11eb-929d-a1bac97cc9ad.png)  
- First visit MC, Every Visit MC  
First visit MC: 하나의 에피소드에서 같은 상태를 여러 번 방문할 경우, 첫 번째로 방문하였을 때의 return 값으로 계산  
Every visit MC: 반대로 방문할 때마다 바뀐 return값으로 계산  
-  **Q Learning** (Q:action-state 쌍의 quality): Q(s,a)는 상태 s에서 행동 a를 취했을 때 value를 다음 상태의 Q 값을 이용해 나타낸 것이다. 반복을 통해 state-action 쌍에 대한 총 reward가 최대인 Q 함수를 찾아 선택한다.  
식) 벨만 방정식의 V를 Q로, Q(s,a) ~ R(s,a) + γQ'(s',a')   
- **Exploration vs Exploitation**: RL할 때, 탐색을 많이 할 지, 기존 지식을 많이 활용할지 딜레마를 겪는다. 이 탐색과 이용의 균형을 찾을 때 가장 높은 보상을 얻는다.  
ε-greedy: ε은 무작위한 행동을 취하는 확률로 ε(탐색)를 증가시키는 전략  
greedy: 반대로, 기존 지식을 최대한 활용(이용)하는 전략   
- Monte Carlo Control의 특징
    1) state-value function은 model-based이기 때문에, action-value function (Q-function)을 사용하여(state뿐만 아니라 action 고려) **model의 정보가 function에** 다 담기게 한다.  
    2) ε-greedy는 local optimum에 빠지는 것을 방지한다.  
    3) Value Iteration에서는 value를 모두 estimate하고 난 후에 최적의 policy를 찾았는데, Q learning을 사용한 Monte Carlo Control에서는 매 episode마다 q를 estimate하고 policy improvment를 한다.  

기타) TPU: 신경망의 행렬 연산을 위해 만들어진 반도체, 텐서 연산 ([활용](https://www.edwith.org/move37/lecture/59796/))  

</br>

## Lecture 4 | Model Free Learning
1. 신경 과학
- 파를로프 실험: 자극에 대한 조건 반응 (호루라기 소리를 들을 때마다 개는 음식을 기대하고 침을 흘림), 동물이 기대한 값과 실제로 받은 값의 차이를 **예측 오차**라고 한다.
=> 연상학습이론의 기초를 형성함. 강화학습에서 state = 자극, state를 통해 잠재적 reward를 예측한다. 실제 반환값에 학습률 α만큼 이전의 가치를 더함 (V(s)=(1-α)V(s)+α[r+γV(s')])
- 레스콜라-바그너 모델: 연상학습이론의 가장 영향력 있는 모델. **예측과 실제 reward이 다를 때(예측 오차) 그 관계에 따라 연관성의 가중치를 높이거나 낮춘다.**    
레스콜라-바그너 모델은 **단일값**만 예측할 수 있는데, **세상의 불확실성**을 반영하기 위해 **여러 가중치에 대한 확률 분포**로 표현. (베이즈 정리 - 확률 분포, 칼만 필터 - 데이터를 살펴봄)  

2. **시간차 학습**: 레스콜라-바그너 모델을 확장하여, 예측 인자에 할인 인자를 추가함으로써 **시간적으로 가까운 보상의 가중치를 높인다.**  
- Kalman TD: 칼만 필터에 있는 불확실성을 포함하는 원리 추가. 가중치에 대한 단일값뿐만 아니라 평균과 공분산 행렬을 함께 예측.  
- 시간차를 반영하기 위해 학습률 α 추가. (V(s)=V(s)+α(r+γV(s')-V(s)), Q(s,a)=Q(s,a)+α(r+γargmax(Q(s'))-V(s)))  
- 시간의 경과에 따라 값이 줄어드는 adaptive epsilon, adaptive learning rate 사용.

3. on policy, off policy
- on policy: 정해진 policy로 행동. (Q Table에 따라 현재의 최선의 행동을 취한다.) 계속해서 모델 update하며 수렴.  
- off policy: 무작위로 행동을 취하여 기록된 데이터로 학습하여 수렴. -> Q Learning  

</br>

## Lecture 5 | RL in Continuous Spaces
1. Robotic manipulation - [참고](https://inmoov.fr/)
- Kinematics: 움직임의 원인을 고려하지 않은 채로 점, 물체 등의 움직임을 설명함.  
Inverse kinematics는 최적으로 특정 지점에 도달하기 위한 방법을 알려주는 방법으로, Forward Kinematics는 물체가 어떻게 이동하는지 알고 어떤 지점으로 갈 수 있는지 알려주는 방법이다.  
Inverse kinematics를 구현하는 방법으로, Gradient Descent 사용.  

2. **Augmented Random Search**(ARS, 2018): 무작위 노이즈를 추가하였을 때 보상의 변화를 관찰함으로써 노이즈의 영향을 파악하는 방법.  
- 유전 알고리즘의 한 종류
- 유한한 노이즈를 추가하는 Method of Finite Differences 사용. Gradient Descent과 기존 딥러닝 방법들에 비해 학습 속도가 매우 빠르다.

3. **Kalman Filter**: 간접적이고 불확실한 측정치로 시스템의 상태를 추정할 때 사용하는 추정 알고리즘이다.   
- 이전 상태의 측정치로부터 현재 상태를 예측한 값과 현재 상태의 측정치를 종합하여 현재 상태를 추정. (평균, 분산 등 데이터의 종합적 지표를 함께 활용)  
- 자율주행차에서 센서 융합할 때, 각 센서에 대해 칼만 필터를 적용하여 예측 정확도 향상  
![image](https://user-images.githubusercontent.com/59794238/103335644-eae13e00-4ab8-11eb-90cc-a76d72237513.png)  

기타) [연속 행동 공간 관련 알고리즘](https://www.edwith.org/move37/lecture/59807/)

</br>

## Lecture 6 | Deep Reinforcement Learning
1. **Deep Q Learning**: Q table의 state-action 연관성을 신경망을 사용해 근사하는 방법.
- state-action 쌍이 너무 많아지면 이들을 저장하고 Q 함수를 **근사**하는 것이 어려워짐. (일반적인 함수로 표현 어려움) 따라서, 모든 state-action 쌍을 저장했던 방식을 **신경망**으로 state-action 쌍에서 Q 값을 근사하는 방법으로 변경. 
- OpenAI Gym Wrapper: 정해진 행동을 실행하고 데이터 가공하는 역할. (함수랑 비슷)  
2. **Dueling DQN(DDQN)**
- DQN에서는 계산 흐름이 하나였지만 DDQN에서는 **value**(특정 상태에 있는 것이 얼마나 좋은지 알려주는 함수), **advantage**(특정 행동이 다른 행동에 비해 얼마나 좋은지 알려주는 함수)를 **각각 계산**하여 마지막 층에서 합친다. (Q(s,a)=A(s,a)+V(s))  
3. 신경망
- 순방향 신경망: 다층 퍼셉트론, 네크워크를 연결된 노드의 그래프로 표현  
- 순환 신경망: RNN, hidden state에 과거 상태를 저장하고 입력값과 hidden state를 종합하여 출력  
- BPTT(Backpropagation Through Time): 경사도가 너무 크거나 작아질 수 있다. 이를 해결하기 위해 LSTM(hidden state뿐만 아니라 셀 상태도 저장), Gated Recurrent Unit 활용  

</br>

## Lecture 7 | Policy Based Methods
1. Meta Learning: 높은 단계의 AI가 낮은 단계의 AI 혹은 그들 여러 개를 최적화하는 것  
- Neuroevolution: 다윈의 진화론처럼 여러 개의 알고리즘을 형성, 점수를 매겨 높은 점수의 멤버만 남기고 그 멤버의 특징을 갖는 다른 멤버를 번식한다. 이 과정을 반복하면 환경에 적합한 알고리즘만 남게 된다. ex. 이미지 분류 AmoebaNet  
2. **정책 검색 알고리즘**: 마르코프 결정 과정을 해결하기 위해 정책을 직접 학습하는 방법  
- 그 예로, 정책 경사 기법이 있다. **정책 경사 기법**은 **가치 함수를 학습**하고 이 가치를 사용해 주어진 마르코프 결정 과정을 위한 정책을 학습한다. 유사하게, **액터 크리틱 기법** 또한 정책 검색 알고리즘의 하위분류다. 정책 경사와 매우 비슷하지만 **추정한 가치 함수로 정책을 비교하여 분산을 줄인다.**  
- Q-러닝, Fitted-Q, LSTD-Q 등은 행동-검색 알고리즘. 각 상태에 대한 행동(혹은 가치 함수 근사를 다룰 때는 특성 벡터)을 최대화하여 최적 정책을 찾기 때문이다.  
- Policy Iteration과의 차이점: 정책 검색은 정책 공간에서 최적 정책을 찾는다.  
- 장점: 더 좋은 수렴성(가치 기반 방법보다 매 스텝 정책이 부드럽게 갱신), 가능한 행동이 무수히 많은 고차원 행동 공간에서 유용, 확률론적인 정책 학습 가능 (탐색/이용 균형과 perceptual aliasing 문제를 고려하지 않아도 된다)  
- 단점: 많은 상황에서 전역 최적점이 아닌 지역 최적점에 수렴  
</br>

## Lecture 8 | Policy Gradient Methods
1. **정책 경사 기법**: 정책을 직접 모델링, 최적화하는 방식  
- 무작위 정책으로 시작 -> 환경에서 몇 가지 행동 추출 -> 행동을 취할 확률을 바꾼다 -> 보상이 기대한 것보다 크면 행동을 취할 확률을 높인다, 더 낮다면 확률을 낮춘다  
- Q-Learning과의 차이점: 탐색/이용 균형을 고려할 필요가 없고 on policy다. 각 행동에 대한 기록인 Target Network가 필요 없다.  
- DQN보다 선호되고, 액터-크리틱 방법에서 액터로 흔히 사용

2. **REINFORCE algorithm** (몬테카를로 정책 경사): 1992년에 발표된 최초의 정책 경사 방법  
- 에피소드 표본에 Monte Carlo Method로 추정한 보상에 따라 정책 매개 변수(θ)를 갱신하는 방법  
- 보상 증가(REward Increment) = 음이 아닌 인자(Nonnegative Factor) x 오프셋 강화(Offset Reinforcement) x 특성 적격성(Characteristic Eligibility)  
![image](https://user-images.githubusercontent.com/59794238/103341179-88446e00-4ac9-11eb-8653-63f76a56d0c5.png)  

3. **Evolved Policy Gradients** (EPG, 진화된 정책 경사): 메타러닝 방법 - [논문](https://storage.googleapis.com/epg-blog-data/epg_2.pdf)
- 두 개의 최적화 루프로 구성. **내부 루프**는 확률적 정책 하강법 (SGD)을 사용하여 외부 루프가 제안한 손실 함수에 대해 **에이전트의 정책을 최적화**한다. **외부 루프**는 내부 루프에서 얻은 누적 보상을 평가하고 더 높은 누적 보상을 얻는 새로운 손실 함수를 제안하기 위해 **진화 전략 (ES)으로 손실 함수의 매개변수를 조정**한다.  
- 지역 최저점을 피할 수 있다.  
</br>

## Lecture 9 | Actor Critic Methods
1. Actor Critic Method  
- Actor: 현 환경 상태를 기반으로 행동을 함, Critic: 상태와 결과를 바탕으로 보상에 대한 중간 시그널을 만듦. (현 상태의 가치를 반환)  
- **Advantage Actor Critic Algorithm** (A2C): advantage는 해당 상태의 기댓값보다 높으면 보상을 준다. (A(S, A) = Q(S, A) - V(S)) Advantage로 정책의 경사를 조정한다.  
- **Asynchronous Advantage Actor Critic Algorithm** (A3C): 정책 경사는 현재 정책에서 얻어진 데이터를 기반으로 학습한다. 이때, 학습 데이터를 독립적이고 동일하게 분포(IID)시키기 위해 여러 개의 환경을 병렬적으로 실행하는 방법.  
- **Continuous Action Space**에서 Actor Critic을 사용하는 방법
    1) 각각의 연속 동작의 평균 μ와 표준 편차 σ를 출력해야 한다.  
    2) negative log function을 loss function으로 사용, entropy 함수 수정  

2. **Deep Deterministic Policy Gradients** (DDPG): Policy Gradient Method의 policy를 Actor로 하여, Actor Critic Method를 함께 사용하는 방법.  
- **Experience replay buffer**에 에이전트의 학습 중 경험을 저장하고 무작위적으로 sampling하는 방법, **Target Network**를 만들어 목표 오류 값을 정한 후 학습 알고리즘을 정규화하는 방법으로 개선함.  

3. **Proximal Policy Optimization** (PPO)  
- 기존 Actor Critic 문제점: Hyperparameter tuning, Outlier의 영향을 많이 받는다. -> 학습을 좀 더 매끄럽게 만들어 해결한 것이 PPO  
    1) Generalized Advantage Estimation (GAE): 보상의 손실을 줄여 학습을 보다 부드럽고 안정적이게 만듦. 분산을 최소화하는 Advantage와 반환값을 계산하는 방법.  
    2) Surrogate Policy Loss: P_t/P_t-1 * advantage로, 점진적인 policy 업데이트가 가능해짐.  
    3) Mini Batch Update: 경험값들이 무작위한 Mini Batch로 나뉨.  

기타) [추가 학습](https://www.edwith.org/move37/lecture/60015/)

</br>
