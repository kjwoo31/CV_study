# [RL Course](https://www.edwith.org/move37/joinLectures/25196)

## Lecture 1 | Markov Decision Processes
  
1. **Markov Chain Model**: 현재 진행 중인 상태는 딱 한 단계 전의 상태에만 의존한다는 전제(마르코브 성질) 하에, 두 단계의 관계인 Transition Matrix를 찾는 것    
- 마르코브 결정 과정(MDP): **(S, A, T, r, γ)**   
state(S): 현재 상태, action(A): 가능한 모든 결정들 (목표를 이루기 위한 일련의 행동들을 모아 policy라 함.), model(T): 어떤 상태에서의 행동의 영향을 알려줌, reward(r): 행동에 대한 반응, γ: 현재의 보상과 미래의 보상 사이의 상대적인 중요도.    
- Policy  
Deterministic policy: 상태 값에 따라 행동 결정  
Stochastic policy: 상태 s에서 확률적으로 a라는 행동을 취함 (a=𝝿(s))  

2. **Bellman Equation**: 현재의 state에서 최적의 reward을 얻는 action을 하도록 하는 것   
식: V(s)=max_a(R(s,a)+rV(s'))  
최종 상태에서 recursive하게 reward를 계산하여 value가 높은 방향으로 이동  

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

[Kaggle 해보기](https://www.kaggle.com/c/two-sigma-financial-modeling/overview/description)

</br>

## Lecture 3 | Monte Carlo Methods
