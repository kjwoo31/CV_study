# [DeepLearning Course](http://introtodeeplearning.com/?fbclid=IwAR2wCGZ_DrzzdpU2OLZHmXjZy9H14NfEXwat0d9L4IdbN76LHkgoHXqlidc)

## Lecture 1 | Introduction to Deep Learning
  
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
