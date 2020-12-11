# [RL Course](https://www.edwith.org/move37/joinLectures/25196)

## Lecture 1 | Markov Decision Processes
  
1. **Markov Chain Model**: 현재 진행 중인 상태는 딱 한 단계 전의 상태에만 의존한다는 전제(마르코브 성질) 하에, 두 단계의 관계인 Transition Matrix를 찾는 것    
- 마르코브 결정 과정(MDP): **(S, A, T, r, γ)**   
state(S): 현재 상태, action(A): 가능한 모든 결정들 (목표를 이루기 위한 일련의 행동들을 모아 policy라 함.), model(T): 어떤 상태에서의 행동의 영향을 알려줌, reward(r): 행동에 대한 반응, γ: 현재의 보상과 미래의 보상 사이의 상대적인 중요도.    
2. **Bellman Equation**: 현재의 state에서 최적의 reward을 얻는 action을 하도록 하는 것   
식: V(s)=max_a(R(s,a)+rV(s'))  
최종 상태에서 recursive하게 reward를 계산하여 value가 높은 방향으로 이동  

</br>

## Lecture 2 | Dynamic Programming
l
