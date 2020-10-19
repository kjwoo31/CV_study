import tensorflow as tf

#1. graph 생성
x_train = [1,2,3]
y_train = [1,2,3]

# tf에서 사용하는 variable
W = tf.Variable(tf.random_normal([1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name= 'bias')

# hypothesis과 cost
hypothesis = x_train*W+b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# GradientDescent 사용하여 cost 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# 2. sess.run, update
sess=tf.Session()

#변수 initialize
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))