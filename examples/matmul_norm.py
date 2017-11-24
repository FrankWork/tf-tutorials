import tensorflow as tf

r = 2
b = 3
h = 4
buf = []
for i in range(r):
  i += 1
  tensor = tf.reshape(i*tf.range(b*h), [b, h])
  buf.append(tensor)

H = tf.stack(buf, axis=1) # (b,r,h)
S = tf.reshape(3*tf.range(b*h), [b, h])

S3 = tf.expand_dims(S, axis=2)# (b,h,1)
loss1 = tf.reduce_sum(
              tf.square(tf.matmul(H, S3)))

H0 = H[:, 0, :]
loss2 = tf.reduce_sum(
  tf.square(tf.matmul(H0, S, transpose_a=True))
)
H1 = H[:, 1, :]
loss2 += tf.reduce_sum(
  tf.square(tf.matmul(H1, S, transpose_a=True))
)

a = tf.matmul(H, S3)
b = tf.matmul(H0, S, transpose_a=True)
with tf.Session() as sess:
  # fetch = [loss1, loss2]
  fetch = [a,b]
  for v in sess.run(fetch):
    print(v)
    print('-'*40)