## 编译 TensorFlow 动态库 和 官网自带例子

```bash
# get code
git clone https://github.com/tensorflow/tensorflow
mv tensorflow tf1.10
cd tf1.10
git branch -r
git checkout r1.10 

# compile
bazel clean
./configure

# 编译动态链接库
bazel build --config=opt //tensorflow:libtensorflow_cc.so # cpu

ls bazel-bin/tensorflow/
    libtensorflow_cc.so     libtensorflow_framework.so

# 编译官网自带的例子
bazel build -c opt //tensorflow/cc:tutorials_example_trainer

export TF_CPP_MIN_LOG_LEVEL=2 
bazel-bin/tensorflow/cc/tutorials_example_trainer

ls bazel-genfiles/
    external/  tensorflow/



```

或者
```bash
cd tensorflow/contrib/makefile
./build_all_linux.sh
```

# 编译自己程序

https://blog.csdn.net/jmh1996/article/details/73197337

```bash
cd tf1.10
mkdir -p tensorflow/cc/example/
vi tensorflow/cc/example/example.cc
vi tensorflow/cc/example/BUILD

bazel build -c opt //tensorflow/cc/example:example
bazel-bin/tensorflow/cc/example/example
```

