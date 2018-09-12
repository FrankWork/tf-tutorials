## 编译 TensorFlow 动态库

```bash
git clone https://github.com/tensorflow/tensorflow
mv tensorflow tf1.10
cd tf1.10
git branch -r
git checkout r1.10 

bazel clean
./configure
bazel build --config=opt //tensorflow:libtensorflow_cc.so # cpu
ls bazel-bin/tensorflow/
    libtensorflow_cc.so     libtensorflow_framework.so

ls bazel-genfiles/
    external/  tensorflow/
```

或者
```bash
cd tensorflow/contrib/makefile
./build_all_linux.sh
```

# 编译主程序

```bash
export TF_ROOT=tf1.10
make
```