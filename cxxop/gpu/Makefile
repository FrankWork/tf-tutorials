# https://github.com/MatteoRagni/tf.ZeroOut.gpu

TF_INC = `python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB = `python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"`

NSYNC_INC = $(TF_INC)/external/nsync/public


CXX = g++
CFLAGS = -std=c++11 -shared
LFLAGS = -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$(TF_INC) -I$(NSYNC_INC) \
 -L$(TF_LIB) -ltensorflow_framework 


NVCC = /usr/local/cuda/bin/nvcc
NVCC_CFLAGS = -std=c++11 -I$(TF_INC)  -c
NVCC_LFLAGS = -x cu -Xcompiler -fPIC


all: example_op.so


example_op_cu.o: kernel_example.cu.cc
	$(NVCC) $(NVCC_CFLAGS) -c $^ $(NVCC_LFLAGS) -o $@

example_op.so: example_op_cu.o example.h example_op.cc kernel_example.cc
	$(CXX) $(CFLAGS) $^ $(LFLAGS) -lcuda -o $@ -O2

