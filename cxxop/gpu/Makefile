# https://github.com/MatteoRagni/tf.ZeroOut.gpu

TF_INC = `python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB = `python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"`

NSYNC_INC = $(TF_INC)/external/nsync/public


CXX = g++
CFLAGS = -std=c++11 -shared
LFLAGS = -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$(TF_INC) -I$(NSYNC_INC) \
 -L$(TF_LIB) -ltensorflow_framework 


NVCC = /usr/local/cuda/bin/nvcc
NVCC_CFLAGS = -std=c++11 -c -I$(TF_INC) -I$(NSYNC_INC) 
NVCC_LFLAGS = -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC


all: example_op.so


example_op.cu.o: kernel_example.cu.cc example.h
	$(NVCC) $(NVCC_CFLAGS) -c $< $(NVCC_LFLAGS) -o $@
# -D_MWAITXINTRIN_H_INCLUDED to avoid errors from mwaitxintrin.h


example_op.so: example_op.cu.o example.h example_op.cc kernel_example.cc
	$(CXX) $(CFLAGS) $^ $(LFLAGS) -lcudart -o $@ -O2

