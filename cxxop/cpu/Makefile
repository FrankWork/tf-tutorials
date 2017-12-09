TF_INC = `python -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB = `python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"`

NSYNC_INC = $(TF_INC)/external/nsync/public

CXX = g++
CFLAGS = -std=c++11 -shared
LFLAGS = -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$(TF_INC) -I$(NSYNC_INC) \
 -L$(TF_LIB) -ltensorflow_framework 

all: grl_op.so

grl_op.so: grl_op.cc grl_kernel.cc
	$(CXX) $(CFLAGS) $^ $(LFLAGS) -o $@ -O2
