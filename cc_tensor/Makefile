
TF_ROOT := /data0/ad_dm1/shixi_zhihui3/tf1.10

TF_INC := $(TF_ROOT)/bazel-genfiles/
TF_LIB := $(TF_ROOT)/bazel-bin/tensorflow/

GCC := g++

OUTPUT := demo
SOURCES := $(wildcard *.cc)
CCFLAGS := -pthread -std=c++11 -g3 -I$(TF_INC) -L$(TF_LIB) -ltensorflow_framework


all: $(OUTPUT)

$(OUTPUT):
	$(GCC) $(SOURCES) $(CCFLAGS) -o $(OUTPUT) 

clean:
	rm $(OUTPUT)

.PHONY: all
