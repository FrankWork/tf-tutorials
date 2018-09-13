
TF_INC := $(TF_ROOT)/include
TF_LIB := $(TF_ROOT)/lib

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
