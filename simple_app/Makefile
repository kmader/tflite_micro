CC=g++
XXD=python2.7 xxd.py
CFLAGS=-g -DNDEBUG --std=c++11 -g -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_DISABLE_X86_NEON
INCS=-I. -I../tensorflow/tensorflow/lite/experimental/micro/tools/make/downloads/ -I../tensorflow/tensorflow/lite/experimental/micro/tools/make/downloads/gemmlowp -I../tensorflow/tensorflow/lite/experimental/micro/tools/make/downloads/flatbuffers/include -I../tensorflow
DEPS=local_min_cnn.h
OBJ=local_min_cnn.o demo.o tiny_conv_micro_features_model_data.o
LIBS=../tensorflow/tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a
.DEFAULT_GOAL:=test

local_min_cnn.cc: local_min_cnn.tflite
	$(XXD) local_min_cnn.tflite > local_min_cnn.cc

%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCS)

demo: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

test: demo
	./demo

.PHONY: clean

clean:
	rm local_min_cnn.cc
	rm -f *.o *~
	rm demo
