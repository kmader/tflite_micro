# Building Demo App

## Setup Dependencies
```
../tensorflow/tensorflow/lite/experimental/micro/tools/make/download_dependencies.sh
```
## Build Static TFLiteMicro library
Must be built in tensorflow directory
```
cd ../tensorflow && make -f tensorflow/lite/experimental/micro/tools/make/Makefile test
```

## Compile/Build

Avoid too many strange rules being added
```sh
make -r test
```

## Making New Models
```
conda env create -f ../binder/environment.yml
conda activate tflite
python run_model.py
```


## Debugging
Since it is compiled using g++ / clang you need to debug with lldb

```
lldb demo
run
```

Then use `bt` to backtrace if necessary
