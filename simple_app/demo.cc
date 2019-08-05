#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "local_min_cnn.h"
#include "tiny_conv_micro_features_model_data.h"
#include <iostream>
using namespace std;
int main(int argc, char *argv[])
{
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter *error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model *model =
      tflite::GetModel(local_min_cnn_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  cout << "Model loaded!, Size:" << local_min_cnn_tflite_len << ", Version:" << model->version();
  auto desc = model->description();

  cout << endl
       << "Model description: " << desc->c_str() << endl;
  // This pulls in all the operation implementations we need.
  tflite::ops::micro::AllOpsResolver resolver;
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  const long tensor_arena_size = 4 * 1024 * 1024;

#if TRUE
  // use the heap
  uint8_t *tensor_arena;
  tensor_arena = (uint8_t *)malloc(sizeof(uint8_t) * tensor_arena_size);
  cout << "Tensor arena allocated on heap" << endl;
#else
  // use the stack like micro_speech
  uint8_t tensor_arena[tensor_arena_size];
  cout << "Tensor arena allocated on stack" << endl;
#endif

  tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                                 tensor_arena_size);

  cout << "Allocator working (ish): Size=" << tensor_allocator.GetDataSize() << endl;
  // Build an interpreter to run the model with.

  tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                       error_reporter);

  cout << "Interpreter loaded!" << endl;

  // print some info about the model
  for (int i = 0; i < interpreter.inputs_size(); i++)
  {
    cout << "Input #" << i << ": ";
    auto c_input = interpreter.input(i);
    for (int j = 0; j < c_input->dims->size; j++)
    {
      cout << c_input->dims->data[j] << ", ";
    }
    cout << "Type:";
    if (c_input->type == kTfLiteUInt8)
      cout << "UInt8";
    if (c_input->type == kTfLiteFloat32)
      cout << "Float32";
    cout << ", " << c_input->type;
    cout << endl;
  }

  for (int i = 0; i < interpreter.outputs_size(); i++)
  {
    cout << "Output #" << i << ": ";
    auto c_output = interpreter.output(i);
    for (int j = 0; j < c_output->dims->size; j++)
    {
      cout << c_output->dims->data[j] << ", ";
    }
    cout << "Type:";
    if (c_output->type == kTfLiteUInt8)
      cout << "UInt8";
    if (c_output->type == kTfLiteFloat32)
      cout << "Float32";
    cout << ", " << c_output->type;
    cout << endl;
  }

  // Get information about the memory area to use for the model's input.
  TfLiteTensor *model_input = interpreter.input(0);
  if ((model_input->dims->size != 4) ||
      (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 180) ||
      (model_input->dims->data[2] != 3) ||
      (model_input->type != kTfLiteUInt8))
  {
    error_reporter->Report("Bad input tensor parameters in model");
    //return 1;
  }
  // set to some random data
  model_input->data.uint8 = (uint8_t *)malloc(sizeof(uint8_t) * 49);
  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  TfLiteTensor *output = interpreter.output(0);
  cout << "Model Results: ";
  for (int j = 0; j < output->dims->data[1]; j++)
  {
    cout << "[" << j << "]=" << output->data.uint8[j] / 255.0 << ", ";
  }
  cout << endl;
  return 0;
}
