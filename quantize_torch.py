from onnxruntime.quantization import quantize_dynamic, QuantType

model_input_path = "./onnx_model/model.onnx"
model_output_path = "./onnx_model/model_quantized.onnx"

quantize_dynamic(
    model_input_path,
    model_output_path,
    weight_type=QuantType.QInt8
)
