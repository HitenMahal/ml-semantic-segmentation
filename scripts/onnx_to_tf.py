import onnx

from onnx_tf.backend import prepare

onnx_model = onnx.load("../app/ariel-segmentation-unet.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("ariel-segmentation-unet.tf")  # export the model
