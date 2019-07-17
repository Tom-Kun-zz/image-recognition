from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd()
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( execution_path + "/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()

print("\nFirst Image should be a Tesla Model")
predictions, percentage_probabilities = prediction.predictImage(execution_path + "/examples/sample1.jpg", result_count=5)
for index in range(len(predictions)):
  print(predictions[index] , " : " , percentage_probabilities[index])

print("\nSecond Image should be a Lamborghini Model")
predictions, percentage_probabilities = prediction.predictImage(execution_path + "/examples/sample2.jpg", result_count=5)
for index in range(len(predictions)):
  print(predictions[index] , " : " , percentage_probabilities[index])

print("\nThird Image should be a Dacia Duster (SUV)")
predictions, percentage_probabilities = prediction.predictImage(execution_path + "/examples/sample3.jpg", result_count=5)
for index in range(len(predictions)):
  print(predictions[index] , " : " , percentage_probabilities[index])
