package neuralnetwork

import(
	//"NN/pkg/matrix"
)



type NeuralNetwork struct{
	numLayers int
	sizes []int
	biases []*Matrix
	weights []*Matrix
}

func NewNeuralNetwork(sizes []int){
	numLayers := len(sizes)

	biases := make([]*Matrix, numLayers - 1)
	weights := make([]*Matrix, numLayers - 1)

	for i := 0; i < numLayers - 1; i++{
		biases[i] = 
	}




}
