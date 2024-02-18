package neural_network

import (
	"NN/pkg/matrix"
	"log"
	"math"
)

type NeuralNetwork struct {
	numLayers int             // количество слоев
	sizes     []int           // количество нейронов в каждом слое
	biases    []matrix.Matrix // смещения
	weights   []matrix.Matrix // веса
}

func NewNeuralNetwork(szs []int) NeuralNetwork {
	numLayers := len(szs)
	sizes := make([]int, numLayers)
	copy(sizes, szs)

	biases := make([]matrix.Matrix, numLayers-1)
	weights := make([]matrix.Matrix, numLayers-1)

	// генерируем веса и смещения
	for i := 0; i < numLayers-1; i++ {
		biases[i] = matrix.RandMatrix(sizes[i+1], 1)
		weights[i] = matrix.RandMatrix(sizes[i+1], sizes[i])
	}

	return NeuralNetwork{
		numLayers: numLayers,
		sizes:     sizes,
		biases:    biases,
		weights:   weights,
	}
}

// прямое распространение
func (n NeuralNetwork) feedforward(x matrix.Matrix) matrix.Matrix {
	if x.GetRows() != n.sizes[0] || x.GetColumns() != 1 {
		log.Fatalf("Dimension of the input matrix must be %d * %d", n.sizes[0], 1)
	}

	for i := 0; i < n.numLayers-1; i++ {

		// z = sigmoid(w * a + b)
		x = n.weights[i].Dot(x).Add(n.biases[i]).ForEach(sigmoid)
	}

	return x
}

func sigmoid(z float64) float64 {
	return 1. / (1. + math.Exp(-z))
}
