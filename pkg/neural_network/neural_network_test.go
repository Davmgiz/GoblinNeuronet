package neural_network

import (
	"NN/pkg/data_frame"
	"NN/pkg/matrix"
	"testing"
)

func TestSvg(t *testing.T) {
	var w11 = matrix.DataToMatrix([][]float64{
		{1., 2., 3., 4.},
		{6., 7., 8., 9.},
		{10., 11., 12., 13.},
	})

	var w22 = matrix.DataToMatrix([][]float64{
		{1., 2., 3.},
		{4., 5., 6.},
	})

	var weights1 = []matrix.Matrix{w11, w22}

	var b11 = matrix.DataToMatrix([][]float64{
		{1.},
		{2.},
		{3.},
	})

	var b22 = matrix.DataToMatrix([][]float64{
		{4.},
		{5.},
	})

	var biases1 = []matrix.Matrix{b11, b22}

	nn := NewNeuralNetwork([]int{4, 3, 2})

	nn.weights = weights1
	nn.biases = biases1

	dfTrain, err := data_frame.ReadCSV("../../data/neural_network_test/test_svg_data_train.csv", 10)
	if err != nil {
		t.Errorf("%s", err)
	}

	dfTrain.Num2Vec(2)

	nn.Sgd(&dfTrain, 200, 7, 0.5, 5, false)

	expectedWeights := []matrix.Matrix{
		matrix.DataToMatrix([][]float64{
			{0.02027589430624981, 0.04868647662544333, 0.08524127141164027, 0.12994980311890036},
			{0.018727510350233248, 0.047860480550955, 0.08740814915519154, 0.13737975471596922},
			{0.014842514211849882, 0.04427084350904176, 0.08829645355982517, 0.1469308100324496},
		}),
		matrix.DataToMatrix([][]float64{
			{0.20795099992526045, 0.2022461429344551, 0.18160957929341925},
			{-0.20795106358776766, -0.20224621431196982, -0.1816096607362056},
		}),
	}

	expectedBiases := []matrix.Matrix{
		matrix.DataToMatrix([][]float64{
			{0.7339216611448136},
			{1.3735887507269593},
			{2.3372439767062936},
		}),
		matrix.DataToMatrix([][]float64{
			{-0.0862577982062465},
			{0.08625824315587409},
		}),
	}

	if len(nn.biases) != len(expectedBiases) || len(nn.weights) != len(expectedWeights) {
		t.Errorf("Different dimension expected weights and result weights or expected biases and result biases")
	}

	for i := 0; i < len(nn.weights); i++ {
		if !matrix.IsMatrixesEqual(nn.weights[i], expectedWeights[i]) {
			t.Errorf("Dont equal expected weights and result weights")
		}
	}

	for i := 0; i < len(nn.biases); i++ {
		if !matrix.IsMatrixesEqual(nn.weights[i], expectedWeights[i]) {
			t.Errorf("Dont equal expected biases and result biases")
		}
	}

}
