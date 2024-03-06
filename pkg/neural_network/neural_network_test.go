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

	dfTrain, err := data_frame.ReadCSV("../../data/test_nn_data_train.csv", 10)
	if err != nil {
		t.Errorf("%s", err)
	}

	dfTrain.Num2Vec(2)

	nn.Sgd(&dfTrain, 200, 7, 0.5, 5)

	expectedWeights := []matrix.Matrix{
		matrix.DataToMatrix([][]float64{
			{0.9772355463324838, 1.9186819212727124, 2.797943950590206, 3.58862646005447},
			{5.996990295467862, 7.07147046830393, 8.189844829396291, 9.36001208731443},
			{9.981961107908273, 11.082523372434508, 12.233070511737322, 13.43967615980169},
		}),
		matrix.DataToMatrix([][]float64{
			{0.31467449146920123, 2.914716559656199, 4.043534718987852},
			{1.0050654028559818, -2.792604634853659, -3.793559399203883},
		}),
	}

	expectedBiases := []matrix.Matrix{
		matrix.DataToMatrix([][]float64{
			{-2.5542994742831575},
			{-0.18720505118996272},
			{0.017451861901720732},
		}),
		matrix.DataToMatrix([][]float64{
			{-2.7592867865128587},
			{2.4949598389355745},
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
