package neural_network

import (
	"testing"

	"github.com/Davmgiz/GoblinNeuronet/pkg/data_frame"
	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// TestSvg проверяет правильно ли обучилась нейронная сеть (структура NeuralNetwork) с помощью
// фиксированного и правильного результата.
// Функция предназначена для быстрых изменений методов обучения и проверки корректности результата.
// Пока что функция может проверить единственную функцию активации сигмоиду.
func TestSvg(t *testing.T) {
	var w1 = matrix.DataToMatrix([][]float64{
		{1., 2., 3., 4.},
		{6., 7., 8., 9.},
		{10., 11., 12., 13.},
	})

	var w2 = matrix.DataToMatrix([][]float64{
		{1., 2., 3.},
		{4., 5., 6.},
	})

	var weights = []matrix.Matrix{w1, w2}

	var b1 = matrix.DataToMatrix([][]float64{
		{1.},
		{2.},
		{3.},
	})

	var b2 = matrix.DataToMatrix([][]float64{
		{4.},
		{5.},
	})

	var biases = []matrix.Matrix{b1, b2}

	nn := NewNeuralNetwork([]int{4, 3, 2}, Sigmoid{})

	nn.weights = weights
	nn.biases = biases

	// тестовый датасет
	dfTrain, err := data_frame.ReadCSV("../../data/neural_network_test/test_svg_data_train.csv", 10)
	if err != nil {
		t.Errorf("%s", err)
	}

	dfTrain.Num2Vec(2)

	nn.Sgd(&dfTrain, 200, 7, 0.5, 5, false, false)

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

	// проверка весов и смещений

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

// TestFeedForward проверяет выдает ли нейронная сеть (структура NeuralNetwork) фиксированный и правильный результат.
// Пока что функция может проверить единственную функцию активации сигмоиду.
func TestFeedForward(t *testing.T) {
	var w1 = matrix.DataToMatrix([][]float64{
		{0.001, 0.002, 0.003, 0.004},
		{0.006, 0.007, 0.008, 0.009},
		{0.001, 0.0011, 0.0012, 0.0013},
	})

	var w2 = matrix.DataToMatrix([][]float64{
		{0.001, 0.002, 0.003},
		{0.004, 0.005, 0.006},
	})

	var weights = []matrix.Matrix{w1, w2}

	var b1 = matrix.DataToMatrix([][]float64{
		{0.001},
		{0.002},
		{0.003},
	})

	var b2 = matrix.DataToMatrix([][]float64{
		{0.004},
		{0.005},
	})

	var biases = []matrix.Matrix{b1, b2}

	nn := NewNeuralNetwork([]int{4, 3, 2}, Sigmoid{})

	nn.weights = weights
	nn.biases = biases

	x := matrix.DataToMatrix([][]float64{
		{0.},
		{1.},
		{2.},
		{3.},
	})

	result := nn.feedforward(x)

	expected := matrix.DataToMatrix([][]float64{
		{0.50175975},
		{0.50315035},
	})

	if !matrix.IsMatrixesEqual(result, expected) {
		t.Errorf("Dont equal expected and result")
	}

	xResult := matrix.DataToMatrix([][]float64{
		{0.},
		{1.},
		{2.},
		{3.},
	})

	if !matrix.IsMatrixesEqual(xResult, x) {
		t.Errorf("It is forbidden to change original matrix")
	}
}
