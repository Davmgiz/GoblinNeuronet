package neural_network

import (
	"NN/pkg/data_frame"
	"NN/pkg/matrix"
	"fmt"
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
func (nn NeuralNetwork) feedforward(x matrix.Matrix) matrix.Matrix {
	if x.GetRows() != nn.sizes[0] || x.GetColumns() != 1 {
		log.Fatalf("Dimension of the input matrix must be %d * %d", nn.sizes[0], 1)
	}

	for i := 0; i < nn.numLayers-1; i++ {

		// z = sigmoid(w * a + b)
		x = nn.weights[i].Dot(x)
		x.AddSelf(nn.biases[i])
		x.ForEachSelf(sigmoid)

	}

	return x
}

func (nn NeuralNetwork) Sgd(dataTrain *data_frame.DataFrame, epochs int, miniBatchSize int, eta float64, lmd float64, isPrintEpoch bool) {
	for epoch := 0; epoch < epochs; epoch++ {

		// вывод текущей эпохи
		if isPrintEpoch {
			fmt.Println("epoch :", epoch+1)
		}

		// перемешивание датафрейма
		dataTrain.Shuffle()

		// разбиваем датафрейм на части длины которых равны miniBatchSize и на основе каждой такой части обновляем веса
		for i := 0; i < dataTrain.Length(); i += miniBatchSize {
			nn.updateMiniBatch(dataTrain.CopyMiniBatch(i, miniBatchSize), eta, lmd, len(dataTrain.Data))
		}

		// обрабатываем остаток
		if dataTrain.Length()%miniBatchSize != 0 {
			start := (dataTrain.Length() / miniBatchSize) * miniBatchSize
			length := dataTrain.Length() - start
			nn.updateMiniBatch(dataTrain.CopyMiniBatch(start, length), eta, lmd, len(dataTrain.Data))
		}
	}
}

func (nn *NeuralNetwork) updateMiniBatch(miniBatch data_frame.DataFrame, eta float64, lmd float64, lenDf int) {
	nablaWeights := zeros(&nn.weights)
	nablaBiases := zeros(&nn.biases)

	for i := 0; i < miniBatch.Length(); i++ {
		deltaNablaBiases, deltaNablaWeights := nn.backProp(miniBatch.GetRow(i))

		for j := 0; j < len(*nablaWeights); j++ {
			//(*nablaWeights)[j] = (*nablaWeights)[j].Add((*deltaNablaWeights)[j])
			(*nablaWeights)[j].AddSelf((*deltaNablaWeights)[j])
		}

		for j := 0; j < len(*nablaBiases); j++ {
			//(*nablaBiases)[j] = (*nablaBiases)[j].Add((*deltaNablaBiases)[j])
			(*nablaBiases)[j].AddSelf((*deltaNablaBiases)[j])
		}

	}

	k := eta / float64(miniBatch.Length())
	wk := 1. - eta*(lmd/float64(lenDf))

	for j := 0; j < len(nn.weights); j++ {
		nn.weights[j].ForEachSelf(func(w float64) float64 {
			return w * wk
		})

		(*nablaWeights)[j].ForEachSelf(func(w float64) float64 {
			return w * k
		})

		//nn.weights[j] = nn.weights[j].Sub((*nablaWeights)[j])
		nn.weights[j].SubSelf((*nablaWeights)[j])
	}

	for j := 0; j < len(nn.biases); j++ {
		(*nablaBiases)[j].ForEachSelf(func(w float64) float64 {
			return w * k
		})

		//nn.biases[j] = nn.biases[j].Sub((*nablaBiases)[j])
		nn.biases[j].SubSelf((*nablaBiases)[j])
	}

}

func (nn *NeuralNetwork) costDerivative(act matrix.Matrix, y matrix.Matrix) matrix.Matrix {
	return act.Sub(y)
}

func (nn *NeuralNetwork) backProp(x matrix.Matrix, y matrix.Matrix) (*[]matrix.Matrix, *[]matrix.Matrix) {
	nablaWeights := zeros(&nn.weights)
	nablaBiases := zeros(&nn.biases)

	activation := x

	activations := make([]matrix.Matrix, nn.numLayers)
	activations[0] = activation

	zs := make([]matrix.Matrix, nn.numLayers-1)

	for i := 0; i < nn.numLayers-1; i++ {

		//z := nn.weights[i].Dot(activation).Add(nn.biases[i])
		z := nn.weights[i].Dot(activation)
		z.AddSelf(nn.biases[i])

		//zs = append(zs, z)
		zs[i] = z

		activation = z.ForEach(sigmoid)

		//activations = append(activations, activation)
		activations[i+1] = activation

	}

	delta := nn.costDerivative(activations[len(activations)-1], y)

	(*nablaBiases)[len(*nablaBiases)-1] = delta

	(*nablaWeights)[len(*nablaWeights)-1] = delta.Dot(activations[len(activations)-2].T())

	for i := 2; i < nn.numLayers; i++ {

		z := zs[len(zs)-i]

		//delta = nn.weights[len(nn.weights)-i+1].T().Dot(delta).HadamardProduct(z.ForEach(sigmoidPrime))
		delta = nn.weights[len(nn.weights)-i+1].T().Dot(delta)
		delta.HadamardProductSelf(z.ForEach(sigmoidPrime))

		(*nablaBiases)[len(*nablaBiases)-i] = delta

		(*nablaWeights)[len(*nablaWeights)-i] = delta.Dot(activations[len(activations)-i-1].T())

	}

	return nablaBiases, nablaWeights
}

func (nn *NeuralNetwork) Accuracy(dataTest data_frame.DataFrame) float64 {
	//mp := make(map[int]int)
	cnt := 0
	for i := 0; i < len(dataTest.Data); i++ {
		pred := matrix.Vec2Dig(nn.feedforward(dataTest.Data[i].GetX()))
		//mp[pred]++
		y := int(matrix.Num(dataTest.Data[i].GetY()))
		if y == pred {
			cnt++
		}
	}

	//fmt.Println(mp)

	return (float64(cnt) / float64(len(dataTest.Data))) * 100
}

func zeros(slc *[]matrix.Matrix) *[]matrix.Matrix {
	res := make([]matrix.Matrix, len(*slc))

	for i := 0; i < len(*slc); i++ {
		res[i] = matrix.Zero((*slc)[i].GetRows(), (*slc)[i].GetColumns())
	}
	return &res
}

func sigmoid(z float64) float64 {
	return 1. / (1. + math.Exp(-z))
}

func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}
