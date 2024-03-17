/*
Package neural_network предоставляет инструментарий для создания и обучения полносвязных нейронных сетей.
Он включает в себя стандартный метод обучения, такой как стохастический градиентный спуск, а также и регуляризации L2.
Данный пакет может быть использован в разнообразных задач классификации.

Основные компоненты пакета включают структуру NeuralNetwork, которая предоставляет основу для создания нейронной сети,
а также набор вспомогательных функций и методов для её настройки и обучения.
Пакет разработан с целью обеспечения гибкости и удобства использования,
позволяя пользователю легко экспериментировать с различными архитектурами сети и параметрами обучения.
*/
package neural_network

import (
	"fmt"

	"github.com/Davmgiz/GoblinNeuronet/pkg/data_frame"
	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// NeuralNetwork представляет структуру полносвязной нейронной сети.
// Функция активации одна на все слои нейронной сети.
// Всегда используется регуляризация L2.
type NeuralNetwork struct {
	numLayers         int             // Количество слоев
	sizes             []int           // Количество нейронов в каждом слое
	biases            []matrix.Matrix // Смещения
	weights           []matrix.Matrix // Веса
	actFunc           activationFunc  // Функция активации
	norm              matrix.Matrix   // Вектор максимальных значений по модулю по всем признакам наблюдений
	haveNormalization bool            // Включена ли нормализация или нет
}

// NewNeuralNetwork возвращает нейронную сеть (структуру NeuralNetwork)
// Принимает слайс из количества нейронов в каждом слое соответственно и
// интерфейс activationFunc который представляет из себя функцию активации.
// Функция вызывает панику, если элементы слайса sizes не положительны.
func NewNeuralNetwork(sizes []int, actFunc activationFunc) NeuralNetwork {
	//
	numLayers := len(sizes)
	for i := 0; i < numLayers; i++ {
		if sizes[i] <= 0 {
			panic("Incorrect slice sizes")
		}
	}

	biases := make([]matrix.Matrix, numLayers-1)
	weights := make([]matrix.Matrix, numLayers-1)

	// генерируем веса и смещения с математическим ожиданием 0 и стандартным отклонением 0.01
	for i := 0; i < numLayers-1; i++ {
		biases[i] = matrix.RandMatrix(sizes[i+1], 1)
		weights[i] = matrix.RandMatrix(sizes[i+1], sizes[i])
	}

	return NeuralNetwork{
		numLayers:         numLayers,
		sizes:             sizes,
		biases:            biases,
		weights:           weights,
		actFunc:           actFunc,
		haveNormalization: false,
	}
}

// feedforward возвращает матрицу (структуру Matrix) результат нейронной сети (структуры NeuralNetwork)
// Метод реализует прямое распространение.
// Метод вызывает панику, если количество строк исходного вектора (матрицы x) не ровняется
// количеству входных нейронов нейронной сети.
func (nn *NeuralNetwork) feedforward(x matrix.Matrix) matrix.Matrix {
	if x.GetRows() != nn.sizes[0] || x.GetColumns() != 1 {
		panic(fmt.Sprintf("Dimension of the input matrix must be %d * %d", nn.sizes[0], 1))
	}

	// если включена нормализация то приводим к нормализованному виду вектор признаков
	if nn.haveNormalization {
		x.HadamardProductInPlace(nn.norm)
	}

	// вычисление производится рекуррентно
	for i := 0; i < nn.numLayers-1; i++ {

		// a = activation_function(w * a + b)

		x = nn.weights[i].Dot(x)
		x.AddInPlace(nn.biases[i])
		x.ForEachInPlace(nn.actFunc.fnc)
	}

	return x
}

// Sgd реализует стохастический градиентный спуск.
// dataTrain датафрейм на котором производим обучение,
// epochs количество эпох,
// miniBatchSize размер minibatch,
// eta скорость обучения,
// lmd коэффициент регуляризации L2,
// isPrintEpoch если true то печатает текущую эпоху,
// haveNormalization если true то выполняет нормализацию.
// Предупреждение: метод может вызвать панику,
// если ошибка для выходного слоя не определена в методе нейронной сети (структуре NeuralNetwork) backProp,
// поскольку внутренне вызывается функция backProp, требующая явного определения
// ошибки для выходного слоя для каждой активационной функции.
// Функция вызывает панику, если возникла ошибка при нормализации дата сета.
func (nn *NeuralNetwork) Sgd(dataTrain *data_frame.DataFrame, epochs int, miniBatchSize int, eta float64, lmd float64, isPrintEpoch, haveNormalization bool) {

	// если включена нормализация, то выполняем нормализацию и сохраняем
	// вектор максимальных значений по модулю по всем признакам наблюдений
	if haveNormalization {
		norm, err := dataTrain.Normalization()
		if err != nil {
			panic(err)
		}

		// устанавливаем параметры
		nn.norm = norm
		nn.haveNormalization = true
	}

	for epoch := 0; epoch < epochs; epoch++ {

		// вывод текущей эпохи
		if isPrintEpoch {
			fmt.Println("epoch :", epoch+1)
		}

		// перемешивание датафрейма
		dataTrain.Shuffle()

		// разбиваем датафрейм на части длины которых равны miniBatchSize и на основе каждой такой части обновляем веса
		for i := 0; i < dataTrain.Lenght(); i += miniBatchSize {
			nn.updateMiniBatch(dataTrain.CopyMiniBatch(i, miniBatchSize), eta, lmd, len(dataTrain.Data))
		}

		// обрабатываем остаток
		if dataTrain.Lenght()%miniBatchSize != 0 {
			start := (dataTrain.Lenght() / miniBatchSize) * miniBatchSize
			length := dataTrain.Lenght() - start
			nn.updateMiniBatch(dataTrain.CopyMiniBatch(start, length), eta, lmd, len(dataTrain.Data))
		}
	}
}

// updateMiniBatch обновляет веса и смещения исходной нейронной сети.
// eta скорость обучения,
// lmd коэффициент регуляризации L2,
// lenDf размер датафрейма на котором производится обучение.
func (nn *NeuralNetwork) updateMiniBatch(miniBatch data_frame.DataFrame, eta float64, lmd float64, lenDf int) {

	// создаем массивы из нулевых матриц такой же размерности что и веса и смещения нейронной сети,
	// чтобы считать градиент по miniBatch
	nablaWeights := matrix.Zeros(&nn.weights)
	nablaBiases := matrix.Zeros(&nn.biases)

	// используем обратное распространение чтобы найти градиент по каждому наблюдению из miniBatch
	for i := 0; i < miniBatch.Lenght(); i++ {
		deltaNablaBiases, deltaNablaWeights := nn.backProp(miniBatch.GetRow(i))

		// считаем градиент для весов по miniBatch
		for j := 0; j < len(*nablaWeights); j++ {
			(*nablaWeights)[j].AddInPlace((*deltaNablaWeights)[j])
		}

		// считаем градиент для смещений по miniBatch
		for j := 0; j < len(*nablaBiases); j++ {
			(*nablaBiases)[j].AddInPlace((*deltaNablaBiases)[j])
		}
	}

	// считаем коэффициенты для регуляризации L2
	k := eta / float64(miniBatch.Lenght())
	wk := 1. - eta*(lmd/float64(lenDf))

	// обновляем веса
	for j := 0; j < len(nn.weights); j++ {
		nn.weights[j].ForEachInPlace(func(w float64) float64 {
			return w * wk
		})

		(*nablaWeights)[j].ForEachInPlace(func(w float64) float64 {
			return w * k
		})

		nn.weights[j].SubInPlace((*nablaWeights)[j])
	}

	// обновляем смещения
	for j := 0; j < len(nn.biases); j++ {
		(*nablaBiases)[j].ForEachInPlace(func(w float64) float64 {
			return w * k
		})

		nn.biases[j].SubInPlace((*nablaBiases)[j])
	}

}

// backProp возвращает градиенты по одному наблюдению.
// Реализует обратное распространение.
// x вектор признаков наблюдения,
// y целевая переменная наблюдения представленная виде матрицы.
func (nn *NeuralNetwork) backProp(x matrix.Matrix, y matrix.Matrix) (*[]matrix.Matrix, *[]matrix.Matrix) {

	// храним градиенты для одного наблюдения
	nablaWeights := matrix.Zeros(&nn.weights)
	nablaBiases := matrix.Zeros(&nn.biases)

	activation := x

	// храним активации слоев
	activations := make([]matrix.Matrix, nn.numLayers)
	activations[0] = activation

	// храним взвешенную сумму слоев
	zs := make([]matrix.Matrix, nn.numLayers-1)

	// заполняем activations и zs
	for i := 0; i < nn.numLayers-1; i++ {

		z := nn.weights[i].Dot(activation)
		z.AddInPlace(nn.biases[i])

		zs[i] = z

		activation = z.ForEach(nn.actFunc.fnc)

		activations[i+1] = activation
	}

	// ошибка для выходного слоя
	delta := nn.actFunc.getDelta(zs[len(zs)-1], activations[len(activations)-1], y)

	// находим градиенты в выходном слое
	(*nablaBiases)[len(*nablaBiases)-1] = delta

	(*nablaWeights)[len(*nablaWeights)-1] = delta.Dot(activations[len(activations)-2].T())

	// находим градиенты в остальных весах
	for i := 2; i < nn.numLayers; i++ {
		z := zs[len(zs)-i]

		delta = nn.weights[len(nn.weights)-i+1].T().Dot(delta)
		delta.HadamardProductInPlace(z.ForEach(nn.actFunc.prime))

		(*nablaBiases)[len(*nablaBiases)-i] = delta

		(*nablaWeights)[len(*nablaWeights)-i] = delta.Dot(activations[len(activations)-i-1].T())

	}

	return nablaBiases, nablaWeights
}
