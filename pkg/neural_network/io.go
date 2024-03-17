package neural_network

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// Write записывает параметры нейронной сети (структуры NeuralNetwork) в объект реализующий интерфейс io.Writer
// и возвращает ошибку, если она возникла при записи.
// Метод записывает в формате:
// сначала число слоев,
// с новой строки перечисление через пробел количество нейронов в каждом слое соответственно,
// с новой строки имя функции активации,
// с новой строки 1, если есть нормализация, далее если есть нормализация, то с новой строки вектор из абсолютных максимумов,
// если 0, то нормализации нет, далее если нормализации нет, то идет с новой строки пустая строка
// далее идут веса,
// и наконец смещения
func (nn *NeuralNetwork) Write(writer io.Writer) error {
	_, err := fmt.Fprintf(writer, "%d\n", nn.numLayers)
	if err != nil {
		return err
	}

	for i := 0; i < nn.numLayers; i++ {
		_, err := fmt.Fprintf(writer, "%d ", nn.sizes[i])
		if err != nil {
			return err
		}
	}

	_, err = fmt.Fprintf(writer, "\n")
	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(writer, nn.actFunc.getName()+"\n")
	if err != nil {
		return err
	}

	if nn.haveNormalization {
		_, err = fmt.Fprintf(writer, "1\n")
		if err != nil {
			return err
		}

		norm := make([]matrix.Matrix, 1)
		norm[0] = nn.norm
		err = matrix.WriteMatrixes(writer, norm)
		if err != nil {
			return err
		}
	} else {
		_, err = fmt.Fprintf(writer, "0\n\n")
		if err != nil {
			return err
		}
	}

	err = matrix.WriteMatrixes(writer, nn.weights)
	if err != nil {
		return err
	}

	err = matrix.WriteMatrixes(writer, nn.biases)
	if err != nil {
		return err
	}

	return nil
}

// Read считывает параметры нейронной сети (структуры NeuralNetwork) из обЪекта реализующего интерфейс io.Writer.
// Возвращает нейронную сеть (структуру NeuralNetwork) ошибку, если она возникла при чтении.
// Функция читает параметры только в формате:
// сначала число слоев,
// с новой строки перечисление через пробел количество нейронов в каждом слое соответственно,
// с новой строки имя функции активации,
// с новой строки 1, если есть нормализация, далее если есть нормализация, то с новой строки вектор из абсолютных максимумов,
// если 0, то нормализации нет, далее если нормализации нет, то идет с новой строки пустая строка
// далее идут веса,
// и наконец смещения
func Read(reader io.Reader) (NeuralNetwork, error) {

	scanner := bufio.NewScanner(reader)

	if !scanner.Scan() {
		return NeuralNetwork{}, fmt.Errorf("unexpected end of file while reading neural network parameters")
	}

	line := strings.Fields(scanner.Text())
	if len(line) != 1 {
		return NeuralNetwork{}, fmt.Errorf("incorrect number of layers")
	}

	numLayers, err := strconv.Atoi(line[0])
	if err != nil {
		return NeuralNetwork{}, err
	}

	if numLayers <= 0 {
		return NeuralNetwork{}, fmt.Errorf("incorrect number of layers")
	}

	if !scanner.Scan() {
		return NeuralNetwork{}, fmt.Errorf("unexpected end of file while reading neural network parameters")
	}

	line = strings.Fields(scanner.Text())
	if len(line) != numLayers {
		return NeuralNetwork{}, fmt.Errorf("incorrect number of layers")
	}

	sizes := make([]int, numLayers)
	for i := 0; i < numLayers; i++ {
		num, err := strconv.Atoi(line[i])
		if err != nil {
			return NeuralNetwork{}, err
		}

		if num <= 0 {
			return NeuralNetwork{}, fmt.Errorf("number of neuron must be positive")
		}

		sizes[i] = num
	}

	if !scanner.Scan() {
		return NeuralNetwork{}, fmt.Errorf("unexpected end of file while reading neural network parameters")
	}

	line = strings.Fields(scanner.Text())
	if len(line) != 1 {
		return NeuralNetwork{}, fmt.Errorf("incorrect name of activation function")
	}

	nameFuncAct := line[0]
	actFunc := nameToActFunc(nameFuncAct)

	if !scanner.Scan() {
		return NeuralNetwork{}, fmt.Errorf("unexpected end of file while reading neural network parameters")
	}

	line = strings.Fields(scanner.Text())
	if len(line) != 1 {
		return NeuralNetwork{}, fmt.Errorf("incorrect flag of normalization")
	}

	haveNormalization := true

	norm := make([]matrix.Matrix, len(line))

	if line[0] == "1" {
		norm, err = matrix.ReadMatrixes(scanner)
		if err != nil {
			return NeuralNetwork{}, err
		}
	} else {
		haveNormalization = false

		if !scanner.Scan() {
			return NeuralNetwork{}, fmt.Errorf("unexpected end of file while reading neural network parameters")
		}
	}

	weights, err := matrix.ReadMatrixes(scanner)
	if err != nil {
		return NeuralNetwork{}, err
	}

	biases, err := matrix.ReadMatrixes(scanner)
	if err != nil {
		return NeuralNetwork{}, err
	}

	return NeuralNetwork{
		numLayers:         numLayers,
		sizes:             sizes,
		biases:            biases,
		weights:           weights,
		actFunc:           actFunc,
		norm:              norm[0],
		haveNormalization: haveNormalization,
	}, nil
}

// WriteToFile записывает параметры нейронной сети (структуры NeuralNetwork) в файл и в случае неудачи возвращает ошибку.
func (nn *NeuralNetwork) WriteToFile(filename string) error {

	// открытие файла для записи, его создание, если не существует, и очистка содержимого перед записью
	file, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		return err
	}
	defer file.Close()

	if err := nn.Write(file); err != nil {
		return err
	}

	return nil
}

// ReadFromFile читает параметры нейронной сети (структуры NeuralNetwork) из файла.
// Возвращает нейронную сеть в случае успешного прочтения и ошибку иначе.
func ReadFromFile(filename string) (NeuralNetwork, error) {
	file, err := os.Open(filename)
	if err != nil {
		return NeuralNetwork{}, err
	}

	defer file.Close()

	nn, err := Read(file)
	if err != nil {
		return NeuralNetwork{}, err
	}

	return nn, nil
}
