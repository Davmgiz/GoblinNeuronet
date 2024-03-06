package main

import (
	"NN/pkg/data_frame"
	"NN/pkg/neural_network"
	"fmt"
	"time"
)

// запуск main.go допустим только из корневой директории проекта
func main() {

	startTime := time.Now()

	dfTrain, err := data_frame.ReadCSV("data/mnist_train.csv", 60000)
	if err != nil {
		fmt.Print(err)
	}
	fmt.Println("Data train read")

	dfTest, err := data_frame.ReadCSV("data/mnist_test.csv", 10000)
	if err != nil {
		fmt.Print(err)
	}
	fmt.Println("Data test read")

	dfTrain.Num2Vec(10)

	nn := neural_network.NewNeuralNetwork([]int{784, 30, 10})
	nn.Sgd(&dfTrain, 1, 10, 0.01, 5)

	fmt.Println("Accuracy: ", nn.Accuracy(dfTest))

	elapsedTime := time.Since(startTime)
	fmt.Printf("Время выполнения: %s\n", elapsedTime)

}
