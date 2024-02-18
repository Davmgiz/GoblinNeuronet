package main

import (
	"NN/pkg/data_frame"
	"fmt"
	"time"
)

// запуск main.go допустим только из корневой директории проекта
func main() {
	startTime := time.Now()

	_, err := data_frame.ReadCSV("data/mnist_train.csv", 60000)
	if err != nil {
		fmt.Print(err)
	}
	elapsedTime := time.Since(startTime)
	fmt.Printf("Время выполнения: %s\n", elapsedTime)

}
