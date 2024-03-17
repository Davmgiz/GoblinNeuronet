package main

import (
	"fmt"
	"log"
	"time"

	"github.com/Davmgiz/GoblinNeuronet/pkg/data_frame"
	goblinet "github.com/Davmgiz/GoblinNeuronet/pkg/neural_network"
)

// Запуск main.go допустим только из корневой директории проекта.
// Обучение производится на датасете MNIST,
// ссылка на набор данных: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download.
// Самого датасета в директории нет.
func main() {

	// Засекаем время.
	startTime := time.Now()

	// Считываем данные для обучения.
	dfTrain, err := data_frame.ReadCSV("mnist_train.csv", 60000)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Data train read")

	// Считываем данные для теста.
	dfTest, err := data_frame.ReadCSV("mnist_test.csv", 10000)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Data test read")

	// Кодируем цифру вектором,
	// так как цифр всего 10, то и длина вектора будет 10.
	err = dfTrain.Num2Vec(10)
	if err != nil {
		log.Fatal(err)
	}

	// Создаем нейронную сеть.
	// Так как MNIST имеет изображение длинной 28 на 28, то всё изображение содержит 28 * 28 = 784 пикселя.
	// Получается входной слой будет содержать 784 нейрона.
	// Последний выходной слой будет содержать 10 нейронов, так как всего 10 цифр.
	// В остальных (скрытых) слоях количество нейронов выбирается в результате экспериментов.
	// Так же будем использовать сигмоидальную функцию активации.
	nn := goblinet.NewNeuralNetwork([]int{784, 30, 10}, goblinet.Sigmoid{})

	// Обучаем нейронную сеть с помощью стохастического градиентного спуска.
	// Передаем указатель на данные на которых будет происходить обучение,
	// количество эпох,
	// размер мини батча,
	// скорость обучения,
	// коэффициент регуляризации,
	// устанавливаем печать текущей эпохи,
	// устанавливаем флаг, чтобы при обучении была нормализация данных.
	nn.Sgd(&dfTrain, 1, 10, 0.01, 5, true, true)

	// Печатаем метрику оценки качества (количество правильно угаданных цифр).
	fmt.Println("Accuracy: ", nn.Accuracy(dfTest))

	// Записываем параметры для нейронной сети в файл net_par.txt,
	// чтобы использовать их в следующие разы и снова не обучать нейронную сеть.
	nn.WriteToFile("net_par.txt")

	// Выводим время выполнения.
	elapsedTime := time.Since(startTime)
	fmt.Printf("Время выполнения: %s\n", elapsedTime)
}
