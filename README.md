# GoblinNeuronet

Эта библиотека предоставляет простой инструментарий для создания и обучения полносвязных нейронных сетей.
Он включает в себя стандартный метод обучения, такой как стохастический градиентный спуск, а также и регуляризации L2.
Данный пакет может быть использован в разнообразных задач классификации.

## Особенности

- Встроенные операции с матрицами для эффективных вычислений.
- Утилиты для работы с данными в формате CSV.
- Примеры использования для быстрого старта.

## Установка

Для начала работы с библиотекой, установите её с помощью `go get`:

```shell
go get github.com/Davmgiz/GoblinNeuronet
```

## Быстрый старт

Пример для обучения и сохранения параметров нейронной сети.
Обучение производится на датасете [MNIST в формате CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download).

```go
package main

import (
	goblinet "github.com/Davmgiz/GoblinNeuronet/pkg/neural_network"
    "github.com/Davmgiz/GoblinNeuronet/pkg/data_frame"
	"fmt"
	"log"
)

func main() {

	// Считываем данные для обучения.
	dfTrain, err := data_frame.ReadCSV("data/mnist_train.csv", 60000)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Data train read")

	// Считываем данные для теста.
	dfTest, err := data_frame.ReadCSV("data/mnist_test.csv", 10000)
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
	nn := neural_network.NewNeuralNetwork([]int{784, 30, 10}, neural_network.Sigmoid{})

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
}
```

Пример для чтения параметров и использования нейронной сети.

```go
package main

import (
	goblinet "github.com/Davmgiz/GoblinNeuronet/pkg/neural_network"
    "github.com/Davmgiz/GoblinNeuronet/pkg/data_frame"
	"fmt"
	"log"
)

func main() {

	// Считываем данные для теста.
	dfTest, err := data_frame.ReadCSV("data/mnist_test.csv", 10000)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Data test read")

    // Считываем параметры 
    nn, err := goblinet.ReadFromFile("net_par.txt")
    if err != nil{
        log.Fatal(err)
    }
	
	// Печатаем метрику оценки качества (количество правильно угаданных цифр).
	fmt.Println("Accuracy: ", nn.Accuracy(dfTest))
}
```

## Документация

Для ознакомления с полной документацией перейдите к файлам внутри пакетов data_frame, matrix, и neural_network.

## Лицензия

Этот проект распространяется под лицензией [MIT](https://opensource.org/licenses/MIT).

## Контакты

Если у вас есть вопросы или предложения, пожалуйста, свяжитесь со мной через [почту](mailto:suhanov173@gmail.com).