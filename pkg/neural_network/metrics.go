package neural_network

// файл содержит метрики

import (
	"github.com/Davmgiz/GoblinNeuronet/pkg/data_frame"
	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// Accuracy возвращает accuracy в процентах (количество правильно угаданных предсказаний)
// Метод принимает тестовый датасет для подсчета.
func (nn *NeuralNetwork) Accuracy(dataTest data_frame.DataFrame) float64 {
	//mp := make(map[int]int)
	cnt := 0
	for i := 0; i < len(dataTest.Data); i++ {

		// находим предсказание
		pred := matrix.Vec2Num(nn.feedforward(dataTest.Data[i].GetX()))
		//mp[pred]++

		//сравниваем предсказание со значением по факту.
		y := int(matrix.Num(dataTest.Data[i].GetY()))
		if y == pred {
			cnt++
		}
	}

	//fmt.Println(mp)

	return (float64(cnt) / float64(len(dataTest.Data))) * 100
}
