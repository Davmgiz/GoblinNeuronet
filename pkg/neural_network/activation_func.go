package neural_network

// файл содержит функции активации

import (
	"math"

	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// activationFunc интерфейс для функций активации.
// Функция активации должна иметь еще производную.
type activationFunc interface {
	fnc(z float64) float64                        // сама функция активации
	prime(z float64) float64                      // производная функции активации
	getName() string                              // имя функции активации, которое используется при записи параметров нейронной сети
	getDelta(z, a, y matrix.Matrix) matrix.Matrix // ошибка на выходном слое
}

// nameToActFunc возвращает интерфейс activationFunc.
// Функция возвращает функцию активации соответствующую принимаемому имени.
// Функция вызывает панику если переданному имени не соответствует никакая функция активации.
func nameToActFunc(name string) activationFunc {
	name2ActFunc := make(map[string]activationFunc)

	name2ActFunc["Sigmoid"] = Sigmoid{}

	actFunc, ok := name2ActFunc[name]
	if !ok {
		panic("Activation function not defined")
	}
	return actFunc
}

// Sigmoid структура имплементирующая интерфейс activationFunc.
type Sigmoid struct {
}

// fnc возвращает результат функции активации.
func (s Sigmoid) fnc(z float64) float64 {
	return 1. / (1. + math.Exp(-z))
}

// prime возвращает результат производной функции активации.
func (s Sigmoid) prime(z float64) float64 {
	return s.fnc(z) * (1. - s.fnc(z))
}

// getName возвращает имя функции активации.
func (s Sigmoid) getName() string {
	return "Sigmoid"
}

// getDelta возвращает ошибку на выходном слое.
func (s Sigmoid) getDelta(z, a, y matrix.Matrix) matrix.Matrix {
	return a.Sub(y)
}
