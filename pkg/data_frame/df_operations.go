package data_frame

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// init инициализирует генератор псевдослучайных чисел
func init() {
	rand.Seed(time.Now().UnixNano())
}

// Shuffle перемешивает датафрейм, изменяя его.
func (df *DataFrame) Shuffle() {
	rand.Shuffle(len(df.Data), func(i int, j int) {
		df.Data[i], df.Data[j] = df.Data[j], df.Data[i]
	})
}

// CopyMiniBatch возвращает датафрейм (структуру DataFrame) который является подвыборкой исходного датафрейма.
// miniBatch содержит все наблюдения исходного датафрейма которые принадлежат отрезку [i, i+miniBatchSize-1].
func (df *DataFrame) CopyMiniBatch(i int, miniBatchSize int) DataFrame {
	return DataFrame{
		Data: df.Data[i : i+miniBatchSize],
	}
}

// Num2Vec кодирует в вектор матрицу размерности n на 1 (структуру Matrix) целевую переменную (структуру Matrix) датафрейма.
// Метод возвращает ошибку, если целевая переменная не может быть представлена целым числом.
func (df *DataFrame) Num2Vec(n int) error {
	for i := 0; i < len(df.Data); i++ {
		target, err := matrix.Matrix2Vector(df.Data[i].y, n)
		if err != nil {
			return fmt.Errorf("incorrect target in data frame %v", err)
		}

		df.Data[i].y = target
	}
	return nil

}

// Normalization возвращает ошибку и вектор максимальных значений по модулю по всем признакам наблюдений
// Метод возвращает ошибку, если датафрейм пустой.
// Метод выполняет нормализацию датафрейма.
// По каждому признаку находит его максимальный модуль по значению и
// делит каждый элемент признака на соответствующий ему максимальный модуль.
func (df *DataFrame) Normalization() (matrix.Matrix, error) {
	if len(df.Data) == 0 {
		return matrix.Matrix{}, errors.New("empty data frame to do normalization")
	}

	// Храним максимальные модули.
	maxAll := make([]float64, df.Data[0].x.GetRows())

	for i := 0; i < df.Lenght(); i++ {
		for j := 0; j < df.Data[0].x.GetRows(); j++ {
			maxAll[j] = maxF64(math.Abs(df.Data[i].x.GetIJ(j, 0)), maxAll[j])
		}
	}

	// Если максимальный модуль равен 0, то подменяем его на 1,
	// чтобы не происходило деление на 0.
	for i := 0; i < len(maxAll); i++ {
		if maxAll[i] == 0 {
			maxAll[i] = 1
		}
	}

	norm := matrix.Zero(df.Data[0].x.GetRows(), 1)
	norm.Slice2Matrix(maxAll)

	for i := 0; i < df.Lenght(); i++ {
		df.Data[i].x.HadamardProductInPlace(norm)
	}

	return norm, nil
}

// maxF64 вспомогательная функция для функции Normalization.
func maxF64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
