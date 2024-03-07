package data_frame

import (
	"NN/pkg/matrix"
	"log"
	"math"
	"math/rand"
	"time"
)

// Структура одного наблюдения.
type rowDataFrame struct {
	x matrix.Matrix // вектор из признаков
	y matrix.Matrix // целевая переменная
}

// Структура датафрейма
type DataFrame struct {
	Data []*rowDataFrame // массив из указателей на структуры наблюдений
}

// инициализация генератора псевдослучайных чисел
func init() {
	rand.Seed(time.Now().UnixNano())
}

func (row *rowDataFrame) GetX() matrix.Matrix {
	return row.x
}

func (row *rowDataFrame) GetY() matrix.Matrix {
	return row.y
}

// перемешивание датафрейма
func (df *DataFrame) Shuffle() {
	rand.Shuffle(len(df.Data), func(i int, j int) {
		df.Data[i], df.Data[j] = df.Data[j], df.Data[i]
	})
}

func (df *DataFrame) Length() int {
	return len(df.Data)
}

func (df *DataFrame) CopyMiniBatch(i int, miniBatchSize int) DataFrame {
	return DataFrame{
		Data: df.Data[i : i+miniBatchSize],
	}
}

// x, y
func (df *DataFrame) GetRow(i int) (matrix.Matrix, matrix.Matrix) {
	return df.Data[i].x, df.Data[i].y
}

func (df *DataFrame) Num2Vec(n int) {
	for i := 0; i < len(df.Data); i++ {
		target, err := matrix.Matrix2Vector(df.Data[i].y, n)
		if err != nil {
			log.Fatal("Incorrect target in data frame", err)
		}

		df.Data[i].y = target
	}

}

func (df *DataFrame) Normalization() {
	if len(df.Data) == 0 {
		log.Fatal("Empty data frame to do normalization")
	}

	maxAll := make([]float64, df.Data[0].x.GetRows())

	for i := 0; i < df.Length(); i++ {
		for j := 0; j < df.Data[0].x.GetRows(); j++ {
			maxAll[j] = maxF64(math.Abs(df.Data[i].x.GetIJ(j, 0)), maxAll[j])
		}
	}

	for i := 0; i < len(maxAll); i++ {
		if maxAll[i] == 0 {
			maxAll[i] = 1
		}
	}

	for i := 0; i < df.Length(); i++ {
		for j := 0; j < df.Data[0].x.GetRows(); j++ {
			df.Data[i].x.SetIJ(j, 0, df.Data[i].x.GetIJ(j, 0)/maxAll[j])
		}
	}

}

func maxF64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
