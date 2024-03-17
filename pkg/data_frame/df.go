/*
Package data_frame предоставляет набор инструментов для работы с данными. Основные возможности
включают в себя создание датафрейма из CSV файла, нормализацию датафрейма.
Пакет предназначен для реализации нейронной сети.
*/
package data_frame

import (
	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// rowDataFrame представляет структуру одного наблюдения.
type rowDataFrame struct {
	x matrix.Matrix // Вектор из признаков (матрица размерности n на 1, где n некоторое число)
	y matrix.Matrix // Целевая переменная
}

// GetX возвращает вектор признаков исходного наблюдения(структуру Matrix)
func (row *rowDataFrame) GetX() matrix.Matrix {
	return row.x
}

// GetY возвращает матрицу (структуру Matrix) размерности 1 на 1,
// в которой содержится целевая переменная исходного наблюдения.
func (row *rowDataFrame) GetY() matrix.Matrix {
	return row.y
}

// DataFrame представляет структуру датафрейма.
// Датафрейм содержит массив из указателей на структуру rowDataFrame.
type DataFrame struct {
	Data []*rowDataFrame // массив из указателей на структуры наблюдений
}

// Length возвращает количество наблюдений датафрейма.
func (df *DataFrame) Lenght() int {
	return len(df.Data)
}

// GetRow возвращает x, y наблюдения с индексом i датафрейма.
// Возвращает вектор признаков и целевую переменную соответственно.
func (df *DataFrame) GetRow(i int) (matrix.Matrix, matrix.Matrix) {
	return df.Data[i].x, df.Data[i].y
}
