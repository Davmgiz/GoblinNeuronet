package matrix

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

/*
Работаем только с указателями на матрицы
Любая функция или любой метод возвращает указатель на матрицу
Сама работа с матрицами происходит только по указателю
*/

// структура матрицы
type Matrix struct {
	columns int         // столбцы
	rows    int         // строки
	arr     [][]float64 // 2ух мерный массив
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

// функция создает матрицы с элементами из нормального распределения со средним значением 0 и стандартным отклонением 1
func RandMatrix(rows, columns int) *Matrix {
	if rows <= 0 || columns <= 0 {
		panic("RandMatrix: row <= 0 or columns <= 0")
	}

	matrix := EmptyMatrix(rows, columns)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			matrix.arr[i][j] = rand.NormFloat64()
		}
	}

	return matrix
}

// функция создает пустую матрицу
func EmptyMatrix(rows, columns int) *Matrix {

	if rows <= 0 || columns <= 0 {
		panic("EmptyMatrix: row <= 0 or columns <= 0")
	}

	arr := make([][]float64, rows)
	for i := range arr {
		arr[i] = make([]float64, columns)
	}

	return &Matrix{
		columns: columns,
		rows:    rows,
		arr:     arr,
	}
}

// функция выводит на экран
func (M *Matrix) Show() {
	for _, row := range M.arr {
		fmt.Println(row)
	}
}

// произведение матрицы A на B
func (A *Matrix) Dot(B *Matrix) (*Matrix, error) {
	if A.columns != B.rows {
		return &Matrix{}, errors.New("Dot: can't multiply matrices")
	}

	wg := new(sync.WaitGroup)
	wg.Add(A.rows * B.columns)

	C := EmptyMatrix(A.rows, B.columns)

	for i := 0; i < A.rows; i++ {
		for j := 0; j < B.columns; j++ {

			go func(i, j int) {
				defer wg.Done()

				sum := float64(0)

				for k := 0; k < A.columns; k++ {
					sum += A.arr[i][k] * B.arr[k][j]
				}
				C.arr[i][j] = sum
			}(i, j)
		}
	}

	wg.Wait()

	return C, nil

}
