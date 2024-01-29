package matrix

import (
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"
)

/*
Работа происходит только с указателями на матрицы
Любая функция или любой метод возвращает указатель на матрицу
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
	matrix := Zeros(rows, columns)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			matrix.arr[i][j] = rand.NormFloat64()
		}
	}

	return matrix
}

// функция создает нулевую матрицу данной размерности
func Zeros(rows, columns int) *Matrix {

	if rows <= 0 || columns <= 0 {
		log.Fatal("Matrix dimensions must be positive: rows and columns should be greater than 0")
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
func (A *Matrix) Dot(B *Matrix) *Matrix {
	if A.columns != B.rows {
		log.Fatal("Incorrect dimension for matrix multiplication")
	}

	wg := new(sync.WaitGroup)
	wg.Add(A.rows * B.columns)

	C := Zeros(A.rows, B.columns)

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

	return C

}

// функция сложения матриц
func (A *Matrix) Addition(B *Matrix) *Matrix {
	if A.rows != B.rows || A.columns != B.columns {
		log.Fatal("Incorrect dimension for matrix additional")
	}

	C := Zeros(A.rows, A.columns)

	for i := 0; i < A.rows; i++ {
		for j := 0; j < A.columns; j++ {
			C.arr[i][j] = A.arr[i][j] + B.arr[i][j]
		}
	}

	return C
}

// функция вычитания матриц
func (A *Matrix) Sub(B *Matrix) *Matrix {
	if A.rows != B.rows || A.columns != B.columns {
		log.Fatal("Incorrect dimension for matrix subtraction")
	}

	C := Zeros(A.rows, A.columns)

	for i := 0; i < A.rows; i++ {
		for j := 0; j < A.columns; j++ {
			C.arr[i][j] = A.arr[i][j] - B.arr[i][j]
		}
	}

	return C
}

// адамарное произведение (поэлиментное произведение)
func (A *Matrix) HadamardProduct(B *Matrix) *Matrix {
	if A.rows != B.rows || A.columns != B.columns {
		log.Fatal("Incorrect dimension for matrix Hadamard product")
	}

	C := Zeros(A.rows, A.columns)

	for i := 0; i < A.rows; i++ {
		for j := 0; j < A.columns; j++ {
			C.arr[i][j] = A.arr[i][j] * B.arr[i][j]
		}
	}

	return C
}

// транспонирование матрицы
func (M *Matrix) T() *Matrix {
	matrix := Zeros(M.columns, M.rows)

	for i := 0; i < M.rows; i++ {
		for j := 0; j < M.columns; j++ {
			matrix.arr[j][i] = M.arr[i][j]
		}
	}

	return matrix
}

// делаем из слайса матрицу данной размерности
func Slice2Matrix(slc []float64, rows, columns int) *Matrix {
	if rows*columns != len(slc) {
		log.Fatal("Incorrect dimension of the result matrix or lenght of the slice for creating a matrix from a slice")
	}

	matrix := Zeros(rows, columns)

	ind := 0

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			matrix.arr[i][j] = slc[ind]
			ind++
		}
	}

	return matrix
}

func (M *Matrix) ForEach(f func(float64) float64) *Matrix {
	matrix := Zeros(M.rows, M.columns)

	for i := 0; i < M.rows; i++ {
		for j := 0; j < M.columns; j++ {
			matrix.arr[i][j] = f(M.arr[i][j])
		}
	}
	return matrix
}
