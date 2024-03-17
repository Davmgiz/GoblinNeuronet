package matrix

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// myMatrix представляет структуру матрицы.
// Индексация элементов начинается с 0. Это означает, что первый элемент в любом столбце или строке
// имеет индекс 0, а не 1. Структура хранит количество столбцов и строк, а также саму матрицу
// в виде слайса [][]float64.
type myMatrix struct {
	columns int         // Количество столбцов в матрице
	rows    int         // Количество строк в матрице
	data    [][]float64 // Данные матрицы, хранящиеся в двумерном массиве
}

// init инициализирует генератор псевдослучайных чисел.
func init() {
	rand.Seed(time.Now().UnixNano())
}

// zero создает и возвращает указатель на новый экземпляр myMatrix с заданными размерами rows и columns.
// Все элементы матрицы инициализируются нулями.
// Функция вызывает панику, если указанные размеры матрицы не являются положительными числами.
func zero(rows, columns int) *myMatrix {
	if rows <= 0 || columns <= 0 {
		panic("myMatrix dimensions must be positive: rows and columns should be greater than 0")
	}

	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, columns)
	}

	return &myMatrix{
		columns: columns,
		rows:    rows,
		data:    data,
	}
}

// randMatrix создает и возвращает указатель на новый экземпляр myMatrix с заданными размерами rows и columns.
// Все элементы матрицы инициализируются случайными значениями из нормального распределения с математическим ожиданием 0
// и стандартным отклонением равным 0,01.
// Предупреждение: функция может вызвать панику, если rows или columns будут не положительными,
// поскольку внутренне вызывается функция zero, требующая положительных значений для этих параметров.
func randMatrix(rows, columns int) *myMatrix {
	myMatrix := zero(rows, columns)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			myMatrix.data[i][j] = rand.NormFloat64() * 0.01
		}
	}

	return myMatrix
}

// getRows возвращает количество строк данной матрицы (структуры myMatrix).
func (M *myMatrix) getRows() int {
	return M.rows
}

// getColumns возвращает количество строк данной матрицы (структуры myMatrix).
func (M *myMatrix) getColumns() int {
	return M.columns
}

// getIJ возвращает элемент i строки, j столбца данной матрицы (структуры myMatrix).
func (M *myMatrix) getIJ(i, j int) float64 {
	return M.data[i][j]
}

// setIJ устанавливает элемент i строки, j столбца данной матрицы (структуры myMatrix).
func (M *myMatrix) setIJ(i, j int, x float64) {
	M.data[i][j] = x
}

// show выводит матрицу (структуру myMatrix) в консоль.
func (M *myMatrix) show() {
	for _, row := range M.data {
		fmt.Println(row)
	}
}

// dataToMatrix предназначена для тестов.
// Создает и возвращает указатель на матрицу (структуру myMatrix).
// Структура myMatrix получена из слайса [][]float64.
// Функция вызывает панику если слайс состоит из неравных по количеству элементов строк.
func dataToMatrix(arr [][]float64) *myMatrix {
	len0 := len(arr[0])
	for i := 1; i < len(arr); i++ {
		if len(arr[i]) != len0 {
			panic("different rows of the matrix have different lengths")
		}
	}

	rows := len(arr)
	columns := len(arr[0])

	return &myMatrix{
		rows:    rows,
		columns: columns,
		data:    arr,
	}
}

// isMatrixesEqual предназначена для тестов.
// Функция возвращает true, если структуры myMatrix равны, иначе false.
func isMatrixesEqual(A, B *myMatrix) bool {

	// радиус окрестности допущения для вещественных чисел
	epsilon := float64(1e-6)

	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		return false
	}

	// проверка на равность
	for i := 0; i < A.getRows(); i++ {
		for j := 0; j < A.getColumns(); j++ {
			if math.Abs(A.data[i][j]-B.data[i][j]) > epsilon {
				return false
			}
		}
	}
	return true
}

// _countUniqueElements предназначена для тестов.
// Функция возвращает количество уникальных элементов матрицы (структуры myMatrix).
// Функция предназначена для проверки функции randMatrix.
func _countUniqueElements(M *myMatrix) int {
	uniqueElements := make(map[float64]bool)
	for i := 0; i < M.getRows(); i++ {
		for j := 0; j < M.getColumns(); j++ {
			uniqueElements[M.data[i][j]] = true
		}
	}

	return len(uniqueElements)
}
