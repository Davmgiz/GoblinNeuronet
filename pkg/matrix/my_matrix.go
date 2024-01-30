package matrix

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

/*
Реализация собственного пакета для работы с матрицами
Работа происходит только с указателями на матрицы
Любая функция или любой метод возвращает указатель на матрицу
*/

// структура матрицы
type myMatrix struct {
	columns int         // столбцы
	rows    int         // строки
	data    [][]float64 // 2ух мерный массив
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

// функция создает нулевую матрицу данной размерности
func zeros(rows, columns int) *myMatrix {

	if rows <= 0 || columns <= 0 {
		log.Fatal("myMatrix dimensions must be positive: rows and columns should be greater than 0")
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

// функция создает матрицы с элементами из нормального распределения со средним значением 0 и стандартным отклонением 1
func randMatrix(rows, columns int) *myMatrix {
	myMatrix := zeros(rows, columns)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			myMatrix.data[i][j] = rand.NormFloat64()
		}
	}

	return myMatrix
}

func (M myMatrix) getRows() int {
	return M.rows
}

func (M myMatrix) getColumns() int {
	return M.columns
}

// функция выводит на экран
func (M *myMatrix) show() {
	for _, row := range M.data {
		fmt.Println(row)
	}
}

// произведение матрицы A на B
func (A *myMatrix) dot(B *myMatrix) *myMatrix {
	if A.getColumns() != B.getRows() {
		log.Fatal("Incorrect dimension for myMatrix multiplication")
	}

	wg := new(sync.WaitGroup)
	wg.Add(A.getRows() * B.getColumns())

	C := zeros(A.getRows(), B.getColumns())

	for i := 0; i < A.getRows(); i++ {
		for j := 0; j < B.getColumns(); j++ {

			go func(i, j int) {
				defer wg.Done()

				sum := float64(0)

				for k := 0; k < A.getColumns(); k++ {
					sum += A.data[i][k] * B.data[k][j]
				}
				C.data[i][j] = sum
			}(i, j)
		}
	}

	wg.Wait()

	return C

}

// функция сложения матриц
func (A *myMatrix) add(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix additional")
	}

	C := zeros(A.getRows(), A.getColumns())

	for i := 0; i < A.getRows(); i++ {
		for j := 0; j < A.getColumns(); j++ {
			C.data[i][j] = A.data[i][j] + B.data[i][j]
		}
	}

	return C
}

// функция вычитания матриц
func (A *myMatrix) sub(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix subtraction")
	}

	C := zeros(A.getRows(), A.getColumns())

	for i := 0; i < A.getRows(); i++ {
		for j := 0; j < A.getColumns(); j++ {
			C.data[i][j] = A.data[i][j] - B.data[i][j]
		}
	}

	return C
}

// адамарное произведение (поэлементное произведение)
func (A *myMatrix) hadamardProduct(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix Hadamard product")
	}

	C := zeros(A.getRows(), A.getColumns())

	for i := 0; i < A.getRows(); i++ {
		for j := 0; j < A.getColumns(); j++ {
			C.data[i][j] = A.data[i][j] * B.data[i][j]
		}
	}

	return C
}

// транспонирование матрицы
func (M *myMatrix) t() *myMatrix {
	myMatrix := zeros(M.getColumns(), M.getRows())

	for i := 0; i < M.getRows(); i++ {
		for j := 0; j < M.getColumns(); j++ {
			myMatrix.data[j][i] = M.data[i][j]
		}
	}

	return myMatrix
}

// функция принимает функцию и применяет ее для каждого элемента матрицы
func (M *myMatrix) forEach(f func(float64) float64) *myMatrix {
	myMatrix := zeros(M.getRows(), M.getColumns())

	for i := 0; i < M.getRows(); i++ {
		for j := 0; j < M.getColumns(); j++ {
			myMatrix.data[i][j] = f(M.data[i][j])
		}
	}
	return myMatrix
}

// заполняем матрицу числами из слайса
func (M *myMatrix) slice2Matrix(slc []float64) {
	if M.getRows()*M.getColumns() != len(slc) {
		log.Fatal("Incorrect dimension of the result myMatrix or lenght of the slice for creating a myMatrix from a slice")
	}

	ind := 0

	for i := 0; i < M.getRows(); i++ {
		for j := 0; j < M.getColumns(); j++ {
			M.data[i][j] = slc[ind]
			ind++
		}
	}

}

/*
Функции предназначены для работы тестов
*/

// преобразуем двухмерный массив в матрицу
func _dataToMatrix(arr [][]float64) *myMatrix {
	len0 := len(arr[0])
	for i := 1; i < len(arr); i++ {
		if len(arr[i]) != len0 {
			log.Fatal("different rows of the matrix have different lengths")
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

// проверка на равенство матриц
func _isMatrixesEqual(A, B *myMatrix) bool {

	// радиус окрестности допущения для вещественных чисел
	epsilon := 1e-9

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

// функция считает кол-во уникальных элементов во всей матрице
func _countUniqueElements(M *myMatrix) int {
	uniqueElements := make(map[float64]bool)
	for i := 0; i < M.getRows(); i++ {
		for j := 0; j < M.getColumns(); j++ {
			uniqueElements[M.data[i][j]] = true
		}
	}

	return len(uniqueElements)
}
