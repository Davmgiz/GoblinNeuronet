/*
Реализация собственного пакета для работы с матрицами
Работа происходит только с указателями на матрицы
Любая функция или любой метод возвращает указатель на матрицу
*/
package matrix

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

// структура матрицы
type myMatrix struct {
	columns int         // столбцы
	rows    int         // строки
	data    [][]float64 // 2ух мерный массив
}

// инициализация генератора псевдослучайных чисел
func init() {
	rand.Seed(time.Now().UnixNano())
}

// функция создает нулевую матрицу данной размерности
func zero(rows, columns int) *myMatrix {

	if rows <= 0 || columns <= 0 {
		log.Fatal("myMatrix dimensions must be positive: rows and columns should be greater than 0")
		//panic("myMatrix dimensions must be positive: rows and columns should be greater than 0")
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
	myMatrix := zero(rows, columns)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			myMatrix.data[i][j] = rand.NormFloat64() * 0.01
		}
	}

	return myMatrix
}

// геттер строк матрицы
func (M *myMatrix) getRows() int {
	return M.rows
}

// геттер столбцов матрицы
func (M *myMatrix) getColumns() int {
	return M.columns
}

func (M *myMatrix) getIJ(i, j int) float64 {
	return M.data[i][j]
}

func (M *myMatrix) setIJ(i, j int, x float64) {
	M.data[i][j] = x
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

	C := zero(A.getRows(), B.getColumns())

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

	C := zero(A.getRows(), A.getColumns())

	wg := new(sync.WaitGroup)
	wg.Add(A.getRows())
	for i := 0; i < A.getRows(); i++ {

		go func(i int) {
			defer wg.Done()
			for j := 0; j < A.getColumns(); j++ {
				C.data[i][j] = A.data[i][j] + B.data[i][j]
			}

		}(i)
	}

	wg.Wait()

	return C
}

func (A *myMatrix) addSelf(B *myMatrix) {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix additional")
	}

	wg := new(sync.WaitGroup)
	wg.Add(A.getRows())
	for i := 0; i < A.getRows(); i++ {

		go func(i int) {
			defer wg.Done()
			for j := 0; j < A.getColumns(); j++ {
				A.data[i][j] = A.data[i][j] + B.data[i][j]
			}

		}(i)
	}

	wg.Wait()
}

// функция вычитания матриц
func (A *myMatrix) sub(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix subtraction")
	}

	C := zero(A.getRows(), A.getColumns())

	wg := new(sync.WaitGroup)
	wg.Add(A.getRows())

	for i := 0; i < A.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < A.getColumns(); j++ {
				C.data[i][j] = A.data[i][j] - B.data[i][j]
			}
		}(i)

	}
	wg.Wait()

	return C
}

func (A *myMatrix) subSelf(B *myMatrix) {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix subtraction")
	}

	wg := new(sync.WaitGroup)
	wg.Add(A.getRows())

	for i := 0; i < A.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < A.getColumns(); j++ {
				A.data[i][j] = A.data[i][j] - B.data[i][j]
			}
		}(i)

	}
	wg.Wait()
}

// адамарное произведение (поэлементное произведение)
func (A *myMatrix) hadamardProduct(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix Hadamard product")
	}

	C := zero(A.getRows(), A.getColumns())

	wg := new(sync.WaitGroup)
	wg.Add(A.getRows())

	for i := 0; i < A.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < A.getColumns(); j++ {
				C.data[i][j] = A.data[i][j] * B.data[i][j]
			}
		}(i)

	}
	wg.Wait()

	return C
}

func (A *myMatrix) hadamardProductSelf(B *myMatrix) {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		log.Fatal("Incorrect dimension for myMatrix Hadamard product")
	}

	wg := new(sync.WaitGroup)
	wg.Add(A.getRows())

	for i := 0; i < A.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < A.getColumns(); j++ {
				A.data[i][j] = A.data[i][j] * B.data[i][j]
			}
		}(i)

	}
	wg.Wait()
}

// транспонирование матрицы
func (M *myMatrix) t() *myMatrix {
	myMatrix := zero(M.getColumns(), M.getRows())

	wg := new(sync.WaitGroup)
	wg.Add(M.getRows())

	for i := 0; i < M.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < M.getColumns(); j++ {
				myMatrix.data[j][i] = M.data[i][j]
			}
		}(i)

	}

	wg.Wait()

	return myMatrix
}

// функция принимает функцию и применяет ее для каждого элемента матрицы
func (M *myMatrix) forEach(f func(float64) float64) *myMatrix {
	myMatrix := zero(M.getRows(), M.getColumns())

	wg := new(sync.WaitGroup)
	wg.Add(M.getRows())

	for i := 0; i < M.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < M.getColumns(); j++ {
				myMatrix.data[i][j] = f(M.data[i][j])
			}
		}(i)

	}
	wg.Wait()

	return myMatrix
}

func (M *myMatrix) forEachSelf(f func(float64) float64) {
	wg := new(sync.WaitGroup)
	wg.Add(M.getRows())

	for i := 0; i < M.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < M.getColumns(); j++ {
				M.data[i][j] = f(M.data[i][j])
			}
		}(i)

	}
	wg.Wait()
}

// заполняем матрицу числами из слайса
func (M *myMatrix) slice2Matrix(slc []float64) {
	if M.getRows()*M.getColumns() != len(slc) {
		log.Fatal("Incorrect dimension of the result myMatrix or lenght of the slice for creating a myMatrix from a slice")
	}

	wg := new(sync.WaitGroup)
	wg.Add(M.getRows())

	for i := 0; i < M.getRows(); i++ {

		go func(i int) {
			defer wg.Done()

			for j := 0; j < M.getColumns(); j++ {
				M.data[i][j] = slc[i*M.columns+j]
			}
		}(i)

	}

	wg.Wait()

}

func (M *myMatrix) num() float64 {
	if M.getColumns() != 1 && M.getRows() != 1 {
		log.Fatal("Matrix dimension  must be 1*1")
	}
	return M.data[0][0]
}

func float64ToInt(x float64) (int, error) {
	xF := int(x)
	if math.Abs(float64(xF)-x) > 1e-9 {
		return -1, errors.New("incorrect matrix value for converting to int")
	}
	return xF, nil
}

func dig2Vec(x float64, n int) (*myMatrix, error) {
	xI, err := float64ToInt(x)
	if err != nil {
		return &myMatrix{}, err
	}
	if xI > n || xI < 0 {
		return &myMatrix{}, fmt.Errorf("digit must be [%v, %v]", 0, n)
	}

	res := zero(n, 1)
	res.data[xI][0] = 1.

	return res, nil

}

func matrix2Vector(M *myMatrix, n int) (*myMatrix, error) {
	digF := M.num()
	res, err := dig2Vec(digF, n)
	if err != nil {
		return &myMatrix{}, err
	}
	return res, nil

}

func vec2Dig(M *myMatrix) int {
	if M.getColumns() != 1 {
		log.Fatal("Incorrect dimension of matrix for converting to digit")
	}
	if M.getRows() <= 1 {
		log.Fatal("Incorrect vector")
	}

	max := M.data[0][0]
	ind := 0

	for i := 1; i < M.getRows(); i++ {
		if M.data[i][0] > max {
			ind = i
			max = M.data[i][0]
		}
	}
	return ind
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
