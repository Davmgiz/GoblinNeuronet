package matrix

import (
	"sync"
)

// dot возвращает указатель на структуру myMatrix.
// Возвращаемая матрица является результатом произведения матриц A и B.
// Метод вызывает панику, если исходные матрицы нельзя перемножить по определению.
// Предупреждение: функция может вызвать панику, если rows или columns будут не положительными,
// поскольку внутренне вызывается функция zero, требующая положительных значений для этих параметров.
func (A *myMatrix) dot(B *myMatrix) *myMatrix {
	if A.getColumns() != B.getRows() {
		panic("Incorrect dimension for myMatrix multiplication")
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

// add возвращает указатель на структуру myMatrix.
// Возвращаемая матрица является результатом суммы матриц A и B.
// Метод вызывает панику, если размерности исходных матриц не равны.
// Предупреждение: функция может вызвать панику, если rows или columns будут не положительными,
// поскольку внутренне вызывается функция zero, требующая положительных значений для этих параметров.
func (A *myMatrix) add(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		panic("Incorrect dimension for myMatrix additional")
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

// sub возвращает указатель на структуру myMatrix.
// Возвращаемая матрица является результатом разности матриц A и B.
// Метод вызывает панику, если размерности исходных матриц не равны.
// Предупреждение: функция может вызвать панику, если rows или columns будут не положительными,
// поскольку внутренне вызывается функция zero, требующая положительных значений для этих параметров.
func (A *myMatrix) sub(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		panic("Incorrect dimension for myMatrix subtraction")
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

// hadamardProduct возвращает указатель на структуру myMatrix.
// Возвращаемая матрица является результатом адамарного произведения матриц A и B.
// Метод вызывает панику, если размерности исходных матриц не равны.
// Предупреждение: функция может вызвать панику, если rows или columns будут не положительными,
// поскольку внутренне вызывается функция zero, требующая положительных значений для этих параметров.
// Результатом адамарного произведение есть матрица такой же размерности что и перемножаемые матрицы,
// где i, j элемент результирующей матрицы равен произведению соответствующих элементов исходных матриц.
func (A *myMatrix) hadamardProduct(B *myMatrix) *myMatrix {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		panic("Incorrect dimension for myMatrix Hadamard product")
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

// t возвращает указатель на структуру myMatrix.
// Возвращаемая матрица является транспонированной копией матрицы M.
// Предупреждение: функция может вызвать панику, если rows или columns будут не положительными,
// поскольку внутренне вызывается функция zero, требующая положительных значений для этих параметров.
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

// forEach возвращает указатель на структуру myMatrix.
// Возвращаемая матрица является результатом применения к каждому элементу исходной матрицы M функцию f func(float64) float64.
// Предупреждение: функция может вызвать панику, если rows или columns будут не положительными,
// поскольку внутренне вызывается функция zero, требующая положительных значений для этих параметров.
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
