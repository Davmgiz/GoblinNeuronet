package matrix

import (
	"sync"
)

// addInPlace реализует сложение матриц A и B (структур myMatrix).
// Результат сохраняется в A, изменяя ее.
// Метод вызывает панику если матрицы по определению нельзя умножить.
func (A *myMatrix) addInPlace(B *myMatrix) {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		panic("Incorrect dimension for myMatrix additional")
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

// subInPlace реализует разность матриц A и B (структур myMatrix).
// Результат сохраняется в A, изменяя ее.
// Метод вызывает панику ,если размерность исходных матриц не равны.
func (A *myMatrix) subInPlace(B *myMatrix) {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		panic("Incorrect dimension for myMatrix subtraction")
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

// hadamardProductInPlace реализует адамарное произведение матриц A и B (структур myMatrix).
// Результат сохраняется в A, изменяя ее.
// Метод вызывает панику ,если размерность исходных матриц не равны.
// Результатом адамарного произведение есть матрица такой же размерности что и перемножаемые матрицы,
// где i, j элемент результирующей матрицы равен произведению соответствующих элементов исходных матриц.
func (A *myMatrix) hadamardProductInPlace(B *myMatrix) {
	if A.getRows() != B.getRows() || A.getColumns() != B.getColumns() {
		panic("Incorrect dimension for myMatrix Hadamard product")
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

// forEachInPlace применяет к каждому элементу исходной матрицы M функцию f func(float64) float64.
// Результат сохраняется в M, изменяя ее.
func (M *myMatrix) forEachInPlace(f func(float64) float64) {
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
