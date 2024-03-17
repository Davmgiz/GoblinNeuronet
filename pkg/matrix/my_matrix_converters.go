package matrix

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// slice2Matrix преобразует слайс []float64 в матрицу (структуру myMatrix).
// Результат сохраняется в M, изменяя ее.
// Метод вызывает панику, если количество элементов матрицы не равно количеству элементов слайса.
func (M *myMatrix) slice2Matrix(slc []float64) {
	if M.getRows()*M.getColumns() != len(slc) {
		panic("Incorrect dimension of the result myMatrix or lenght of the slice for creating a myMatrix from a slice")
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

// num возвращает единственный элемент матрицы (структуры myMatrix) размерности 1 на 1
// Функция вызывает панику, если размерность исходной матрицы не 1 на 1.
func (M *myMatrix) num() float64 {
	if M.getColumns() != 1 && M.getRows() != 1 {
		panic("Matrix dimension  must be 1*1")
	}
	return M.data[0][0]
}

// float64ToInt возвращает преобразованное вещественное число в целое.
// Если преобразование невозможно метод возвращает ошибку.
func float64ToInt(x float64) (int, error) {
	xF := int(x)
	if math.Abs(float64(xF)-x) > 1e-6 {
		return -1, errors.New("incorrect matrix value for converting to int")
	}
	return xF, nil
}

// num2Vec возвращает указатель на матрицу (структуру myMatrix) и ошибку.
// Возвращаемая матрица является вектором, то есть матрицей размерности n на 1.
// Возвращающая матрица является нулевой, но на месте x-ого находится 1.
// Функция возвращает ошибку, если невозможно преобразовать вещественное число x в целочисленное.
// Так же функция возвращает ошибку если данная размерность n слишком мала.
func num2Vec(x float64, n int) (*myMatrix, error) {
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

// matrix2Vector возвращает указатель на структуру myMatrix и ошибку.
// Возвращаемая матрица является вектором, то есть матрицей размерности n на 1.
// Возвращающая матрица является нулевой, но 1 элемент этого вектора равен 1,
// индекс этого элемента есть единственный элемент матрицы M.
// Функция возвращает ошибку, если невозможно преобразовать вещественное число x в целочисленное.
// Так же функция возвращает ошибку если данная размерность n слишком мала.
// Предупреждение: функция может вызвать панику, если матрица M не имеет размерность 1 на 1,
// поскольку внутренне вызывается метод num, требующий матрицу размерности 1 на 1.
func matrix2Vector(M *myMatrix, n int) (*myMatrix, error) {
	digF := M.num()
	res, err := num2Vec(digF, n)
	if err != nil {
		return &myMatrix{}, err
	}
	return res, nil

}

// vec2Dig возвращает целое число - индекс наибольшего элемента вектора (матрицы M размерности n на 1).
// функция работает обратно функции num2Vec.
// Функция вызывает панику, если матрица не является вектором и если матрица имеет размерность 1 на 1.
func vec2Num(M *myMatrix) int {
	if M.getColumns() != 1 {
		panic("Incorrect dimension of matrix for converting to digit")
	}
	if M.getRows() <= 1 {
		panic("Incorrect vector")
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
