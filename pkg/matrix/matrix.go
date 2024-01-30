package matrix

/*
Оболочка над реализацией матриц
*/

/*
Структура должна содержать следующие методы:

GetColumns(),
GetRows(),
Dot(B Matrix) Matrix // Умножение матриц
Add(B Matrix) Matrix // Сложение матриц
Sub(B Matrix) Matrix // Разность матриц
HadamardProduct(B Matrix) Matrix // Адамарное произведение матриц
T() Matrix // Транспонирование матрицы
ForEach(f func(float64) float64) Matrix // поэлементное применение переданной функции к матрицы
Slice2Matrix(slc []float64) // заполнение матрицы элементами из слайса

Должны быть определены функции работающие со структурой:

Zeros(rows, columns int) Matrix // возвращает нулевую матрицу данной размерности
RandMatrix(rows, columns int) Matrix // возвращает матрицу с рандомными элементами

Должны быть определены следующие функции для работы тестов:

dataToMatrix(arr [][]float64) Matrix // из 2д массива делает матрицу
isMatrixesEqual(A, B Matrix) bool // проверяет на равность матрицы
countUniqueElements(M Matrix) int // возвращает количество уникальных элементов

*/
type Matrix struct {
	matrix *myMatrix
}

func Zeros(rows, columns int) Matrix {
	return Matrix{
		matrix: zeros(rows, columns),
	}
}

func RandMatrix(rows, columns int) Matrix {
	return Matrix{
		matrix: randMatrix(rows, columns),
	}
}

func (M Matrix) GetColumns() int {
	return M.matrix.columns
}

func (M Matrix) GetRows() int {
	return M.matrix.rows
}

func (A Matrix) Dot(B Matrix) Matrix {
	return Matrix{
		matrix: A.matrix.dot(B.matrix),
	}
}

func (A Matrix) Add(B Matrix) Matrix {
	return Matrix{
		matrix: A.matrix.add(B.matrix),
	}
}

func (A Matrix) Sub(B Matrix) Matrix {
	return Matrix{
		matrix: A.matrix.sub(B.matrix),
	}
}

func (A Matrix) HadamardProduct(B Matrix) Matrix {
	return Matrix{
		matrix: A.matrix.hadamardProduct(B.matrix),
	}
}

func (M Matrix) T() Matrix {
	return Matrix{
		matrix: M.matrix.t(),
	}
}

func (M Matrix) ForEach(f func(float64) float64) Matrix {
	return Matrix{
		matrix: M.matrix.forEach(f),
	}
}

func (M Matrix) Slice2Matrix(slc []float64) {
	M.matrix.slice2Matrix(slc)
}

/*
Функции предназначены для работы тестов
*/

func dataToMatrix(arr [][]float64) Matrix {
	return Matrix{
		matrix: _dataToMatrix(arr),
	}
}

func isMatrixesEqual(A, B Matrix) bool {
	return _isMatrixesEqual(A.matrix, B.matrix)
}

func countUniqueElements(M Matrix) int {
	return _countUniqueElements(M.matrix)
}
