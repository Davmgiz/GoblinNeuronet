package matrix

/*
Оболочка над реализацией матриц

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

func Zero(rows, columns int) Matrix {
	return Matrix{
		matrix: zero(rows, columns),
	}
}

func RandMatrix(rows, columns int) Matrix {
	return Matrix{
		matrix: randMatrix(rows, columns),
	}
}

func (M Matrix) GetColumns() int {
	return M.matrix.getColumns()
}

func (M Matrix) GetRows() int {
	return M.matrix.getRows()
}

func (M Matrix) GetIJ(i, j int) float64 {
	return M.matrix.getIJ(i, j)
}

func (M Matrix) SetIJ(i, j int, x float64) {
	M.matrix.setIJ(i, j, x)
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

func (M Matrix) ForEachInner(f func(float64) float64) {
	M.matrix.forEachInner(f)
}

func (M Matrix) Slice2Matrix(slc []float64) {
	M.matrix.slice2Matrix(slc)

}

func Matrix2Vector(M Matrix, n int) (Matrix, error) {
	myMatr, err := matrix2Vector(M.matrix, n)
	if err != nil {
		return Matrix{}, err
	}

	return Matrix{
		matrix: myMatr,
	}, nil
}

func Vec2Dig(M Matrix) int {
	return vec2Dig(M.matrix)
}

func Num(M Matrix) float64 {
	return M.matrix.num()
}

/*
Функции предназначены для работы тестов
*/

func DataToMatrix(arr [][]float64) Matrix {
	return Matrix{
		matrix: _dataToMatrix(arr),
	}
}

func IsMatrixesEqual(A, B Matrix) bool {
	return _isMatrixesEqual(A.matrix, B.matrix)
}

func countUniqueElements(M Matrix) int {
	return _countUniqueElements(M.matrix)
}

func (M Matrix) Show() {
	M.matrix.show()
}
