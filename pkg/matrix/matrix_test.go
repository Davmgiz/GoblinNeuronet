package matrix

import (
	"math"
	"testing"
)

// проверка рандома который заполняет матрицы
func TestRandMatrix(t *testing.T) {

	// 2 проверяющие матрицы
	matrix1 := RandMatrix(3, 3)
	matrix2 := RandMatrix(4, 4)

	// если коэф повторяемости элементов матрицы ниже данной границы, то такой рандом нас не устраивает
	threshold := 0.8

	// подсчет коэф для обеих матриц
	repeatedMatrix1 := float64(countUniqueElements(matrix1)) / float64(matrix1.rows*matrix1.columns)
	repeatedMatrix2 := float64(countUniqueElements(matrix2)) / float64(matrix2.rows*matrix2.columns)

	if repeatedMatrix1 < threshold {
		t.Errorf("Too many repeated elements in matrix1. Expect %f, got %f", threshold, repeatedMatrix1)
	}

	if repeatedMatrix2 < threshold {
		t.Errorf("Too many repeated elements in matrix2. Expect %f, got %f", threshold, repeatedMatrix2)
	}

}

// функция считает кол-во повторяющихся элементов во всей матрице
func countUniqueElements(m *Matrix) int {
	uniqueElements := make(map[float64]bool)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.columns; j++ {
			uniqueElements[m.arr[i][j]] = true
		}
	}

	return len(uniqueElements)
}

// проверка произведения матриц
func TestDot(t *testing.T) {

	A := &Matrix{
		rows:    3,
		columns: 2,
		arr: [][]float64{
			{1., 2.},
			{3., 4.},
			{5., 6.},
		},
	}

	B := &Matrix{
		rows:    2,
		columns: 4,
		arr: [][]float64{
			{1., 2., 3., 10.},
			{4., 5., 6., 9.},
		},
	}

	expected := &Matrix{
		rows:    3,
		columns: 4,
		arr: [][]float64{
			{9., 12., 15., 28.},
			{19., 26., 33., 66.},
			{29., 40., 51., 104.},
		},
	}

	result, err := A.Dot(B)

	if err != nil {
		t.Errorf("It is impossible to multiply matrixes A and B")
	}

	// проверка размерности получившейся матрицы
	if result.columns != expected.columns || result.rows != expected.rows {
		t.Errorf("Don't right dimension. Expected %d * %d, got %d * %d", expected.rows, expected.columns, result.rows, result.columns)
	}

	// радиус окрестности допущения для вещественных чисел
	epsilon := 1e-9

	//проверка на равность матриц
	for i := 0; i < result.rows; i++ {
		for j := 0; j < result.columns; j++ {
			if math.Abs(result.arr[i][j]-expected.arr[i][j]) > epsilon {
				t.Errorf("Don't equal matrix. Result != Expected")
			}
		}
	}

}
