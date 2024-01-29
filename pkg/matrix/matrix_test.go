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

	result := A.Dot(B)

	// проверка размерности получившейся матрицы
	if result.columns != expected.columns || result.rows != expected.rows {
		t.Errorf("Matrix multiplication error: Expected %d * %d, got %d * %d", expected.rows, expected.columns, result.rows, result.columns)
	}

	// проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix multiplication error: Result != Expected")
	}

}

func isMatrixesEqual(A, B *Matrix) bool {

	// радиус окрестности допущения для вещественных чисел
	epsilon := 1e-9

	if A.rows != B.rows || A.columns != B.columns {
		return false
	}

	// проверка на равность
	for i := 0; i < A.rows; i++ {
		for j := 0; j < A.columns; j++ {
			if math.Abs(A.arr[i][j]-B.arr[i][j]) > epsilon {
				return false
			}
		}
	}
	return true
}

func TestAddition(t *testing.T) {
	A := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		},
	}

	B := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{9.0, 8.0, 7.0},
			{6.0, 5.0, 4.0},
			{3.0, 2.0, 1.0},
		},
	}

	expected := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{10.0, 10.0, 10.0},
			{10.0, 10.0, 10.0},
			{10.0, 10.0, 10.0},
		},
	}

	result := A.Addition(B)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix additional error: Result != Expected")
	}
}

func TestSub(t *testing.T) {
	A := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		},
	}

	B := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{9.0, 8.0, 7.0},
			{6.0, 5.0, 4.0},
			{3.0, 2.0, 1.0},
		},
	}

	expected := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{-8.0, -6.0, -4.0},
			{-2.0, 0.0, 2.0},
			{4.0, 6.0, 8.0},
		},
	}

	result := A.Sub(B)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix subtraction error: Result != Expected")
	}
}

func TestHadamardProduct(t *testing.T) {
	A := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{1.0, 2.0, 2.0},
			{-4.0, 0.0, 1.0},
			{5.0, 3.0, -10.0},
		},
	}

	B := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{1.0, 5.0, 3.0},
			{-3.0, 5.0, -4.0},
			{3.0, 2.0, 1.0},
		},
	}

	expected := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{1.0, 10.0, 6.0},
			{12.0, 0.0, -4.0},
			{15.0, 6.0, -10.0},
		},
	}

	result := A.HadamardProduct(B)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix Hadamard product error: Result != Expected")
	}
}

func TestT(t *testing.T) {
	M := &Matrix{
		rows:    2,
		columns: 3,
		arr: [][]float64{
			{12.0, 0.0, -4.0},
			{15.0, 6.0, -10.0},
		},
	}

	expected := &Matrix{
		rows:    3,
		columns: 2,
		arr: [][]float64{
			{12.0, 15.0},
			{0.0, 6.0},
			{-4.0, -10.0},
		},
	}

	result := M.T()

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix transposition error: Result != Expected")
	}
}

func TestSlice2Matrix(t *testing.T) {
	rows := 2
	columns := 3
	slc := []float64{12.0, 0.0, -4.0, 15.0, 6.0, -10.0}

	expected := &Matrix{
		rows:    2,
		columns: 3,
		arr: [][]float64{
			{12.0, 0.0, -4.0},
			{15.0, 6.0, -10.0},
		},
	}

	result := Slice2Matrix(slc, rows, columns)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Slice2Matrix error: Result != Expected")
	}
}

func TestForEach(t *testing.T) {
	M := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{1.0, 10.0, 6.0},
			{12.0, 0.0, -4.0},
			{15.0, 6.0, -10.0},
		},
	}

	expected := &Matrix{
		rows:    3,
		columns: 3,
		arr: [][]float64{
			{1.0, 100.0, 36.0},
			{144.0, 0.0, 16.0},
			{225.0, 36.0, 100.0},
		},
	}

	result := M.ForEach(square)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("ForEach error: Result != Expected")
	}
}

func square(x float64) float64 {
	return x * x
}
