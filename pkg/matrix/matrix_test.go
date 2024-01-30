package matrix

import (
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
	repeatedMatrix1 := float64(countUniqueElements(matrix1)) / float64(matrix1.matrix.rows*matrix1.matrix.columns)
	repeatedMatrix2 := float64(countUniqueElements(matrix2)) / float64(matrix2.matrix.rows*matrix2.matrix.columns)

	if repeatedMatrix1 < threshold {
		t.Errorf("Too many repeated elements in matrix1. Expect %f, got %f", threshold, repeatedMatrix1)
	}

	if repeatedMatrix2 < threshold {
		t.Errorf("Too many repeated elements in matrix2. Expect %f, got %f", threshold, repeatedMatrix2)
	}

}

func TestDot(t *testing.T) {
	a := [][]float64{
		{1., 2.},
		{3., 4.},
		{5., 6.},
	}
	A := dataToMatrix(a)

	b := [][]float64{
		{1., 2., 3., 10.},
		{4., 5., 6., 9.},
	}
	B := dataToMatrix(b)

	e := [][]float64{
		{9., 12., 15., 28.},
		{19., 26., 33., 66.},
		{29., 40., 51., 104.},
	}
	expected := dataToMatrix(e)

	result := A.Dot(B)

	// проверка размерности получившейся матрицы
	if result.GetColumns() != expected.GetColumns() || result.GetRows() != expected.GetRows() {
		t.Errorf("Matrix multiplication error: Expected %d * %d, got %d * %d", expected.GetRows(), expected.GetColumns(), result.GetRows(), result.GetColumns())
	}

	// проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix multiplication error: Result != Expected")
	}

}

func TestAddition(t *testing.T) {
	a := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	A := dataToMatrix(a)

	b := [][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	}
	B := dataToMatrix(b)

	e := [][]float64{
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
	}
	expected := dataToMatrix(e)

	result := A.Add(B)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix additional error: Result != Expected")
	}
}

func TestSub(t *testing.T) {
	a := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	A := dataToMatrix(a)

	b := [][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	}
	B := dataToMatrix(b)

	e := [][]float64{
		{-8.0, -6.0, -4.0},
		{-2.0, 0.0, 2.0},
		{4.0, 6.0, 8.0},
	}
	expected := dataToMatrix(e)

	result := A.Sub(B)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix subtraction error: Result != Expected")
	}
}

func TestHadamardProduct(t *testing.T) {
	a := [][]float64{
		{1.0, 2.0, 2.0},
		{-4.0, 0.0, 1.0},
		{5.0, 3.0, -10.0},
	}
	A := dataToMatrix(a)

	b := [][]float64{
		{1.0, 5.0, 3.0},
		{-3.0, 5.0, -4.0},
		{3.0, 2.0, 1.0},
	}
	B := dataToMatrix(b)

	e := [][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	expected := dataToMatrix(e)

	result := A.HadamardProduct(B)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Matrix Hadamard product error: Result != Expected")
	}
}

func TestT(t *testing.T) {
	m := [][]float64{
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	M := dataToMatrix(m)

	e := [][]float64{
		{12.0, 15.0},
		{0.0, 6.0},
		{-4.0, -10.0},
	}
	expected := dataToMatrix(e)

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

	e := [][]float64{
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	expected := dataToMatrix(e)

	A := Zeros(rows, columns)
	A.Slice2Matrix(slc)

	result := A

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("Slice2Matrix error: Result != Expected")
	}
}

func TestForEach(t *testing.T) {
	m := [][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	M := dataToMatrix(m)

	e := [][]float64{
		{1.0, 100.0, 36.0},
		{144.0, 0.0, 16.0},
		{225.0, 36.0, 100.0},
	}
	expected := dataToMatrix(e)

	result := M.ForEach(square)

	//проверка на равность
	if !isMatrixesEqual(result, expected) {
		t.Errorf("ForEach error: Result != Expected")
	}
}

func square(x float64) float64 {
	return x * x
}
