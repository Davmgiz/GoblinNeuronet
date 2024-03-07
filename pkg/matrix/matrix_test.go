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
	A := DataToMatrix(a)

	b := [][]float64{
		{1., 2., 3., 10.},
		{4., 5., 6., 9.},
	}
	B := DataToMatrix(b)

	e := [][]float64{
		{9., 12., 15., 28.},
		{19., 26., 33., 66.},
		{29., 40., 51., 104.},
	}
	expected := DataToMatrix(e)

	result := A.Dot(B)

	// проверка размерности получившейся матрицы
	if result.GetColumns() != expected.GetColumns() || result.GetRows() != expected.GetRows() {
		t.Errorf("Matrix multiplication error: Expected %d * %d, got %d * %d", expected.GetRows(), expected.GetColumns(), result.GetRows(), result.GetColumns())
	}

	// проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix multiplication error: Result != Expected")
	}

}

func TestAdd(t *testing.T) {
	a := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	A := DataToMatrix(a)

	b := [][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	}
	B := DataToMatrix(b)

	e := [][]float64{
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
	}
	expected := DataToMatrix(e)

	result := A.Add(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix additional error: Result != Expected")
	}
}

func TestAddInner(t *testing.T) {
	r := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	result := DataToMatrix(r)

	b := [][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	}
	B := DataToMatrix(b)

	e := [][]float64{
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
	}
	expected := DataToMatrix(e)

	result.AddSelf(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix additional error: Result != Expected")
	}
}

func TestSub(t *testing.T) {
	a := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	A := DataToMatrix(a)

	b := [][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	}
	B := DataToMatrix(b)

	e := [][]float64{
		{-8.0, -6.0, -4.0},
		{-2.0, 0.0, 2.0},
		{4.0, 6.0, 8.0},
	}
	expected := DataToMatrix(e)

	result := A.Sub(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix subtraction error: Result != Expected")
	}
}

func TestSubInner(t *testing.T) {
	r := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	result := DataToMatrix(r)

	b := [][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	}
	B := DataToMatrix(b)

	e := [][]float64{
		{-8.0, -6.0, -4.0},
		{-2.0, 0.0, 2.0},
		{4.0, 6.0, 8.0},
	}
	expected := DataToMatrix(e)

	result.SubSelf(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix subtraction error: Result != Expected")
	}
}

func TestHadamardProduct(t *testing.T) {
	a := [][]float64{
		{1.0, 2.0, 2.0},
		{-4.0, 0.0, 1.0},
		{5.0, 3.0, -10.0},
	}
	A := DataToMatrix(a)

	b := [][]float64{
		{1.0, 5.0, 3.0},
		{-3.0, 5.0, -4.0},
		{3.0, 2.0, 1.0},
	}
	B := DataToMatrix(b)

	e := [][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	expected := DataToMatrix(e)

	result := A.HadamardProduct(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix Hadamard product error: Result != Expected")
	}
}

func TestHadamardProductInner(t *testing.T) {
	r := [][]float64{
		{1.0, 2.0, 2.0},
		{-4.0, 0.0, 1.0},
		{5.0, 3.0, -10.0},
	}
	result := DataToMatrix(r)

	b := [][]float64{
		{1.0, 5.0, 3.0},
		{-3.0, 5.0, -4.0},
		{3.0, 2.0, 1.0},
	}
	B := DataToMatrix(b)

	e := [][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	expected := DataToMatrix(e)

	result.HadamardProductSelf(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix Hadamard product error: Result != Expected")
	}
}

func TestT(t *testing.T) {
	m := [][]float64{
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	M := DataToMatrix(m)

	e := [][]float64{
		{12.0, 15.0},
		{0.0, 6.0},
		{-4.0, -10.0},
	}
	expected := DataToMatrix(e)

	result := M.T()

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
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
	expected := DataToMatrix(e)

	A := Zero(rows, columns)
	A.Slice2Matrix(slc)

	result := A

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Slice2Matrix error: Result != Expected")
	}
}

func TestForEach(t *testing.T) {
	m := [][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	M := DataToMatrix(m)

	e := [][]float64{
		{1.0, 100.0, 36.0},
		{144.0, 0.0, 16.0},
		{225.0, 36.0, 100.0},
	}
	expected := DataToMatrix(e)

	result := M.ForEach(square)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("ForEach error: Result != Expected")
	}
}

func TestForEachInner(t *testing.T) {
	r := [][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	}
	result := DataToMatrix(r)

	e := [][]float64{
		{1.0, 100.0, 36.0},
		{144.0, 0.0, 16.0},
		{225.0, 36.0, 100.0},
	}
	expected := DataToMatrix(e)

	result.ForEachSelf(square)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("ForEachInner error: Result != Expected")
	}
}

func square(x float64) float64 {
	return x * x
}

func TestMatrix2Vector(t *testing.T) {
	m := [][]float64{{5.}}
	M := DataToMatrix(m)

	result, err := Matrix2Vector(M, 10)
	if err != nil {
		t.Error(err)
	}

	e := [][]float64{{0.}, {0.}, {0.}, {0.}, {0.}, {1.}, {0.}, {0.}, {0.}, {0.}}
	expected := DataToMatrix(e)

	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Incorrect converting matrix to vector")
	}
}

func TestNum(t *testing.T) {
	m := [][]float64{{5.}}
	M := DataToMatrix(m)

	result := Num(M)

	expected := 5.

	if math.Abs(result-expected) > 1e-9 {
		t.Errorf("Incorrect converting matrix to float64")
	}

}

func TestVec2Dig(t *testing.T) {
	m := [][]float64{{0.33333}, {0.122345}, {0.0000001}, {0.0030003}, {7.65764756}, {10.8478374}, {0.06565}, {0.111}, {0.1212}, {0.89}}
	M := DataToMatrix(m)
	result := Vec2Dig(M)

	expected := 5

	if result != expected {
		t.Errorf("Incorrect converting vector to digit")
	}

}
