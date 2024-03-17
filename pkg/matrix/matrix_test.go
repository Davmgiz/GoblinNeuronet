package matrix

import (
	"math"
	"testing"
)

// TestRandMatrix проверяет генерацию случайных матриц (структуры Matrix).
// Тест убеждается, что матрицы содержат достаточное количество уникальных элементов,
// чтобы соответствовать ожидаемому распределению случайных чисел. Для этого измеряется
// коэффициент повторяемости элементов в матрицах разных размеров (3 на 3 и 4 на 4). Если коэффициент
// повторяемости ниже заданного порога (threshold), тест считается неудачным, поскольку это может
// указывать на недостаточную случайность или неправильную работу функции RandMatrix.
func TestRandMatrix(t *testing.T) {

	// 2 проверяющие матрицы.
	matrix1 := RandMatrix(3, 3)
	matrix2 := RandMatrix(4, 4)

	// Если коэф повторяемости элементов матрицы ниже данной границы, то такой рандом нас не устраивает.
	threshold := 0.8

	// Подсчет коэф для обеих матриц
	repeatedMatrix1 := float64(countUniqueElements(matrix1)) / float64(matrix1.matrix.rows*matrix1.matrix.columns)
	repeatedMatrix2 := float64(countUniqueElements(matrix2)) / float64(matrix2.matrix.rows*matrix2.matrix.columns)

	if repeatedMatrix1 < threshold {
		t.Errorf("Too many repeated elements in matrix1. Expect %f, got %f", threshold, repeatedMatrix1)
	}

	if repeatedMatrix2 < threshold {
		t.Errorf("Too many repeated elements in matrix2. Expect %f, got %f", threshold, repeatedMatrix2)
	}

}

// TestZero проверяет нулевую матрицу данной размерности,
// так же функция перехватывает панику при нулевой или отрицательной размерности матрицы
func TestZero(t *testing.T) {
	result := Zero(2, 4)

	expected := DataToMatrix([][]float64{
		{0., 0., 0., 0.},
		{0., 0., 0., 0.},
	})

	// проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix Zero error: Result != Expected")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic on negative dimensions")
		}
	}()

	_ = Zero(-1, 0)

}

// TestDot проверяет умножение матриц (структур Matrix)
func TestDot(t *testing.T) {
	A := DataToMatrix([][]float64{
		{1., 2.},
		{3., 4.},
		{5., 6.},
	})

	B := DataToMatrix([][]float64{
		{1., 2., 3., 10.},
		{4., 5., 6., 9.},
	})

	expected := DataToMatrix([][]float64{
		{9., 12., 15., 28.},
		{19., 26., 33., 66.},
		{29., 40., 51., 104.},
	})

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

// TestAdd проверяет сложение матриц (структур Matrix)
func TestAdd(t *testing.T) {
	A := DataToMatrix([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	})

	B := DataToMatrix([][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	})

	expected := DataToMatrix([][]float64{
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
	})

	result := A.Add(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix additional error: Result != Expected")
	}
}

// TestAddInner проверяет сложение inPlace матриц (структур Matrix)
func TestAddInner(t *testing.T) {
	result := DataToMatrix([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	})

	B := DataToMatrix([][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	})

	expected := DataToMatrix([][]float64{
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
		{10.0, 10.0, 10.0},
	})

	result.AddInPlace(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix additional error: Result != Expected")
	}
}

// TestSub проверяет разность матриц (структур Matrix)
func TestSub(t *testing.T) {
	A := DataToMatrix([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	})

	B := DataToMatrix([][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	})

	expected := DataToMatrix([][]float64{
		{-8.0, -6.0, -4.0},
		{-2.0, 0.0, 2.0},
		{4.0, 6.0, 8.0},
	})

	result := A.Sub(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix subtraction error: Result != Expected")
	}
}

// TestSubInner проверяет разность inPlace матриц (структур Matrix)
func TestSubInner(t *testing.T) {
	result := DataToMatrix([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	})

	B := DataToMatrix([][]float64{
		{9.0, 8.0, 7.0},
		{6.0, 5.0, 4.0},
		{3.0, 2.0, 1.0},
	})

	expected := DataToMatrix([][]float64{
		{-8.0, -6.0, -4.0},
		{-2.0, 0.0, 2.0},
		{4.0, 6.0, 8.0},
	})

	result.SubInPlace(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix subtraction error: Result != Expected")
	}
}

// TestHadamardProduct проверяет адамарное произведение матриц (структур Matrix)
func TestHadamardProduct(t *testing.T) {
	A := DataToMatrix([][]float64{
		{1.0, 2.0, 2.0},
		{-4.0, 0.0, 1.0},
		{5.0, 3.0, -10.0},
	})

	B := DataToMatrix([][]float64{
		{1.0, 5.0, 3.0},
		{-3.0, 5.0, -4.0},
		{3.0, 2.0, 1.0},
	})

	expected := DataToMatrix([][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	})

	result := A.HadamardProduct(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix Hadamard product error: Result != Expected")
	}
}

// TestHadamardProductInner проверяет адамарное произведение inPlace матриц (структур Matrix)
func TestHadamardProductInner(t *testing.T) {
	result := DataToMatrix([][]float64{
		{1.0, 2.0, 2.0},
		{-4.0, 0.0, 1.0},
		{5.0, 3.0, -10.0},
	})

	B := DataToMatrix([][]float64{
		{1.0, 5.0, 3.0},
		{-3.0, 5.0, -4.0},
		{3.0, 2.0, 1.0},
	})

	expected := DataToMatrix([][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	})

	result.HadamardProductInPlace(B)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix Hadamard product error: Result != Expected")
	}
}

// TestT проверяет транспонирование матрицы (структуры Matrix)
func TestT(t *testing.T) {
	M := DataToMatrix([][]float64{
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	})

	expected := DataToMatrix([][]float64{
		{12.0, 15.0},
		{0.0, 6.0},
		{-4.0, -10.0},
	})

	result := M.T()

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Matrix transposition error: Result != Expected")
	}
}

// TestSlice2Matrix проверяет преобразование слайса []float64 в матрицу (структуру Matrix)
func TestSlice2Matrix(t *testing.T) {
	rows := 2
	columns := 3
	slc := []float64{12.0, 0.0, -4.0, 15.0, 6.0, -10.0}

	expected := DataToMatrix([][]float64{
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	})

	A := Zero(rows, columns)
	A.Slice2Matrix(slc)

	result := A

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("Slice2Matrix error: Result != Expected")
	}
}

// TestForEach поэлементное применение функции сигнатуры f func(float64) float64 для матрицы (структуры Matrix)
func TestForEach(t *testing.T) {
	M := DataToMatrix([][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	})

	expected := DataToMatrix([][]float64{
		{1.0, 100.0, 36.0},
		{144.0, 0.0, 16.0},
		{225.0, 36.0, 100.0},
	})

	result := M.ForEach(square)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("ForEach error: Result != Expected")
	}
}

// TestForEachInner проверяет поэлементное применение функции сигнатуры f func(float64) float64
// матриц inPlace (структур Matrix)
func TestForEachInner(t *testing.T) {
	result := DataToMatrix([][]float64{
		{1.0, 10.0, 6.0},
		{12.0, 0.0, -4.0},
		{15.0, 6.0, -10.0},
	})

	expected := DataToMatrix([][]float64{
		{1.0, 100.0, 36.0},
		{144.0, 0.0, 16.0},
		{225.0, 36.0, 100.0},
	})

	result.ForEachInPlace(square)

	//проверка на равность
	if !IsMatrixesEqual(result, expected) {
		t.Errorf("ForEachInner error: Result != Expected")
	}
}

// square вспомогательная функция для функций TestForEachInner и TestForEach.
// Имеет сигнатуру f func(float64) float64.
func square(x float64) float64 {
	return x * x
}

// TestMatrix2Vector проверяет кодирование единственного элемента матрицы в вектор.
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

// TestNum проверяет извлечение единственного элемента из матрицы размерностью 1 на 1
func TestNum(t *testing.T) {
	M := DataToMatrix([][]float64{{5.}})

	result := Num(M)

	expected := 5.

	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Incorrect converting matrix to float64")
	}

}

// TestVec2Dig возвращает индекс наибольшего элемента вектора (матрицы n на 1)
func TestVec2Dig(t *testing.T) {
	m := [][]float64{{0.33333}, {0.122345}, {0.0000001}, {0.0030003}, {7.65764756}, {10.8478374}, {0.06565}, {0.111}, {0.1212}, {0.89}}
	M := DataToMatrix(m)
	result := Vec2Num(M)

	expected := 5

	if result != expected {
		t.Errorf("Incorrect converting vector to digit")
	}

}
