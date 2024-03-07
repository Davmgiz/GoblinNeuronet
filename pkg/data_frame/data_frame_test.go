package data_frame

import (
	"NN/pkg/matrix"
	"testing"
)

// Запуск только данного теста производится из текущей директории.
// Тест можно запустить находясь в корневой директории с помощью запуска всех тестов (go test ./...)
func TestReadCSV(t *testing.T) {
	res, err := ReadCSV("../../data/data_frame_test/test_read_csv.csv", 4)
	if err != nil {
		t.Error(err)
	}

	x1 := matrix.Zero(5, 1)
	x1.Slice2Matrix([]float64{2., 3., 4., 5., 6.})

	x2 := matrix.Zero(5, 1)
	x2.Slice2Matrix([]float64{2., 3., 4., 5., 7.})

	x3 := matrix.Zero(5, 1)
	x3.Slice2Matrix([]float64{2., 3., 4., 5., 8.})

	x4 := matrix.Zero(5, 1)
	x4.Slice2Matrix([]float64{2., 3., 4., 5., 9.})

	y := matrix.Zero(1, 1)
	y.Slice2Matrix([]float64{1.})

	rows := []*rowDataFrame{
		{
			y: y,
			x: x1,
		},
		{
			y: y,
			x: x2,
		},
		{
			y: y,
			x: x3,
		},
		{
			y: y,
			x: x4,
		},
	}

	expected := DataFrame{
		Data: rows,
	}

	if len(expected.Data) != len(res.Data) {
		t.Errorf("Expected number of records not Equal number of result records %v != %v", len(expected.Data), len(res.Data))
	}

	for i := 0; i < 4; i++ {
		if !matrix.IsMatrixesEqual(expected.Data[i].y, res.Data[i].y) {
			t.Errorf("Expected target %v record not equal result target %v records", i, i)
		}
		if !matrix.IsMatrixesEqual(expected.Data[i].x, res.Data[i].x) {
			t.Errorf("Expected features %v record not equal result features %v record", i, i)
		}
	}

}

func TestNormalization(t *testing.T) {
	result, err := ReadCSV("../../data/data_frame_test/test_normalization.csv", 10)
	if err != nil {
		t.Error(err)
	}

	result.Normalization()

	expected, err := ReadCSV("../../data/data_frame_test/test_normalization_expected.csv", 10)
	if err != nil {
		t.Error(err)
	}

	if expected.Length() != result.Length() {
		t.Errorf("Dont equal length expected data frame and result data frame")
	}

	for i := 0; i < result.Length(); i++ {
		if !matrix.IsMatrixesEqual(expected.Data[i].x, result.Data[i].x) {
			t.Errorf("Dont equal expected data frame and result data frame")
		}
	}
}
