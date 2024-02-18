package data_frame

import (
	"NN/pkg/matrix"
	"math"
	"testing"
)

// Запуск только данного теста производится из текущей директории.
// Тест можно запустить находясь в корневой директории с помощью запуска всех тестов (go test ./...)
func TestReadCSV(t *testing.T) {

	res, err := ReadCSV("../../data/TestReadCSV.csv", 4)
	if err != nil {
		t.Error(err)
	}

	x1 := matrix.Zeros(5, 1)
	x1.Slice2Matrix([]float64{2., 3., 4., 5., 6.})

	x2 := matrix.Zeros(5, 1)
	x2.Slice2Matrix([]float64{2., 3., 4., 5., 7.})

	x3 := matrix.Zeros(5, 1)
	x3.Slice2Matrix([]float64{2., 3., 4., 5., 8.})

	x4 := matrix.Zeros(5, 1)
	x4.Slice2Matrix([]float64{2., 3., 4., 5., 9.})

	rows := []*rowDataFrame{
		{
			y: 1.,
			x: x1,
		},
		{
			y: 1.,
			x: x2,
		},
		{
			y: 1.,
			x: x3,
		},
		{
			y: 1.,
			x: x4,
		},
	}

	expected := DataFrame{
		Data: rows,
	}

	epsilon := float64(1e-9)

	if len(expected.Data) != len(res.Data) {
		t.Errorf("Expected number of records not Equal number of result records %v != %v", len(expected.Data), len(res.Data))
	}

	for i := 0; i < 4; i++ {
		if math.Abs(expected.Data[i].y-res.Data[i].y) > epsilon {
			t.Errorf("Expected target %v record not equal result target %v records %f != %f", i, i, expected.Data[i].y, res.Data[i].y)
		}
		if !matrix.IsMatrixesEqual(expected.Data[i].x, res.Data[i].x) {
			t.Errorf("Expected features %v record not equal result features %v record", i, i)
		}
	}

}
