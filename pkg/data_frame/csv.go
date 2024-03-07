package data_frame

import (
	"NN/pkg/matrix"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

//ссылка на набор данных
//https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv

var (
	sep   string = "," // разделитель в ячейке между данными
	comma rune   = ';' // разделитель между ячейками
)

// Считывает датасет из CSV и возвращает датафрейм с этими же данными.
// Функция может работать только с CSV файлами в которых содержатся только числа.
// Принимает название CSV файла и предполагаемый размер датасета.
func ReadCSV(filename string, capacity int) (DataFrame, error) {
	// открытие csv файла
	file, err := os.Open(filename)
	if err != nil {
		return DataFrame{}, fmt.Errorf("error opening a csv file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = comma

	// cap + 2, чтобы был запас на всякий случай
	data := make([]*rowDataFrame, 0, capacity+2)

	minLenRecord := math.MaxInt
	maxLenRecord := math.MinInt

	// чтение файла
	for i := 0; ; i++ {
		record, err := reader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			return DataFrame{}, fmt.Errorf("error reading a csv file: %v", err)
		}

		// первая строчка отвечает за названия признаков
		if i == 0 {
			continue
		}

		// CSV файлы содержат данные одного наблюдения в самой первой ячейке
		recordDataString := strings.Split(record[0], sep)

		minLenRecord = min(minLenRecord, len(recordDataString))
		maxLenRecord = max(maxLenRecord, len(recordDataString))

		recordDataFloat64 := make([]float64, len(recordDataString))

		// преобразуем элементы массива recordDataString типа string элементы массива recordDataFloat64 типа float64
		for i := 0; i < len(recordDataString); i++ {
			num, err := strconv.ParseFloat(recordDataString[i], 64)
			if err != nil {
				return DataFrame{}, fmt.Errorf("incorrect csv file format: %v", err)
			}

			recordDataFloat64[i] = num
		}

		// добавляем в новую строку датафрейма целевую переменную и признаки
		//y := recordDataFloat64[0]
		y := matrix.Zero(1, 1)
		y.Slice2Matrix([]float64{recordDataFloat64[0]})
		x := matrix.Zero(len(recordDataFloat64)-1, 1)
		x.Slice2Matrix(recordDataFloat64[1:])

		rowDf := &rowDataFrame{
			x: x,
			y: y,
		}

		data = append(data, rowDf)

	}

	if maxLenRecord != minLenRecord {
		return DataFrame{}, errors.New("unequal size of records in csv")
	}

	return DataFrame{
		Data: data,
	}, nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
