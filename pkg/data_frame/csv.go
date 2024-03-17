package data_frame

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/Davmgiz/GoblinNeuronet/pkg/matrix"
)

// Настройка разделителей формата csv.
var (
	sep   string = "," // разделитель в ячейке между данными
	comma rune   = ';' // разделитель между ячейками
)

// ReadCSV возвращает считанный датафрейм из csv (структуру DataFrame) и ошибку.
// Функция принимает название CSV файла и предполагаемый размер датасета.
// Функция может работать только с CSV файлами в которых содержатся только числа.
// Возвращает ошибку, если: не удалось открыть файл,
// возникли ошибки при чтении, в csv формате содержатся не числа,
// данные разных наблюдений имеют разную длину по столбцам.
func ReadCSV(filename string, capacity int) (DataFrame, error) {
	// открытие csv файла
	file, err := os.Open(filename)
	if err != nil {
		return DataFrame{}, fmt.Errorf("error opening a csv file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = comma

	// capacity + 2, чтобы был запас на всякий случай
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

	// Контролируем чтобы все наблюдения были равными по длине
	if maxLenRecord != minLenRecord {
		return DataFrame{}, errors.New("unequal size of records in csv")
	}

	return DataFrame{
		Data: data,
	}, nil
}

// max вспомогательная функция для ReadCSV.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min вспомогательная функция для ReadCSV.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
