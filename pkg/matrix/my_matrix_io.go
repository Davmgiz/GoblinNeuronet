package matrix

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// writeMatrix записывает данные нескольких матриц (структур myMatrix) в поток вывода, реализующий интерфейс io.Writer,
// если запись не удалась функция возвращает ошибку.
// Функция форматирует данные каждой матрицы следующим образом:
// сначала записывается общее количество матриц в переданном слайсе,
// с новой строки для каждой матрицы записывается ее размерность (количество строк и столбцов),
// с новой строки данные самой матрицы.
// Элементы каждой строки матрицы разделяются пробелом, а строки матрицы - переводами строк.
// Функция не открывает и не закрывает поток вывода, управление потоком
// должно осуществляться вне этой функции.
func writeMatrixes(writer io.Writer, matrixes []*myMatrix) error {
	// записываем количество матриц
	_, err := fmt.Fprintf(writer, "%d\n", len(matrixes))
	if err != nil {
		return err
	}

	for _, matrix := range matrixes {
		// записываем размерность матрицы
		_, err := fmt.Fprintf(writer, "%d %d\n", matrix.rows, matrix.columns)
		if err != nil {
			return err
		}

		// записываем элементы матрицы
		for r := 0; r < matrix.rows; r++ {
			for c := 0; c < matrix.columns; c++ {
				_, err := fmt.Fprintf(writer, "%f ", matrix.data[r][c])
				if err != nil {
					return err
				}
			}

			// добавляем перенос строки после каждой строки матрицы
			_, err = fmt.Fprintln(writer)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// readMatrix возвращает слайс из указателей на матрицы (структуры myMatrix),
// прочитанные из файла, и ошибку, если она возникла.
// Функция читает данные, начиная с текущей позиции в потоке сканера.
// Каждый блок матрицы в файле должен быть отформатирован следующим образом:
// сначала идет число - общее количество матриц,
// затем с новой строки для каждой матрицы должна быть ее размерность (количество строк и столбцов),
// с новой строки данные самой матрицы.
// Элементы каждой строки матрицы разделяются пробелом, а строки матрицы - переводами строк.
// Функция принимает уже инициализированный сканер структуры *bufio.Scanner.
// Функция не открывает и не закрывает поток вывода, управление потоком
// должно осуществляться вне этой функции.
func readMatrixes(scanner *bufio.Scanner) ([]*myMatrix, error) {
	matrixes := []*myMatrix{}

	// Считываем количество матриц
	if scanner.Scan() {
		lenMatrixesStr := scanner.Text()
		lenMatrixesInt, err := strconv.Atoi(lenMatrixesStr)
		if err != nil {
			return nil, err
		}

		for i := 0; i < lenMatrixesInt; i++ {

			//считываем размерность матрицы
			if !scanner.Scan() {
				return nil, fmt.Errorf("expected matrix dimensions, found EOF")
			}
			dimension := strings.Fields(scanner.Text())
			if len(dimension) != 2 {
				return nil, fmt.Errorf("incorrect matrix dimension")
			}

			// количество строк
			rows, err := strconv.Atoi(dimension[0])
			if err != nil {
				return nil, err
			}

			// количество столбцов
			columns, err := strconv.Atoi(dimension[1])
			if err != nil {
				return nil, err
			}

			// возвращаемый слайс
			matrix := make([][]float64, rows)

			for r := 0; r < rows; r++ {

				// считывание строки
				if !scanner.Scan() {
					return nil, fmt.Errorf("unexpected end of file while reading matrix data")
				}

				values := strings.Fields(scanner.Text())
				if len(values) != columns {
					return nil, fmt.Errorf("incorrect number of columns in matrix")
				}

				matrix[r] = make([]float64, columns)
				for c := 0; c < columns; c++ {

					// преобразуем в вещественное число
					num, err := strconv.ParseFloat(values[c], 64)
					if err != nil {
						return nil, err
					}

					matrix[r][c] = num
				}
			}

			matrixes = append(matrixes, &myMatrix{
				rows:    rows,
				columns: columns,
				data:    matrix,
			})
		}
	} else if err := scanner.Err(); err != nil {
		return nil, err
	} else {
		return nil, fmt.Errorf("unexpected end of file while reading the number of matrices")
	}

	return matrixes, nil
}
