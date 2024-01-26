package matrix

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type Matrix struct {
	columns int
	rows    int
	arr     [][]float64
}

func (M *Matrix) GetRows() int {
	return M.rows
}

func (M *Matrix) GetColumns() int {
	return M.rows
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

func RandMatrix(rows, columns int) *Matrix {
	if rows <= 0 || columns <= 0 {
		panic("RandMatrix: row <= 0 or columns <= 0")
	}

	matrix := EmptyMatrix(rows, columns)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			matrix.arr[i][j] = rand.NormFloat64()
		}
	}

	return matrix
}

func EmptyMatrix(rows, columns int) *Matrix {

	if rows <= 0 || columns <= 0 {
		panic("EmptyMatrix: row <= 0 or columns <= 0")
	}

	arr := make([][]float64, rows)
	for i := range arr {
		arr[i] = make([]float64, columns)
	}

	return &Matrix{
		columns: columns,
		rows:    rows,
		arr:     arr,
	}
}

func (M *Matrix) Show() {
	for _, row := range M.arr {
		fmt.Println(row)
	}
}

func (A *Matrix) Dot(B *Matrix) (*Matrix, error) {
	if A.columns != B.rows {
		return &Matrix{}, errors.New("Dot: can't multiply matrices")
	}

	wg := new(sync.WaitGroup)
	wg.Add(A.rows * B.columns)

	C := EmptyMatrix(A.rows, B.columns)

	for i := 0; i < A.rows; i++ {
		for j := 0; j < B.columns; j++ {

			go func(i, j int) {
				defer wg.Done()

				sum := float64(0)

				for k := 0; k < A.columns; k++ {
					sum += A.arr[i][k] * B.arr[k][j]
				}
				C.arr[i][j] = sum
			}(i, j)
		}
	}

	wg.Wait()

	return C, nil

}
