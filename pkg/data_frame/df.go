package data_frame

import (
	"NN/pkg/matrix"
)

type rowDataFrame struct {
	x matrix.Matrix
	y float64
}

type DataFrame struct {
	Data []*rowDataFrame
}
