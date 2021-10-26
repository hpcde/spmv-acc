package mtx

type DlLinks struct {
	Matlab           string // matlab format
	RutherfordBoeing string // Rutherford Boeing format
	MatrixMarket     string // Matrix Market format
}

// MatrixMeta described the base information of a matrix
type MatrixMeta struct {
	ID      string
	Name    string
	Group   string
	Rows    int64
	Cols    int64
	NNZ     int64 // we use int64, nnz can be very large (can be larger than 2^32)
	Kind    string
	Date    string
	DlLinks DlLinks
}
