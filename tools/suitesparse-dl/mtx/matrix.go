package mtx

type DlLinks struct {
	Matlab           string `json:"matlab"` // matlab format
	RutherfordBoeing string `json:"rb"`     // Rutherford Boeing format
	MatrixMarket     string `json:"mm"`     // Matrix Market format
}

// MatrixMeta described the base information of a matrix
type MatrixMeta struct {
	ID      string  `json:"id"`
	Name    string  `json:"name"`
	Group   string  `json:"group"`
	Rows    int64   `json:"rows"`
	Cols    int64   `json:"cols"`
	NNZ     int64   `json:"nnz"` // we use int64, nnz can be very large (can be larger than 2^32)
	Kind    string  `json:"kind"`
	Date    string  `json:"date"`
	DlLinks DlLinks `json:"links"`
}
