package main

type DlLinks struct {
	Matlab           string // matlab format
	RutherfordBoeing string // Rutherford Boeing format
	MatrixMarket     string // Matrix Market format
}
type MatrixMeta struct {
	ID      string
	Name    string
	Group   string
	Rows    int
	Cols    int
	NNZ     int
	Kind    string
	Date    string
	DlLinks DlLinks
}
