package conv

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

type TpIndex int32
type TpFloat float64

type Entry struct {
	row   TpIndex
	col   TpIndex
	value TpFloat
}

type Entries []Entry

func (e Entries) Len() int {
	return len(e)
}

func (e Entries) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

func (e Entries) Less(i, j int) bool {
	if e[i].row != e[j].row {
		return e[i].row < e[j].row
	}
	return e[i].col < e[j].col
}

// MMHeader contains the header descriptor of matrix market
type MMHeader struct {
	numRows      TpIndex
	numColumns   TpIndex
	numNonZeroes TpIndex // nnz in file body.
	pattern      bool
	hermitian    bool
	complex      bool
	symmetric    bool
}

type MatrixMarket struct {
	header MMHeader
	data   Entries
}

func (mm *MatrixMarket) Sort() {
	sort.Sort(mm.data)
}

// read matrix market file and convert to csr binary file.
func conv(filepath string) (*MatrixMarket, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}

	// read header
	reader := bufio.NewReader(f)
	header, lineCounter, err := parseHeader(reader, filepath)
	if err != nil {
		return nil, err
	}

	reserve := header.numNonZeroes
	if header.symmetric || header.hermitian {
		reserve *= 2
	}

	// read body
	data := make([]Entry, 0, reserve)
	var read, nnzDia TpIndex
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return nil, err
			}
		}
		lineCounter++
		if entry1, entry2, err := parseBodyLine(line, filepath, *header, lineCounter, &nnzDia, &read); err != nil {
			return nil, err
		} else {
			// add elements to results list.
			if entry1 != nil {
				data = append(data, *entry1)
			}
			if entry2 != nil {
				data = append(data, *entry2)
			}
		}
	}
	// assert read count.
	if read+nnzDia != reserve {
		return nil, fmt.Errorf("mismatch non-zeros number, expect %d, but got %d", reserve, read)
	}
	return &MatrixMarket{header: *header, data: data}, nil
}

func parseHeader(reader *bufio.Reader, filename string) (*MMHeader, TpIndex, error) {
	// parse the first line
	line, err := reader.ReadString('\n')
	if err != nil {
		return nil, 0, err
	}

	if len(line) < 32 || line[:32] != "%%MatrixMarket matrix coordinate" {
		return nil, 0, errors.New("can only read MatrixMarket format that is in coordinate form")
	}

	tokens := strings.Split(strings.TrimSpace(line), " ")
	if len(tokens) < 5 {
		return nil, 0, errors.New("bad market matrix  header")
	}

	header := MMHeader{
		numRows:      0,
		numColumns:   0,
		numNonZeroes: 0,
		pattern:      false,
		hermitian:    false,
		complex:      false,
		symmetric:    false,
	}

	if tokens[3] == "pattern" {
		header.pattern = true
	} else if tokens[3] == "complex" {
		header.complex = true
	} else if tokens[3] != "real" && tokens[3] != "integer" { // we treat integer type as real type.
		return &header, 0, fmt.Errorf("matrix market data type does not match matrix format on filename: %s", filename)
	}

	if tokens[4] == "general" {
		header.symmetric = false
		header.hermitian = false
	} else if tokens[4] == "symmetric" {
		header.symmetric = true
	} else if tokens[4] == "Hermitian" {
		header.hermitian = true
	} else {
		return nil, 0, errors.New("can only read MatrixMarket format that is either symmetric, general or hermitian")
	}

	var lineCounter TpIndex
	lineCounter = 0
	// skip comments and read metadata.
	for {
		line2, err := reader.ReadString('\n')
		lineCounter++
		if err != nil {
			return nil, 0, err
		}
		if len(line2) == 0 {
			continue
		}

		// skip header
		if line2[0] == '%' {
			continue
		}

		lineBuffer := bytes.Buffer{}
		lineBuffer.WriteString(line2)

		var rows, columns, nnz TpIndex
		if n, err := fmt.Fscanln(&lineBuffer, &rows, &columns, &nnz); err != nil {
			return nil, 0, err
		} else if n == 0 {
			return nil, 0, fmt.Errorf("failed to read matrix market header from `%s`", filename)
		}

		header.numRows = rows
		header.numColumns = columns
		header.numNonZeroes = nnz
		// fmt.Println("Read matrix header")
		// fmt.Println("rows: ", rows, " columns: ", columns, " nnz: ", nnz)
		break
	}
	return &header, lineCounter, nil
}

// parseBodyLine parses a line of body of matrix market file
// returning at most 2 elements (If the matrix is symmetric or hermitian, it may return 2 elements).
func parseBodyLine(line, filename string, header MMHeader, lineCounter TpIndex, nnzDia, read *TpIndex) (*Entry, *Entry, error) {
	if len(line) == 0 {
		return nil, nil, nil
	}
	if line[0] == '%' {
		return nil, nil, nil
	}

	r, c, value, err := parseLineValue(line, header.pattern)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read data at line %d from matrix market file %s", lineCounter, filename)
	}

	if r > header.numRows {
		return nil, nil, fmt.Errorf("row index out of bounds at line %d in matrix market file %s", lineCounter, filename)
	}
	if c > header.numColumns {
		return nil, nil, fmt.Errorf("column index out of bounds at line %d  in matrix market file %s", lineCounter, filename)
	}

	entry1 := Entry{row: r - 1, col: c - 1, value: value}
	*read++
	if (header.symmetric || header.hermitian) && r == c {
		*nnzDia++
	}

	if (header.symmetric || header.hermitian) && r != c {
		entry2 := Entry{row: c - 1, col: r - 1, value: value} // exchange row and column.
		*read++
		return &entry1, &entry2, nil
	} else {
		return &entry1, nil, nil
	}
}

func parseLineValue(line string, pattern bool) (TpIndex, TpIndex, TpFloat, error) {
	lineBuffer := bytes.Buffer{}
	lineBuffer.WriteString(strings.TrimSpace(line))

	var row, col TpIndex
	var value TpFloat

	var n int
	var err error
	if pattern {
		n, err = fmt.Fscanln(&lineBuffer, &row, &col)
		value = 1.0
	} else {
		n, err = fmt.Fscanln(&lineBuffer, &row, &col, &value)
	}

	// check error
	if err != nil {
		return 0, 0, 0.0, err
	} else if n == 0 {
		return 0, 0, 0.0, fmt.Errorf("failed to read matrix market header")
	}
	return row, col, value, nil
}
