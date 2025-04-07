package conv

import (
	"encoding/binary"
	"errors"
	"flag"
	"github.com/genshen/cmds"
	"io"
	"os"
)

var convCommand = &cmds.Command{
	Name:        "conv",
	Summary:     "conv matrix market format to csr binary format",
	Description: "conv matrix market format to csr binary format",
	CustomFlags: false,
	HasOptions:  true,
}

func init() {
	c := Conv{}
	convCommand.Runner = &c
	fs := flag.NewFlagSet("conv", flag.ContinueOnError)
	convCommand.FlagSet = fs
	convCommand.FlagSet.StringVar(&(c.input), "mm", "", `matrix market path.`)
	convCommand.FlagSet.StringVar(&(c.output), "o", "", `filepath of binary output.`)
	convCommand.FlagSet.BoolVar(&(c.litterEndian), "l", true, `byte order (litterEndian or BigEndian).`)
	convCommand.FlagSet.Usage = convCommand.Usage // use default usage provided by cmds.Command.
	cmds.AllCommands = append(cmds.AllCommands, convCommand)
}

type Conv struct {
	input         string
	output       string
	litterEndian bool
}

func (c *Conv) PreRun() error {
	if c.output == "" {
		c.output = c.input + ".bin"
	}
	return nil
}

func (c *Conv) Run() error {
	mm, err := conv(c.input)
	if err != nil {
		return err
	}
	// to CSR
	mm.Sort()
	rowPtr := make([]TpIndex, mm.header.numRows+1)
	colIndex := make([]TpIndex, len(mm.data))
	nonZeros := make([]TpFloat, len(mm.data))
	for i, ele := range mm.data {
		if ele.row >= mm.header.numRows {
			return errors.New("out of range when converting to CSR")
		}
		colIndex[i] = ele.col
		nonZeros[i] = ele.value
		rowPtr[ele.row+1]++
	}
	var i TpIndex = 0
	for ; i < mm.header.numRows; i++ {
		nnzThisRow := rowPtr[i+1]
		rowPtr[i+1] = rowPtr[i] + nnzThisRow
	}

	// write binary
	if outfile, err := os.Create(c.output); err != nil {
		return err
	} else {
		var byteOrder binary.ByteOrder = binary.LittleEndian
		if !c.litterEndian {
			byteOrder = binary.BigEndian
		}

		// write binary header to the binary file
		if err := writeBinHeader(outfile, byteOrder, mm.header.val_type); err != nil {
			return err
		}

		// write CSR matrix
		var nnz TpIndex = (TpIndex)(len(mm.data))
		if err := binary.Write(outfile, byteOrder, &(mm.header.numRows)); err != nil {
			return err
		}
		if err := binary.Write(outfile, byteOrder, &(mm.header.numColumns)); err != nil {
			return err
		}
		if err := binary.Write(outfile, byteOrder, &(nnz)); err != nil {
			return err
		}
		if err := binary.Write(outfile, byteOrder, rowPtr); err != nil {
			return err
		}
		if err := binary.Write(outfile, byteOrder, colIndex); err != nil {
			return err
		}
		// write CSR matrix: the value part
		err := writeCsrMatrixValues(outfile, byteOrder, mm.header.val_type, nonZeros)
		if err != nil {
			return err
		}
	}
	return nil
}

func writeBinHeader(outfile io.Writer, byteOrder binary.ByteOrder, valTp MMValueType) error {
	// write magic number
	var magicNumber int32 = 0x20211015
	if err := binary.Write(outfile, byteOrder, &(magicNumber)); err != nil {
		return err
	}
	// write format version
	var formatVersion int32 = 0x2
	if err := binary.Write(outfile, byteOrder, &(formatVersion)); err != nil {
		return err
	}
	// write value type
	if err := binary.Write(outfile, byteOrder, &(valTp)); err != nil {
		return err
	}
	return nil
}

func writeCsrMatrixValues(outfile io.Writer, byteOrder binary.ByteOrder, valTp MMValueType, nonZeros []TpFloat) error {
	if valTp == MMValueTypeUnknown {
		return errors.New("unknown value type in mm file")
	}
	if valTp == MMValueTypePattern {
		return nil
	}

	if valTp == MMValueTypeInt {
		nonZerosInt := make([]TpValInt32, len(nonZeros))
		for i, zs := range nonZeros {
			nonZerosInt[i] = TpValInt32(zs)
		}
		if err := binary.Write(outfile, byteOrder, nonZerosInt); err != nil {
			return err
		}
		return nil
	} else {
		if err := binary.Write(outfile, byteOrder, nonZeros); err != nil {
			return err
		}
		return nil
	}
}
