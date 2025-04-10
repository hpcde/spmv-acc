package main

import (
	"errors"
	"flag"
	"log"

	"github.com/genshen/cmds"
	_ "suitesparse-dl/batch-gen"
	_ "suitesparse-dl/conv"
	_ "suitesparse-dl/dl"
	_ "suitesparse-dl/fetch"
	_ "suitesparse-dl/list"
)

func main() {
	cmds.SetProgramName("suitesparse-dl")
	if err := cmds.Parse(); err != nil {
		if err == flag.ErrHelp {
			return
		}
		// skip error in sub command parsing, because the error has been printed.
		if !errors.Is(err, &cmds.SubCommandParseError{}) {
			log.Println(err)
		}
	}
}
