package list

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/genshen/cmds"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"suitesparse-dl/mtx"
)

var listCommand = &cmds.Command{
	Name:        "list",
	Summary:     "list matrix data in a directory",
	Description: "list all matrix information in a given directory downloaded from suitesparse collection.",
	CustomFlags: false,
	HasOptions:  true,
}

func init() {
	l := List{}
	listCommand.Runner = &l
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	listCommand.FlagSet = fs
	listCommand.FlagSet.StringVar(&(l.dir), "d", "./", `director of .mtx file stored.`)
	listCommand.FlagSet.Usage = listCommand.Usage // use default usage provided by cmds.Command.
	cmds.AllCommands = append(cmds.AllCommands, listCommand)
}

type List struct {
	url string
	dir string
}

func (l *List) PreRun() error {
	return nil // if error != nil, function Run will be not execute.
}

func (l *List) Run() error {
	// search all .mtx file under path l.dir and save path in a set.
	mtxPathMap := make(map[string]string)

	if err := filepath.Walk(l.dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if filepath.Ext(path) == ".mtx" {
			mtxName := strings.TrimSuffix(info.Name(), ".mtx")
			mtxPathMap[mtxName] = path
		}
		return nil
	}); err != nil {
		return err
	}

	// read metadata.json and link matrix to file path
	jsonBytes, err := ioutil.ReadFile("metadata.json")
	if err != nil {
		return err
	}

	matMates := make([]mtx.MatrixMeta, 0, 0)

	// parsing json file to get matrix metadata
	if err := json.Unmarshal(jsonBytes, &matMates); err != nil {
		return err
	} else {
		for _, mat := range matMates {
			if path, ok := mtxPathMap[mat.Name]; ok {
				// the file exists, print it
				PrintMtxInfo(mat, path)
			}
		}
	}
	return nil
}
func PrintMtxInfo(mat mtx.MatrixMeta, path string) {
	fmt.Printf("%s,%s\n", mat.Name, path)

	// 		mp := map[string]int{
	// 			"1k":   0,
	// 			"10k":  0,
	// 			"100k": 0,
	// 			"1M":   0,
	// 			"10M":  0,
	// 			"100M": 0,
	// 			"1G":   0,
	// 			"10G":  0,
	// 		}
	//   for
	// 			if _, ok := mp[mat.Name]; ok {
	// 				fmt.Println(mat.Name, mat.ID, mat.DlLinks.MatrixMarket)
	// 			} else {
	// 				mp[mat.Name] = true
	// 			}
	// 			if mat.NNZ < 1000 {
	// 				mp["1k"]++
	// 			}else if mat.NNZ < 10*1000 {
	// 				mp["10k"]++
	// 			}else if mat.NNZ < 100*1000 {
	// 				mp["100k"]++
	// 			}else if mat.NNZ < 1000*1000 {
	// 				mp["1M"]++
	// 			}else if mat.NNZ < 1000*1000 {
	// 				mp["1M"]++
	// 			}else if mat.NNZ < 10*1000*1000 {
	// 				mp["10M"]++
	// 			}else if mat.NNZ < 100*1000*1000 {
	// 				mp["100M"]++
	// 			}else if mat.NNZ < 1000*1000*1000 {
	// 				mp["1G"]++
	// 			}else if mat.NNZ < 10*1000*1000*1000 {
	// 				mp["10G"]++
	// 			}
	//}
	//fmt.Println(mp)
}
