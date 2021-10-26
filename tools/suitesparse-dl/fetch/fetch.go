package fetch

import (
	"bytes"
	"flag"
	"fmt"
	"github.com/genshen/cmds"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"suitesparse-dl/mtx"
)

var fetchCommand = &cmds.Command{
	Name:        "fetch",
	Summary:     "fetch data from suitesparse website",
	Description: "fetch data from suitesparse website and save to csv file.",
	CustomFlags: false,
	HasOptions:  true,
}

const DefaultUrl = "https://sparse.tamu.edu/?per_page=All"

func init() {
	f := fetch{}
	fetchCommand.Runner = &f
	fs := flag.NewFlagSet("fetch", flag.ContinueOnError)
	fetchCommand.FlagSet = fs
	fetchCommand.FlagSet.StringVar(&(f.url), "url", DefaultUrl, `url of suitesparse site.`)
	fetchCommand.FlagSet.StringVar(&(f.output), "o", "suitesparse.csv", `output of csv file.`)
	fetchCommand.FlagSet.Usage = fetchCommand.Usage // use default usage provided by cmds.Command.
	cmds.AllCommands = append(cmds.AllCommands, fetchCommand)
}

type fetch struct {
	url    string
	output string
}

func (f *fetch) PreRun() error {
	return nil // if error != nil, function Run will be not execute.
}

func (f *fetch) Run() error {
	resp, err := http.Get(f.url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	log.Println("fetch metadata...")
	htmlBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	// parsing html file to get matrix metadata
	if matMates, err := mtx.SourceParse(bytes.NewReader(htmlBytes)); err != nil {
		return err
	} else {
		out, err := os.Create(f.output)
		if err != nil {
			return err
		}
		defer out.Close()

		if _, err := out.WriteString("ID, Name, Group, Kind, Date, Rows, Cols, NNZ, MatrixMarket, Matlab, RutherfordBoeing\n"); err != nil {
			return err
		}

		// dump to csv
		for _, mat := range matMates {
			if _, err := out.WriteString(fmt.Sprintf("%s, %s, %s, %s, %s, %d, %d, %d, %s, %s, %s\n",
				mat.ID, mat.Name, mat.Group, mat.Kind, mat.Date, mat.Rows, mat.Cols, mat.NNZ,
				mat.DlLinks.MatrixMarket, mat.DlLinks.Matlab, mat.DlLinks.RutherfordBoeing)); err != nil {
				return err
			}

		// dump to json
		if metaJsonBytes, err := json.Marshal(matMates); err != nil {
			return err
		} else {
			jsonFile, err := os.Create("metadata.json")
			if err != nil {
				return err
			}
			defer out.Close()

			if _, err := jsonFile.Write(metaJsonBytes); err != nil {
				return err
			}
		}
		return nil
	}
}
