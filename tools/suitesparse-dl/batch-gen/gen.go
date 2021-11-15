package batch_gen

import (
	"bytes"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"text/template"

	"github.com/genshen/cmds"
)

var genCommand = &cmds.Command{
	Name:        "gen",
	Summary:     "generate batch file",
	Description: "generate batch file from a template file for submitting jobs.",
	CustomFlags: false,
	HasOptions:  true,
}

func init() {
	g := gen{}
	genCommand.Runner = &g
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	genCommand.FlagSet = fs
	genCommand.FlagSet.StringVar(&(g.templatePath), "tpl", "template.sh", `the template file path for rending.`)
	genCommand.FlagSet.StringVar(&(g.outputPath), "output", "spmv_batch.sh", `output path of batch file.`)
	genCommand.FlagSet.StringVar(&(g.dataPath), "data", "./", `path of storing the matrices (e.g. "./dl_mm/10k/").`)
	genCommand.FlagSet.Usage = genCommand.Usage // use default usage provided by cmds.Command.
	cmds.AllCommands = append(cmds.AllCommands, genCommand)
}

type gen struct {
	templatePath    string
	templateContent string
	dataPath        string // data path storing the matrices.
	outputPath      string
}

type MtxFileMeta struct {
	Name    string
	Path    string
	AbsPath string
}

type Matrices struct {
	Metas []MtxFileMeta
	Size  int
}

func (g *gen) PreRun() error {
	// load template file.
	content, err := ioutil.ReadFile(g.templatePath)
	if err != nil {
		return err
	}
	g.templateContent = string(content)
	return nil
}

func (g *gen) Run() error {
	files, err := ioutil.ReadDir(g.dataPath)
	if err != nil {
		return err
	}

	allMatrices := make([]MtxFileMeta, 0)

	for _, dir := range files {
		if dir.IsDir() {
			mtxName := dir.Name() + ".mtx"
			mtxFilePath := filepath.Join(g.dataPath, dir.Name(), mtxName)

			// matrix fil info
			if fileInfo, err := os.Stat(mtxFilePath); os.IsNotExist(err) {
				log.Printf("file %s does not exist, skip it.", mtxFilePath)
			} else if err != nil {
				return err
			} else if fileInfo.IsDir() {
				log.Printf("%s is a directory, skip it.", mtxFilePath)
			} else {
				// get absolute path
				mtxAbsPath, err := filepath.Abs(mtxFilePath)
				if err != nil {
					return err
				}
				allMatrices = append(allMatrices, MtxFileMeta{
					Name:    mtxName,
					Path:    mtxFilePath,
					AbsPath: mtxAbsPath,
				})
			}
		}
	}

	// render template
	matrices := Matrices{
		Metas: allMatrices,
		Size:  len(allMatrices),
	}

	if tpl, err := template.New("batch").Parse(g.templateContent); err != nil {
		return err
	} else {
		buffer := bytes.Buffer{}
		if err = tpl.Execute(&buffer, matrices); err != nil {
			return err
		} else {
			if err = ioutil.WriteFile(g.outputPath, buffer.Bytes(), 0744); err != nil {
				return err
			}
		}
	}
	log.Printf("batch file writen to `%s`", g.outputPath)
	return nil
}
