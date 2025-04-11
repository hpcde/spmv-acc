package batch_gen

import (
	"bytes"
	"flag"
	"fmt"
	"io/fs"
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
	genCommand.FlagSet.StringVar(&(g.mtxType), "type", "mtx", `type of the input matrix. Value can only be mtx or bin2 (file extension of .mtx or .bin2)`)
	genCommand.FlagSet.Usage = genCommand.Usage // use default usage provided by cmds.Command.
	cmds.AllCommands = append(cmds.AllCommands, genCommand)
}

type gen struct {
	templatePath    string
	templateContent string
	dataPath        string // data path storing the matrices.
	mtxType         string // matrix type: mtx or bin2
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

	for _, file := range files {
		if mtxMeta, err := g.matchMatrix(file, g.mtxType); err != nil {
			return err
		} else {
			if mtxMeta != nil {
				allMatrices = append(allMatrices, *mtxMeta)
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

func (g *gen) matchMatrix(file fs.FileInfo, mtxTp string) (*MtxFileMeta, error) {
	if g.mtxType == "mtx" {
		if !file.IsDir() {
			return nil, nil // skip normal file
		}
		dir := file.Name()
		mtxName := dir + ".mtx" // dirName.mtx
		mtxFilePath := filepath.Join(g.dataPath, dir, mtxName)
		// matrix file info
		if fileInfo, err := os.Stat(mtxFilePath); os.IsNotExist(err) {
			log.Printf("file %s does not exist, skip it.", mtxFilePath)
		} else if err != nil {
			return nil, err
		} else if fileInfo.IsDir() {
			log.Printf("%s is a directory, skip it.", mtxFilePath)
		} else {
			// get absolute path
			mtxAbsPath, err := filepath.Abs(mtxFilePath)
			if err != nil {
				return nil, err
			}
			return &MtxFileMeta{Name: mtxName, Path: mtxFilePath, AbsPath: mtxAbsPath}, nil
		}

	}
	if g.mtxType == "bin2" {
		if file.IsDir() { // skip dir
			return nil, nil
		}
		mtxName := file.Name() + ".bin2"
		mtxFilePath := filepath.Join(g.dataPath, mtxName)
		mtxAbsPath, err := filepath.Abs(mtxFilePath)
		if err != nil {
			return nil, err
		}
		return &MtxFileMeta{Name: mtxName, Path: mtxFilePath, AbsPath: mtxAbsPath}, nil
	}
	return nil, fmt.Errorf("unsupported input matrix type: %s", g.mtxType)
}
