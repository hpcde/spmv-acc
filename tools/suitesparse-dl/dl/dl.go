package dl

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/genshen/cmds"
	"github.com/pterm/pterm"

	m "suitesparse-dl/mtx"
)

var dlCommand = &cmds.Command{
	Name:        "dl",
	Summary:     "download matrices",
	Description: "download matrices from suitesparse",
	CustomFlags: false,
	HasOptions:  true,
}

const (
	DefaultDlGoroutines int    = 4 // goroutines number for parallel downloading.
	DefaultDlRoot       string = "dl/"
)

type dlOptions struct {
	NGoroutines int    // goroutines number for parallel downloading.
	DownloadDir string // director for storing result
}

var options dlOptions

func init() {
	dlCommand.Runner = &dl{}
	fs := flag.NewFlagSet("dl", flag.ContinueOnError)
	dlCommand.FlagSet = fs
	dlCommand.FlagSet.StringVar(&options.DownloadDir, "dir", DefaultDlRoot, `director for storing matrices.`)
	dlCommand.FlagSet.IntVar(&options.NGoroutines, "p", DefaultDlGoroutines, `goroutines for parallel downloading.`)
	dlCommand.FlagSet.Usage = dlCommand.Usage // use default usage provided by cmds.Command.
	cmds.AllCommands = append(cmds.AllCommands, dlCommand)
}

type dl struct{}

func (d *dl) PreRun() error {
	return nil // if error != nil, function Run will be not execute.
}

func (d *dl) Run() error {
	download()
	return nil
}

func download() {
	jsonBytes, err := ioutil.ReadFile("metadata.json")
	if err != nil {
		log.Fatal(err)
		return
	}

	matMates := make([]m.MatrixMeta, 0, 0)

	// parsing json file to get matrix metadata
	if err := json.Unmarshal(jsonBytes, &matMates); err != nil {
		log.Fatal(err)
		return
	} else {
		fmt.Printf("Downloading %d matrices using %d go goroutines, saved at `%s`", len(matMates), options.NGoroutines, options.DownloadDir)

		// set terminal processing bar
		pterm.DefaultSection.Println("Download Matrices")
		processbar, err := pterm.DefaultProgressbar.WithTotal(len(matMates)).WithTitle("Downloading matrices").Start()
		if err != nil {
			log.Fatal(err)
			return
		}

		var wg sync.WaitGroup
		var terminalLock = &sync.RWMutex{}
		dlTasks := make(chan bool, options.NGoroutines)
		stop := make(chan bool, options.NGoroutines) // listen for stop chan signal
		for _, mat := range matMates {
			select {
			case <-stop:
				goto OutFor // break for loop
			default:
				wg.Add(1)
				dlTasks <- true
				_mat := mat
				go func() {
					err := dl_matrix(_mat, processbar, terminalLock)
					<-dlTasks
					wg.Done()
					if err != nil {
						stop <- true
					}
				}()
			}
		}
	OutFor:
		wg.Wait()
		if _, err := processbar.Stop(); err != nil {
			return
		}
	}
}

func dl_matrix(mat m.MatrixMeta, processbar *pterm.ProgressbarPrinter, terminalLock *sync.RWMutex) error {
	terminalLock.Lock()
	processbar.Title = "Downloading " + mat.Name
	terminalLock.Unlock()

	if skipped, err := downloadFile(options.DownloadDir, mat.Name+".tar.gz", mat.NNZ, mat.DlLinks.MatrixMarket); err != nil {
		pterm.Error.Printf(err.Error())
		// stop the whole downloading task if it has error
		return err
	} else {
		terminalLock.Lock()
		if skipped {
			pterm.Warning.Printfln("Matrix %s skipped, maybe the matrix already exists.", mat.Name)
		} else {
			pterm.Success.Println("Downloaded matrix", mat.Name)
		}
		processbar.Increment()
		terminalLock.Unlock()
		return nil
	}
}

var letterRunes = []rune("1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

func RandStringRunes(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[rand.Intn(len(letterRunes))]
	}
	return string(b)
}

const (
	Nk    = 1000
	N10k  = 10 * 1000
	N100k = 100 * 1000
	NM    = 1000 * 1000
	N10M  = 10 * 1000 * 1000
	N100M = 100 * 1000 * 1000
	NG    = 1000 * 1000 * 1000
)

// generate file path by number of non-zeros
func filepathByNNz(basepath string, filename string, nnz int64) string {
	var category string
	if nnz < Nk {
		category = "1k"
	} else if nnz < N10k {
		category = "10k"
	} else if nnz < N100k {
		category = "100k"
	} else if nnz < NM {
		category = "1M" // =106
	} else if nnz < N10M {
		category = "10M"
	} else if nnz < N100M {
		category = "100M"
	} else if nnz < NG {
		category = "1G" // =1e9
	} else {
		category = "10G"
	}
	return filepath.Join(basepath, category, filename)
}

func downloadFile(basepath string, filename string, nnz int64, url string) (skipped bool, err error) {
	targetFilePath := filepathByNNz(basepath, filename, nnz)

	if _, err := os.Stat(targetFilePath); err == nil {
		return true, nil // "file exists"
	} else if errors.Is(err, os.ErrNotExist) {
		// fall through
	} else {
		return true, err
	}

	// Create the temp file
	tempFileName := filepath.Join(basepath, filename+".tmp."+RandStringRunes(6))
	out, err := os.Create(tempFileName)
	if err != nil {
		return false, err
	}
	defer out.Close()

	// Get the data
	client := http.Client{} // {Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	// Check server response
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("bad status: %s", resp.Status)
	}

	// Writer the body to file
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return false, err
	}

	// rename temp file to tar.gz file
	if err := os.Rename(tempFileName, targetFilePath); err != nil {
		return false, err
	}

	return false, nil
}
