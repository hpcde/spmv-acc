package dl

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
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
	DlGoroutines int    = 4 // goroutines number for parallel downloading.
	DlRoot       string = "dl/"
)


func init() {
	dlCommand.Runner = &dl{}
	fs := flag.NewFlagSet("dl", flag.ContinueOnError)
	dlCommand.FlagSet = fs
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
	htmlBytes, err := ioutil.ReadFile("index.html")
	if err != nil {
		log.Fatal(err)
	}

	// parsing html file to get matrix metadata
	if matMates, err := m.SourceParse(bytes.NewReader(htmlBytes)); err != nil {
		log.Fatal(err)
		return
	} else {
		fmt.Printf("Downloading %d matrices using %d go goroutines", len(matMates), DlGoroutines)

		// set terminal processing bar
		pterm.DefaultSection.Println("Download Matrices")
		processbar, err := pterm.DefaultProgressbar.WithTotal(len(matMates)).WithTitle("Downloading matrices").Start()
		if err != nil {
			log.Fatal(err)
			return
		}

		var wg sync.WaitGroup
		var terminalLock = &sync.RWMutex{}
		dlTasks := make(chan bool, DlGoroutines)
		stop := make(chan bool, DlGoroutines) // listen for stop chan signal
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

	if skipped, err := downloadFile(DlRoot+mat.Name+".tar.gz", mat.DlLinks.MatrixMarket); err != nil {
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

func downloadFile(filepath string, url string) (skipped bool, err error) {
	if _, err := os.Stat(filepath); err == nil {
		return true, nil // "file exists"
	} else if errors.Is(err, os.ErrNotExist) {
		// fall through
	} else {
		return true, err
	}

	// Create the temp file
	tempFileName := filepath + ".tmp." + RandStringRunes(6)
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
	if err := os.Rename(tempFileName, filepath); err != nil {
		return false, err
	}

	return false, nil
}
