package main

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"github.com/andybalholm/cascadia"
	"golang.org/x/net/html"
	"io"
	"log"
	"strconv"
	"strings"
)

type XmlMatrixAttr struct {
	XmlAttr string `xml:",innerxml"`
}

type XmlMatrix struct {
	XmlMatrixAttrs []XmlMatrixAttr `xml:"td"`
}

type XmlMatrices struct {
	XmlMatrix []XmlMatrix `xml:"tr"`
}

type LinkExtractor struct {
	Href string `xml:"href,attr"`
	Text string `xml:",innerxml"`
}

const (
	AttrsNumInMatrix int = 9
)

func sourceParse(reader io.Reader) ([]MatrixMeta, error) {
	doc, err := html.Parse(reader)
	if err != nil {
		return nil, err
	}

	// filter tbody in table
	tbody := cascadia.MustCompile("tbody").MatchFirst(doc)
	tbodyBuf := bytes.Buffer{}
	if err := html.Render(&tbodyBuf, tbody); err != nil {
		return nil, err
	}

	// parsing tbody element
	var mtx XmlMatrices
	if err := xml.Unmarshal(tbodyBuf.Bytes(), &mtx); err != nil {
		return nil, err
	}

	matMates := make([]MatrixMeta, 0)
	for _, tr := range mtx.XmlMatrix {
		if len(tr.XmlMatrixAttrs) != AttrsNumInMatrix {
			return nil, errors.New("matrix attributions number does not match")
		}

		// parsing download links
		dlLink, err := extractDlLink(tr.XmlMatrixAttrs[8].XmlAttr)
		if err != nil {
			return nil, err
		}

		// store matrix attribution
		matMeta := MatrixMeta{
			ID:      tr.XmlMatrixAttrs[0].XmlAttr,
			Name:    getLinkText(tr.XmlMatrixAttrs[1].XmlAttr),
			Group:   getLinkText(tr.XmlMatrixAttrs[2].XmlAttr),
			Rows:    parseInt(tr.XmlMatrixAttrs[3].XmlAttr),
			Cols:    parseInt(tr.XmlMatrixAttrs[4].XmlAttr),
			NNZ:     parseInt(tr.XmlMatrixAttrs[5].XmlAttr),
			Kind:    tr.XmlMatrixAttrs[6].XmlAttr,
			Date:    tr.XmlMatrixAttrs[7].XmlAttr,
			DlLinks: dlLink,
		}
		matMates = append(matMates, matMeta)
	}
	return matMates, nil
}

func extractDlLink(linkText string) (DlLinks, error) {
	type Links struct {
		Links []LinkExtractor `xml:"a"`
	}

	var links Links
	if err := xml.Unmarshal([]byte("<link>"+linkText+"</link>"), &links); err != nil {
		return DlLinks{}, err
	} else {
		if len(links.Links) != 3 {
			return DlLinks{}, fmt.Errorf("download links number must be 3, but got %d when pasing `%s`", len(links.Links), linkText)
		}
		return DlLinks{
			Matlab:           links.Links[0].Href,
			RutherfordBoeing: links.Links[1].Href,
			MatrixMarket:     links.Links[2].Href,
		}, err
	}
}

func getLinkText(linkText string) string {
	var link LinkExtractor
	if err := xml.Unmarshal([]byte(linkText), &link); err != nil {
		log.Fatal(err)
		return ""
	} else {
		return link.Text
	}
}

func parseInt(str string) int {
	num, err := strconv.Atoi(strings.Replace(str, ",", "", -1))
	if err != nil {
		log.Fatalln(err)
		return 0
	}
	return num
}
