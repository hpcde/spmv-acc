# suitsparse-dl
> A tool to download matrices from SuiteSparse Matrix Collection(https://sparse.tamu.edu) site.

## Build
```bash
go build
```

## How to use
```bash
curl "https://sparse.tamu.edu/?per_page=All" -o index.html
mkdir -p dl/1k dl/10k dl/100k dl/1M dl/10M dl/100M dl/1G dl/10G # create directories   
./suitesparse-dl dl
```

## Extract Matrix Market from .tar.gz
We can use following srcipt to extract matrix market file from downloaded .tar.gz file.
```bash
#!/bin/bash

root_dir='dl'
target_dir='dl_mm'
mkdir -p $target_dir
find $root_dir -name "*tar.gz" | sort > tar_list.txt
for tar_path in `cat tar_list.txt`; do
    tar -zxvf $tar_path -C $target_dir --strip-components 1 > /dev/null
done
rm tar_list.txt
```
