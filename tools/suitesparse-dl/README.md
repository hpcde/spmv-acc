# suitsparse-dl
> A tool to download matrices from SuiteSparse Matrix Collection(https://sparse.tamu.edu) site.

## Build
```bash
go build
```

## How to use
```bash
./suitesparse fetch # fetch metadata
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

## Special note for matrices with the same name
The matrix name is not the unique key in SuiteSparse Matrix Collection.
But suitsparse-dl use matrix name as filename (key).
Thus, matrices with the same name will only be downloaded once when the name first apprears.
In other words, if 2 or more matrices with the same name, we only download the first matrix, other matrices will be ignored.

A workaround for this problem is: user can **manually** download the matrices with the same name and give them different filenames.

The matrices with the same name are list as following:

| Name         | IDs |
| ------------ | --- |
| nasa1824     | 363, 757  |
| nasa2910     | 364, 759  |
| nasa4704     | 365, 760  |
| barth        | 754, 865  |
| barth4       | 755, 866  |
| barth5       | 756, 867  |
| pwt          | 762, 880, 1273 |
| shuttle_eddy | 763, 881  |
| skirt        | 764, 882  |
| copter2      | 1230, 1256|
| ex3sta1      | 1379, 1709 (*) |
| pf2177       | 1394, 1753 (*) |
| fxm3_6       | 1380, 1805 (*) |
| fxm4_6       | 1381, 1807 (*) |
| football     | 1474, 2397 (*) |
