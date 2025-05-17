#!/bin/bash

LINKER="hipcc"
newcmd="$LINKER"
flag_xcpp_appear=0

for arg in $@
do
  case $arg in
    "-hc") # 移除参数 `-hc`
      ;;

    # OpenMP
    "-fopenmp") # 添加参数 `-Xcompiler`
     newcmd="$newcmd -Xcompiler -fopenmp";;

    "-fpch-preprocess")
     newcmd="$newcmd -Xcompiler -fpch-preprocess";;

    "-xc++")
     flag_xcpp_appear=1
     newcmd="$newcmd -Xcompiler -xc++";;

    "-fdiagnostics-color=always")
    newcmd="$newcmd -Xcompiler -fdiagnostics-color=always";;

    *)
      newcmd="$newcmd $arg";;
  esac
done


#if [ "$flag_xcpp_appear" -eq 1 ]; then
  echo "flag '-xc++' appears in this commane." >&2
  echo $newcmd >&2
#fi
# Finally execute the new command
exec $newcmd
