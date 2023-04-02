#!/bin/sh

# this wrapper script is used for nvlink/hipcc to remove `-std=gnu++14` argument.

LINKER=/opt/compilers/rocm/4.2.0/bin/hipcc
newcmd="$LINKER"
REMOVE="-std=gnu++14"

for arg in $@
do
  case $arg in
    $REMOVE) # remove argument
      ;;

    # OpenMP
    "-fopenmp")
     newcmd="$newcmd -Xcompiler -fopenmp";;

    "-fopenmp=libomp")
      newcmd="$newcmd -Xcompiler -fopenmp";;
    *)
      newcmd="$newcmd $arg";;
  esac
done

# Finally execute the new command
exec $newcmd --gpu-architecture=compute_70 --gpu-code=sm_70
