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
    *)
      newcmd="$newcmd $arg";;
  esac
done

# Finally execute the new command
exec $newcmd
