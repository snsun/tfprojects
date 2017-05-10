#!/bin/bash
if [ $# != 3 ]; then
  echo "USAGE: $0 train feats.scp, targets.scp"
  exit 1;
fi

name=$1
feats_scp=$2
targets_scp=$3
cut -d ' ' -f 2 $targets_scp  > tmp
paste -d ' ' $feats_scp tmp > config/$name.lst
