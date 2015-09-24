#!/bin/bash
master=$1
master_par=$2
slave=$3
slave_par=$4
resampled=$5
#Get basename
slcpath=$(dirname $master)
mastername=$(basename $master .slc)
slavename=$(basename $slave .slc)
master_mli=$mastername'.mli'
slave_mli=$slavename'.mli'
#Offset parameters
rpos=812
azpos=7970
thresh=0.25
offrlks=1
offazlks=1
#Mli parameter
azlk=1
rlk=1
scl=1.2
#Interferogram parameters
intazlks=1
intrlks=1
#Multilook
parname=${mastername}_${slavename}'.off_par'
multi_look ${slave} ${slave_par} ${slave_mli} ${slave_mli}'.par' $rlk $azlk - - - $scl
multi_look ${master} ${master_par} ${master_mli} ${master_mli}'.par' $rlk $azlk - - - $scl
create_diff_par  ${master_mli}'.par' ${slave_mli}'.par' $rlks $azlks  $parname 1 0
#Compute initial offset
patch=256
init_offsetm ${master_mli} ${slave_mli} $parname $offrlks $offazlks $rpos $azpos - - $thresh $patch 1
offset_pwrm ${master_mli} ${slave_mli} $parname offs snr - - - - - - $thresh -
offset_fitm offs snr $parname
interp_cpx $slave $parname $resampled
