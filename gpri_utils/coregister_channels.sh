#!/bin/bash
master=$1
master_par=$2
slave=$3
slave_par=$4
resampled=$5
resampled_par=$6
diff_par=$7
#If master and slave are the same
if [ "$master" = "$slave" ];then
	cp $master $resampled
	cp ${master_par} ${resampled_par}
else
	#Get basename
	slcpath=$(dirname $master)
	mastername=$(basename $master .slc)
	slavename=$(basename $slave .slc)
	master_mli=$mastername'.mli'
	slave_mli=$slavename'.mli'
	#Offset parameters
	rpos=469
	azpos=6265
	thresh=15
	#Mli parameter
	azlk=1
	rlk=1
	scl=1.2
	#Multilook
	diffparname=${diff_par}
	multi_look ${slave} ${slave_par} ${slave_mli} ${slave_mli}'.par' $rlk $azlk - $nlines - $scl
	multi_look ${master} ${master_par} ${master_mli} ${master_mli}'.par' $rlk $azlk - $nlines - $scl
	create_diff_par ${master_mli}'.par' ${slave_mli}'.par' $diffparname 1 0
	#Compute initial offset
	patch=128
	init_offsetm ${master_mli} ${slave_mli} $diffparname - - $rpos $azpos - - $thresh $patch 1
	rwin=32
	azwin=32
	nr=20
	naz=50
	offset_pwrm ${master_mli} ${slave_mli} $diffparname offs snr $rwin $azwin offsets - $nr $naz $thresh - 1 -
	#Use offset tracking
	offset_fitm offs snr $diffparname - - - 1
	interp_cpx $slave $diffparname $resampled
	cp ${master_par} ${resampled_par}
	rm ${master_mli}
	rm ${slave_mli}
fi
