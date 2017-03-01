#!/bin/bash
rawpath=$1
chanpath=$2
corrpath=$3
slcpath=$4
decpath=$5
ridx=$6
azidx=$7
if [ ! -d  $chanpath ];
then
	mkdir $chanpath
fi
if [ ! -d  $corrpath ];
then
	mkdir $corrpath
fi
if [ ! -d  $slcpath ];
then
	mkdir $slcpath
fi
if [ ! -d  $decpath ];
then
	mkdir $decpath
fi
#Name of dataset
dsname=$(basename $1 .raw)
for chan in AAAl ABBl BAAl BBBl AAAu ABBu BAAu BBBu;
do
	#Extract the channels
	chan_name=${chanpath}"/"${dsname}"_"$chan
	echo ${chan_name}
	#~ extract_channel.py $rawpath $rawpath'_par' ${chan_name}".raw" ${chan_name}".raw_par" $chan
	#Correct the squint
	corr_name=${corrpath}"/"${dsname}"_"$chan
	#~ correct_squint.py ${chan_name}".raw" ${chan_name}".raw_par" ${corr_name}".raw" ${corr_name}".raw_par"
	#Compress
	slc_name=${slcpath}"/"${dsname}"_"$chan
	#~ range_compression.py ${corr_name}".raw" ${corr_name}".raw_par"  ${slc_name}".slc" ${slc_name}".slc.par"  -z 500 -d 1 -k 14 -r 50 -R 3000
	#Get distortion parameters
	r_ph=$(measure_phase_center.py ${slc_name}'.slc' ${slc_name}'.slc.par' $ridx $azidx 0.25 -w 100)
	#Correct
	slc_corr_name=${slcpath}"/"${dsname}"_"$chan"_corr"
	azimuth_correction.py ${slc_name}'.slc' ${slc_name}'.slc.par' ${slc_corr_name}'.slc' ${slc_corr_name}'.slc.par' --r_ph ${r_ph}
	if [ "$chan" != "AAAl" ];
	then
		master=${slcpath}"/"${dsname}"_AAAl_corr"
		slave=${slc_corr_name}
		coreg_name='temp.slc'
		coregister_channels.sh ${master}'.slc' ${master}'.slc.par' ${slave}'.slc' ${slave}'.slc.par' ${coreg_name}
		mv ${coreg_name} ${slave}'.slc'
	fi
	slc_dec_name=${decpath}"/"${dsname}"_"$chan"_corr"
	decimate.py ${slc_corr_name}'.slc' ${slc_corr_name}'.slc.par' ${slc_dec_name}'.slc' ${slc_dec_name}'.slc.par' 5
done










