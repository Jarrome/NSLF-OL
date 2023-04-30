cd ..
for model in HG # AFFM one_HG
do
	for data_type in replica
	do 
		for data in room0 room1 room2 office0 office1 office2 office3 office4
		do 
python evaluate/gen_out_im.py configs/${data_type}/HG/replica_${data}.yaml
		done
	done
done
