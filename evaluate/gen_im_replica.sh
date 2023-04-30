for data in office4 room1 room2 #office0 office1 office2 office3 office4 room1 room2
do 
	for model in HG
	do 
		python evaluate/gen_out_im.py configs/replica/${model}/replica_${data}.yaml
	done
done	
