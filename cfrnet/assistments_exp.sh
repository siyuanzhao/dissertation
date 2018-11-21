ps=$1
sea=$2
cnt=50
rname='lstm-autoencoder/results/'$1'_result.pkl'
directory='cfrnet'
mkdir results
if [ -d "$directory/results/$ps" ]
then
    rm -rf $directory/results/$ps
fi
mkdir $directory/results/$ps

python $directory/cfr_param_search.py $directory/configs/assistments_exp.txt $cnt $ps $sea $rname

python $directory/evaluate.py $directory/configs/assistments_exp.txt 1 $sea $rname $ps
