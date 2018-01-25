#PBS-q
#cd $PBS_O_WORKDIR
export CUDA_VISIBLE_DEVICES=''
python cgan.py
