#PBS-q
#cd $PBS_O_WORKDIR
export CUDA_VISIBLE_DEVICES='3'
python dcgan.py
