# cd to current directory
module load anaconda/2023a

conda config --add channels conda-forge
conda create --name expprog --file requirements.txt 

conda activate expprog

mkdir /state/partition1/user/$USER
export TMPDIR=/state/partition1/user/$USER
python3.12 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --user