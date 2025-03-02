#!/bin/bash

CURR_CONDA=$CONDA_DEFAULT_ENV
CONDA_BASE=$(conda info --base)
docker pull xingyu/softgym
docker run \
    -v ${PWD}/PyFleX:/workspace/PyFleX \
    -v ${CONDA_PREFIX}:/workspace/anaconda \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash \
    -c "export PATH=/workspace/anaconda/bin:$PATH; cd /workspace/PyFleX; export PYFLEXROOT=/workspace/PyFleX; export PYTHONPATH=/workspace/PyFleX/bindings/build:$PYTHONPATH; export LD_LIBRARY_PATH=$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH;cd bindings; rm -rf build; mkdir build; cd build; /usr/bin/cmake ..; make -j"

echo '' >> /home/youngjoon/.bashrc
echo '# PyFleX' >> /home/youngjoon/.bashrc
echo "export PYFLEXROOT=${PWD}/PyFleX" >> /home/youngjoon/.bashrc
echo 'export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH' >> /home/youngjoon/.bashrc
echo 'export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH' >> /home/youngjoon/.bashrc
echo '' >> /home/youngjoon/.bashrc

source /home/youngjoon/.bashrc
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $CURR_CONDA
