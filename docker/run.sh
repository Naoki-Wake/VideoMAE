#!/bin/bash
docker run --rm \
       --network=host \
       --privileged \
       --gpus all \
       --volume="/home/nawake/:/home/nawake" \
       --device /dev/snd:/dev/snd \
       -it lfovision.azurecr.io/base_videomae
#       --volume="/dev:/dev" \
#       --volume="/mnt/ssd_2T/video/sthv2:/mmaction2/data/sthv2" \
#       --volume="/mnt/ssd_2T/video/household:/mmaction2/data/household" \
#       --volume="/mnt/ssd_2T/video/demo:/mmaction2/data/demo" \
#       --volume="/home/nawake/sthv2:/sthv2" \