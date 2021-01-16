#!/bin/bash
#pele=../../../bin/pelegant.sif
pele=/home/mti/gitlab-hzb/containers/bin/pelegant.sif

while getopts "p" OPTION; do
        case $OPTION in

                p)
                        PARALLEL="true"
                        ;;
        esac

	shift
done

if [ "$PARALLEL" = "true" ]; then
        #cmd="bash run_pelegant_in_container.sh"
        cmd="bash temp_run_pelegant.sh"
else
	cmd=elegant
fi

echo $cmd
#cmd=elegant
$pele $cmd $1


filename=$(basename -- "$1")
extension="${filename##*.}"
filename="${filename%.*}"
echo $filename

#  print resulting tunes and chromas
RED='\033[0;31m'
NC='\033[0m' # No Color
echo -e "${RED}Tunes and chroma for simple tracking${NC}"
$pele  sddsprintout -parameters='(dnu?/dp,nu?)' fodofma.twi

# plot fmap
$pele sddsplot $filename.fma -column=x,y -split=column=diffusionRate -graph=sym,vary=subtype,scale=2,fill,type=2 -order=spectral 

