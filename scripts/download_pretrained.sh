# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://convaisharables.blob.core.windows.net/villa'

for MODEL in uniter-base uniter-large villa-base villa-large; do
    wget $BLOB/pretrained/$MODEL.pt -P $DOWNLOAD/pretrained/
done

for MODEL in uniter-base-vcr-2nd-stage uniter-large-vcr-2nd-stage; do
    wget $BLOB/pretrained/$MODEL.pt -P $DOWNLOAD/pretrained/
done

for MODEL in villa-base-vcr-2nd-stage villa-large-vcr-2nd-stage; do
    wget $BLOB/pretrained/$MODEL.pt -P $DOWNLOAD/pretrained/
done
