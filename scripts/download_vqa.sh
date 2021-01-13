# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://convaisharables.blob.core.windows.net/uniter'

# image dbs
for SPLIT in 'train2014' 'val2014' 'test2015'; do
    if [ ! -d $DOWNLOAD/img_db/coco_$SPLIT ] ; then
        echo "Downloading default coco_$SPLIT ..."
        wget $BLOB/img_db/coco_$SPLIT.tar -P $DOWNLOAD/img_db/
        tar -xvf $DOWNLOAD/img_db/coco_$SPLIT.tar -C $DOWNLOAD/img_db
    else
        echo "Found default coco_$SPLIT, skipping ..."
    fi
    if [ ! -f $DOWNLOAD/img_db/coco_$SPLIT/nbb_th0.075_max100_min10.json ] ; then
        echo "Downloading coco_$SPLIT with conf 0.075..."
        wget $BLOB/img_db/coco_${SPLIT}_0.075.tar -P $DOWNLOAD/img_db/
        tar -xvf $DOWNLOAD/img_db/coco_${SPLIT}_0.075.tar -C $DOWNLOAD/img_db
    else
        echo "Found coco_$SPLIT with conf 0.075, skipping ..."
    fi
done
if [ ! -d $DOWNLOAD/img_db/vg ] ; then 
    echo "Downloading default vg ..."
    wget $BLOB/img_db/vg.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/vg.tar -C $DOWNLOAD/img_db
else
    echo "Found default vg, skipping ..."
fi
if [ ! -f $DOWNLOAD/img_db/vg/nbb_th0.075_max100_min10.json ] ; then
    echo "Downloading vg with conf 0.075..."
    wget $BLOB/img_db/vg_0.075.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/vg_0.075.tar -C $DOWNLOAD/img_db
else
    echo "Found vg with conf 0.075, skipping ..."
fi

# text dbs
for SPLIT in 'train' 'trainval' 'devval' 'test' 'vg'; do
    wget $BLOB/txt_db/vqa_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/vqa_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

BLOB='https://convaisharables.blob.core.windows.net/villa'

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    wget $BLOB/pretrained/uniter-base.pt -P $DOWNLOAD/pretrained/
fi

if [ ! -f $DOWNLOAD/pretrained/villa-base.pt ] ; then
    wget $BLOB/pretrained/villa-base.pt -P $DOWNLOAD/pretrained/
fi

