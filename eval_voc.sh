
resume=../1sw/run/pascal/deeplab_v14_3/model_best.pth.tar
save_dir=../1sw/output/Deeplab/deeplab_v14_3/
crf=True
python eval.py --resume $resume --save_dir $save_dir --backbone resnet --batch-size 1 --gpu-ids 0 --dataset pascal --mode eval --crf $crf
