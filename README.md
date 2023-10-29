# Real-time Diagnosis of Intracerebral Hemorrhage by Generating Dual-energy CT from Single-energy CT
This repo contains the supported pytorch code and configuration files of our work
![Overview of the STDGAN](img/framework.png?raw=true)


## System requirements
This software was originally designed and run on a system running Ubuntu.

## Environment setup

Create a virtual environment 
- virtualenv -p /usr/bin/python3.8 STDGAN
- source STDGAN/bin/activate

Install other dependencies
- pip install -r requirements.txt

## Train Model

python train.py

## Acknowledgements

This repository makes liberal use of code from [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer), [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)


