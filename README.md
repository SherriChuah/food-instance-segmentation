# food-instance-segmentation
Testing maskrcnn_resnet50_fpn_v2 model for instance segmentation

## Setup and run
1. Clone repo

`
$ git clone {link}
$ cd maskrcnn-instance-segmentation
`

2. Create and activate conda environment (install conda first)

 `
 $ conda --name {env-name} python=3.10
 $ conda activate {env-name}
 `

3. Run environments.yml file

`
$ conda env update -n {env-name} --file environments.yml
`

4. Run program

`
$ python3 scripts/inference.py
`
