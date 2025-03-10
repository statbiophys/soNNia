## Command line console scripts

There are three command line console scripts (the scripts can still be called as executables if SoNNia is not installed):
1. ```sonnia evaluate```
  * evaluates Ppost, Pgen or selection factors of sequences according to a generative V(D)J model and selection model.
2. ```sonnia generate```
  * generates CDR3 sequences, before (like olga) or after selection
3. ```sonnia infer```
  * infers a selection model with respect to a generative V(D)J model

For any of them you can execute with the -h or --help flags to get the options.

We offer a quick demonstration of the console scripts. This will show how to generate and evaluate sequences and infer a selection model using the default generation model for human TCR beta chains that ships with the SONIA software. In order to run the commands below you need to download the examples folder. 

1. ```$ sonnia infer --model humanTRB -i examples/data_seqs.csv.gz```
  * This reads in the full file example_seqs.csv.gz, infers a selection model and saves to the folder sel_model
2. ```$ sonnia generate --model examples/sonnia_model --post -n 100```
  * Generate 100 human TRB CDR3 sequences from the post-selection repertoire and print to stdout along with the V and J genes used to generate them.
3. ```$ sonnia evaluate --model examples/sonnia_model -i examples/data_seqs.csv.gz --ppost ```
  * This computes Ppost,Pgen and Q of the first 100 seqs in the data_seqs file.  
  
### Specifying a default V(D)J model (or a custom model folder)
All of the console scripts require specifying a V(D)J model. SoNNia ships with 6 default models that can be indicated by flags, or a custom model folder can be indicated.

| Models                                         | Description                                      |
|-------------------------------------------------|--------------------------------------------------|
| **humanTRA**                                  | Default human T cell alpha chain model (VJ)      |
| **humanTRB**                                  | Default human T cell beta chain model (VDJ)      |
| **humanIGH**                                  | Default human B cell heavy chain model (VDJ)     |
| **humanIGK**                                  | Default human B cell light kappa chain model (VJ)|
| **humanIGL**                                  | Default human B cell light lambda chain model (VJ)|
| **mouseTRB**                                  | Default mouse T cell beta chain model (VDJ)      |
| **mouseIGH**                                  | Default mouse B cell heavy chain model (VDJ)      |

Note, if specifying a folder for a custom VJ recombination model
(e.g. an alpha or light chain model) or a custom VDJ recombination model
(e.g. a beta or heavy chain model), the folder must contain the following files
with the exact naming convention:

* model_params.txt 
* model_marginals.txt 
* V_gene_CDR3_anchors.csv (V anchor residue position and functionality file)
* J_gene_CDR3_anchors.csv (J anchor residue position and functionality file)
* features.tsv (if you want to load the selection model as well: not required for in the sonia-infer command)
* log.txt (if you want to load the selection model as well: not required in the sonia-infer command) 
* model.h5 (if you want to load a non linear selection model as well: not required in the sonia-infer command) 

The console scripts can only read files of the assumed anchor.csv/[IGoR](https://github.com/qmarcou/IGoR) syntaxes. See the default models in the sonia directory for examples.

### Options common to all commands

| Options                                         | Description                                      |
|-------------------------------------------------|--------------------------------------------------|
|  -h, --help                                     | show command options                             |
|  --model=MODEL_TYPE                             | specify model type.                              |
|  -i PATH/TO/FILE, --infile=PATH/TO/FILE         | read in CDR3 sequences (and optionally V/J masks) from PATH/TO/FILE|
|  -o PATH/TO/FILE, --outfile=PATH/TO/FILE        | output to PATH/TO/FILE                           |
|  -m N, --max_number_of_seqs=N                   | read at most N sequences.                        |
for command specific options use the --help flag.