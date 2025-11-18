## Command line console scripts

SoNNia provides three main command line console scripts accessible via the `sonnia` command (installed via pip or can be called as executables):

1. **`sonnia infer`**
   * Infers a selection model with respect to a generative V(D)J model. Can infer both linear (Sonia) and non-linear (SoNNia) models, for both single-chain and paired-chain sequences.

2. **`sonnia evaluate`**
   * Evaluates Ppost, Pgen, and selection factors (Q) of sequences according to a generative V(D)J model and selection model.

3. **`sonnia generate`**
   * Generates CDR3 (junction) sequences, either before selection (pre-selection, like OLGA) or after selection (post-selection).

For any command, you can execute with the `-h` or `--help` flags to get detailed options.

### Quick Start Examples

We offer a quick demonstration of the console scripts. This will show how to generate and evaluate sequences and infer a selection model using the default generation model for human TCR beta chains. In order to run the commands below you need to download the examples folder. 

1. **Infer a selection model:**
   ```bash
   $ sonnia infer --model humanTRB -i examples/data_seqs.csv.gz
   ```
   This reads in the file, infers a non-linear selection model (SoNNia) and saves to the folder `sonnia_model` (default output directory). The command also generates several plot files: `model_learning.png`, `marginals.png`, `log_Q.png`, and `Q_ratio.png`.

2. **Infer a linear model:**
   ```bash
   $ sonnia infer --model humanTRB -i examples/data_seqs.csv.gz --linear
   ```
   This infers a linear selection model (Sonia) instead.

3. **Generate sequences:**
   ```bash
   $ sonnia generate --model sonnia_model --post -n 100
   ```
   Generate 100 human TRB CDR3 (junction) sequences from the post-selection repertoire and print to stdout along with the V and J genes used to generate them.

4. **Evaluate sequences:**
   ```bash
   $ sonnia evaluate --model sonnia_model -i examples/data_seqs.csv.gz -o evaluated_seqs.tsv
   ```
   This computes Ppost, Pgen, and Q for all sequences in the input file and saves to `evaluated_seqs.tsv`.  
   
### Specifying a default V(D)J model (or a custom model folder)

All of the console scripts require specifying a V(D)J model. SoNNia ships with several default models that can be indicated by name, or a custom model folder can be specified.

#### Single-chain models:

| Model Name                                     | Description                                      |
|------------------------------------------------|--------------------------------------------------|
| **humanTRA**                                   | Default human T cell alpha chain model (VJ)      |
| **humanTRB**                                   | Default human T cell beta chain model (VDJ)     |
| **humanIGH**                                   | Default human B cell heavy chain model (VDJ)    |
| **humanIGK**                                   | Default human B cell light kappa chain model (VJ)|
| **humanIGL**                                   | Default human B cell light lambda chain model (VJ)|
| **mouseTRA**                                   | Default mouse T cell alpha chain model (VJ)      |
| **mouseTRB**                                   | Default mouse T cell beta chain model (VDJ)     |
| **mouseIGH**                                   | Default mouse B cell heavy chain model (VDJ)    |

#### Paired-chain models:

| Model Name                                     | Description                                      |
|------------------------------------------------|--------------------------------------------------|
| **humanTCR**                                   | Human T cell receptor (alpha-beta paired)        |
| **humanIGHK**                                  | Human B cell receptor (heavy-kappa paired)      |
| **humanIGHL**                                  | Human B cell receptor (heavy-lambda paired)     |

#### Custom model folder

If specifying a folder for a custom VJ recombination model (e.g., an alpha or light chain model) or a custom VDJ recombination model (e.g., a beta or heavy chain model), the folder must contain the following files with the exact naming convention:

* `model_params.txt` 
* `model_marginals.txt` 
* `V_gene_CDR3_anchors.csv` (V anchor residue position and functionality file)
* `J_gene_CDR3_anchors.csv` (J anchor residue position and functionality file)
* `features.tsv` (required to load the selection model; not required for `sonnia infer` command)
* `log.txt` (optional; contains training log)
* `model.h5` (required to load a non-linear selection model; not required for `sonnia infer` command)

For paired-chain models, the folder should contain `heavy_chain/` and `light_chain/` subdirectories, each with the above files.

The console scripts can read files in various formats (CSV, TSV, etc.) and automatically detect the delimiter. See the default models in the `sonnia/default_models/` directory for examples.

### Command-specific options

#### `sonnia infer` options

| Option                                          | Description                                      |
|-------------------------------------------------|--------------------------------------------------|
| `-i, --infile`                                  | Path to input file (required)                   |
| `--model`                                       | Model name or path to custom model folder (optional) |
| `-o, --outdir`                                  | Output directory (default: `sonnia_model`)      |
| `--linear`                                      | Infer linear model instead of non-linear        |
| `--paired`                                      | Use paired-chain model. Assumes heavy and light chains are in separate columns named `junction_aa_heavy`, `v_gene_heavy`, `j_gene_heavy`, `junction_aa_light`, `v_gene_light`, `j_gene_light`. |
| `--max-seqs`                                    | Maximum number of sequences to use (default: 1e8)|
| `--max-gen-seqs`                                | Maximum number of sequences to generate (default: 1e6)|
| `--n-gen-seqs`                                  | Number of sequences to generate (default: 0, which auto-calculates as min(max_gen_seqs, 3 * len(data_seqs))) |
| `--epochs`                                      | Number of training epochs (default: 50)         |
| `--batch-size`                                  | Batch size for training (default: 5000)         |
| `--validation-split`                            | Validation split ratio (default: 0.2)           |
| `--infile-gen`                                  | Path to pre-generated sequences file (optional). If provided, uses these sequences instead of generating new ones. |
| `--junction-column`                             | Column name for junction sequences (default: `junction_aa`)|
| `--v-gene-column`                               | Column name for V gene (default: `v_gene`)      |
| `--j-gene-column`                               | Column name for J gene (default: `j_gene`)      |
| `--no-header`                                   | Input file does not have a header               |
| `--delimiter`                                   | File delimiter (default: `auto`, inferred from file extension) |

#### `sonnia evaluate` options

| Option                                          | Description                                      |
|-------------------------------------------------|--------------------------------------------------|
| `-i, --infile`                                  | Path to input file (required)                   |
| `--model`                                       | Model name or path to model folder (required)   |
| `-o, --outfile`                                 | Output file path (default: `evaluated_seqs.tsv`)|
| `-m, --max_seqs`                                | Maximum number of sequences to evaluate (default: 1e8) |
| `--paired`                                      | Use paired-chain model. Assumes heavy and light chains are in separate columns named `junction_aa_heavy`, `v_gene_heavy`, `j_gene_heavy`, `junction_aa_light`, `v_gene_light`, `j_gene_light`. |
| `--junction-column`                             | Column name for junction sequences (default: `junction_aa`, single chain only) |
| `-v, --v-gene-column`                           | Column name for V gene (default: `v_gene`, single chain only) |
| `-j, --j-gene-column`                           | Column name for J gene (default: `j_gene`, single chain only) |
| `--no-header`                                   | Input file does not have a header               |
| `-d, --delimiter`                               | File delimiter (default: `auto`, inferred from file extension) |

#### `sonnia generate` options

| Option                                          | Description                                      |
|-------------------------------------------------|--------------------------------------------------|
| `--model`                                       | Model name or path to model folder (required)   |
| `-n, --number_of_seqs`                          | Number of sequences to generate (required)      |
| `-o, --outfile`                                 | Output file path (optional; prints to stdout if not specified)|
| `--pre`                                         | Generate sequences using pre-selection model (required: either `--pre` or `--post` must be specified) |
| `--post`                                        | Generate sequences using post-selection model (required: either `--pre` or `--post` must be specified) |
| `--rejection-bound`                             | Rejection bound for post-selection (default: 10)|
| `--chunk-size`                                  | Chunk size for generation (default: 1000)      |
| `--paired`                                      | Use paired-chain model                          |
| `--junction-column`                             | Column name for junction sequences (default: `junction_aa`) |
| `--v-gene-column`                               | Column name for V gene (default: `v_gene`)      |
| `--j-gene-column`                               | Column name for J gene (default: `j_gene`)      |
| `--no-header`                                   | Input file does not have a header               |
| `--delimiter`                                   | File delimiter (default: `auto`, inferred from file extension) |

For detailed help on any command, use:
```bash
sonnia <command> --help
```
