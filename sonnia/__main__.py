import os
import typer
import pandas as pd
from sonnia.sonia import Sonia
from sonnia.sonnia import SoNNia
from typing import Optional
from sonnia.plotting import Plotter
import numpy as np
app = typer.Typer(add_completion=False)

MODELS = {
    "humanTRA": "human_T_alpha",
    "humanTRB": "human_T_beta",
    "humanIGH": "human_B_heavy",
    "humanIGK": "human_B_kappa",
    "humanIGL": "human_B_lambda",
    "mouseTRA": "mouse_T_alpha",
    "mouseTRB": "mouse_T_beta",
    "mouseIGH": "mouse_B_heavy",
}

@app.command()
def evaluate(
    infile: str = typer.Option(..., "--infile", "-i", help="Path to input file"),
    model: Optional[str] = typer.Option(..., "--model", help=f"Select model. If any from {MODELS.values()} then use default model. Otherwise, specify custom model folder."),
    outfile: str = typer.Option("evaluated_seqs.tsv", "--outfile", "-o", help="Path to output file"),
    max_seqs: int = typer.Option(int(1e8), "--max_seqs", "-m", help="Maximum number of sequences to evaluate"),
    junction_column: str = typer.Option("junction_aa", "--junction_column", "-j", help="Column name for junction sequences"),
    v_gene_column: str = typer.Option("v_gene", "--v_gene_column", "-v", help="Column name for V gene sequences"),
    j_gene_column: str = typer.Option("j_gene", "--j_gene_column", "-j", help="Column name for J gene sequences"),
):
    """Infer model on input sequences."""
    
    # Determine delimiter
    delimiter = "\t" if ".tsv" in infile else "," if ".csv" in infile else ";"
    
    # Initialize model
    typer.echo(f"Initializing model from {model}")
    try:
        sonia_model = SoNNia(ppost_model=model)
    except:
        sonia_model = Sonia(ppost_model=model)
    
    # Load input data
    typer.echo("Loading input data...")
    try:
        data_seqs = pd.read_csv(infile, delimiter=delimiter)[[junction_column, v_gene_column, j_gene_column]].values
    except:
        data_seqs = pd.read_csv(infile, delimiter=delimiter, header=None).values
    
    if pd.isna(data_seqs[:,1]).all() and pd.isna(data_seqs[:,2]).all():
        typer.echo("No V or J genes found in the data.")
        include_genes=False
    else:
        include_genes=True

    # Evaluate sequences
    typer.echo("Evaluating sequences...")
    sonia_model.update_model(add_data_seqs=data_seqs[:max_seqs])
    Q, pgen, ppost = sonia_model.evaluate_seqs(sonia_model.data_seqs, include_genes=include_genes)
    df_out = pd.DataFrame(sonia_model.data_seqs, columns=['junction_aa', 'v_gene', 'j_gene','sequence_id'])
    df_out['Q'] = Q
    df_out['Pgen'] = pgen
    df_out['Ppost'] = ppost
    
    # Save results
    typer.echo(f"Saving output to {outfile}")
    df_out.to_csv(outfile, sep="\t" if ".tsv" in outfile else "," if ".csv" in outfile else ";", index=False)

@app.command()
def infer(
    infile: str = typer.Option(..., "--infile", "-i", help="Path to input file"),
    model: Optional[str] = typer.Option(None, "--model",
            help=(f"Select model. If any from {list(MODELS.values())} then use default pgen model. Otherwise, specify custom pgen model folder.")),
    outdir: str = typer.Option("sonnia_model", "--outdir", "-o", help="Path to output directory"),
    linear: bool = typer.Option(False, "--linear", help="Infer linear model"),
    max_seqs: int = typer.Option(int(1e8), "--max_seqs", "-m", help="Maximum number of sequences to evaluate"),
    recompute_norm: bool = typer.Option(False, "--recompute_productive_norm", help="Recompute productive normalization"),
    n_gen_seqs: int = typer.Option(0, "--n_gen_seqs", "-n", help="Number of sequences to generate"),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Number of epochs to train model"),
    batch_size: int = typer.Option(int(5e3), "--batch_size", "-b", help="Batch size for training model"),
    validation_split: float = typer.Option(0.2, "--validation_split", "-v", help="Validation split for training model"),
    infile_gen: Optional[str] = typer.Option(None, "--infile_gen", help="Path to input file for generated sequences"),
    junction_column: str = typer.Option("junction_aa", "--junction_column", "-j", help="Column name for junction sequences"),
    v_gene_column: str = typer.Option("v_gene", "--v_gene_column", "-v", help="Column name for V gene sequences"),
    j_gene_column: str = typer.Option("j_gene", "--j_gene_column", "-j", help="Column name for J gene sequences"),
):
    """Evaluate sequences using a generative model."""

    delimiter = "\t" if ".tsv" in infile else "," if ".csv" in infile else ";"

    # Load input data
    typer.echo("Loading input data...")
    try:
        data_seqs = pd.read_csv(infile, delimiter=delimiter)[[junction_column, v_gene_column, j_gene_column]].values.astype(str)
    except:
        data_seqs = pd.read_csv(infile, delimiter=delimiter, header=None).values.astype(str)
    typer.echo(f'Succesfully loaded {len(data_seqs)} sequences')

    # define number of gen_seqs:
    gen_seqs = []
    generate_sequences = False
    if infile_gen is None:
        generate_sequences = True
        if n_gen_seqs == 0:
            n_gen_seqs = len(data_seqs) #np.max([int(5e5), 3 * len(data_seqs)])
    else:
        gen_seqs = pd.read_csv(infile_gen, delimiter=delimiter)[[junction_column,'v_gene','j_gene']].values.astype(str)

    typer.echo(f"Initializing model from {model}")
    if not linear:
        sonia_model = SoNNia(pgen_model=model,data_seqs=data_seqs,gen_seqs=gen_seqs)
    else:
        sonia_model = Sonia(pgen_model=model,data_seqs=data_seqs,gen_seqs=gen_seqs)

    typer.echo("Recomputing productive normalization.")
    sonia_model.norm_productive = sonia_model.pgen_model.compute_regex_CDR3_template_pgen("CX{0,}")

    if generate_sequences:
        sonia_model.add_generated_seqs(n_gen_seqs)

    typer.echo("Model initialised. Start inference")
    sonia_model.infer_selection(epochs=epochs,verbose=1,batch_size=batch_size,validation_split=validation_split)
    
    typer.echo("Save Model")
    sonia_model.save_model(outdir)
    pl = Plotter(sonia_model)
    pl.plot_model_learning(os.path.join(outdir, "model_learning.png"))
    pl.plot_vjl(os.path.join(outdir, "marginals.png"))
    pl.plot_logQ(os.path.join(outdir, "log_Q.png"))
    pl.plot_ratioQ(os.path.join(outdir, "Q_ratio.png"))

@app.command()
def generate(
    model: str = typer.Option(..., "--model", help=f"Select model. If any from {MODELS.values()} then use default model. Otherwise, specify custom model folder."),
    number_of_seqs: int = typer.Option(..., "-n", help="Number of sequences to generate"),
    outfile_name: str = typer.Option(None, "--outfile", "-o", help="Path to output file"),
    pgen: bool = typer.Option(False, "--pre", help="Generate sequences using pre-selection model"),
    ppost: bool = typer.Option(False, "--post", help="Generate sequences using post-selection model"),
    rejection_bound: int = typer.Option(10, "--rejection_bound", "-r", help="Rejection bound for post-selection model"),
    chunck_size: int = typer.Option(1000, "--chunck_size", "-c", help="Chunck size for generating sequences"),
    junction_column: str = typer.Option("junction_aa", "--junction_column", "-j", help="Column name for junction sequences"),
    v_gene_column: str = typer.Option("v_gene", "--v_gene_column", "-v", help="Column name for V gene sequences"),
    j_gene_column: str = typer.Option("j_gene", "--j_gene_column", "-j", help="Column name for J gene sequences"),
):
    """Generate sequences using the model"""
    
    def chuncks(n, size):
        if n % size:
            return int(n / size) * [size] + [n % size]
        else:
            return int(n / size) * [size]

    # Initialize model
    typer.echo(f"Initializing model from {model}")
    try:
        sonia_model = SoNNia(ppost_model=model)
    except:
        sonia_model = Sonia(ppost_model=model)
    
    out_df = []
    to_generate = chuncks(number_of_seqs, chunck_size)
    for t in to_generate:
        if pgen:
            seqs = sonia_model.generate_sequences_pre(num_seqs=t, nucleotide=True)
        elif ppost:
            seqs = sonia_model.generate_sequences_post(
                num_seqs=t, nucleotide=True, upper_bound=rejection_bound
            )
        else:
            print("ERROR: give option between --pre or --post")
            return -1
        
        if outfile_name is not None:  # OUTFILE SPECIFIED
            out_df.append(pd.DataFrame(seqs,columns=[junction_column,v_gene_column,j_gene_column,'junction']))
        else:  # print to stdout
            print(junction_column,v_gene_column,j_gene_column,'junction')
            for seq in seqs:
                print(seq[0], seq[1], seq[2], seq[3])

    if outfile_name is not None: 
        pd.concat(out_df).to_csv(outfile_name,
            sep="\t" if ".tsv" in outfile_name else "," if ".csv" in outfile_name else ";",
            index=False
        )

if __name__ == "__main__":
    app()
