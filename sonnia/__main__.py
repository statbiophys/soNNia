import os
import typer
import pandas as pd
from sonnia.sonia import Sonia
from sonnia.sonnia import SoNNia
from sonnia.sonia_paired import SoniaPaired
from sonnia.sonnia_paired import SoNNiaPaired
from sonnia.utils import load_sequences, chunks
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
    "humanTCR": "human_T_beta_alpha",
    "humanIGHK": "human_B_heavy_kappa",
    "humanIGHL": "human_B_heavy_lambda",
}

@app.command()
def infer(
    infile: str = typer.Option(..., "--infile", "-i", help="Path to input file"),
    model: Optional[str] = typer.Option(None, help=(f"Select model. If any from {list(MODELS.values())} then use default pgen model. Otherwise, specify custom pgen model folder.")),
    outdir: str = typer.Option("sonnia_model", "--outdir", "-o", help="Path to output directory"),
    linear: bool = typer.Option(False, help="Infer linear model"),
    max_seqs: int = typer.Option(int(1e8), help="Maximum number of sequences to evaluate"),
    max_gen_seqs: int = typer.Option(int(1e6), help="Maximum number of sequences to generate"),
    n_gen_seqs: int = typer.Option(0, help="Number of sequences to generate"),
    epochs: int = typer.Option(50, help="Number of epochs to train model"),
    batch_size: int = typer.Option(int(5e3), help="Batch size for training model"),
    validation_split: float = typer.Option(0.2, help="Validation split for training model"),
    infile_gen: Optional[str] = typer.Option(None, help="Path to input file for generated sequences"),
    junction_column: str = typer.Option("junction_aa", help="Column name for junction sequences"),
    v_gene_column: str = typer.Option("v_gene", help="Column name for V gene sequences"),
    j_gene_column: str = typer.Option("j_gene", help="Column name for J gene sequences"),
    paired: bool = typer.Option(False, help="Use paired chain model. Assumes heavy and light chains are in separate columns named junction_aa_heavy, v_gene_heavy, j_gene_heavy, junction_aa_light, v_gene_light, j_gene_light."),
    no_header: bool = typer.Option(False, help="Input file does not have a header"),
    delimiter: str = typer.Option("auto", help="Delimiter for input file. By default, it is inferred from the file extension."),
):
    """Infer a generative model on input sequences.

    Args:
        infile: Path to input file containing sequences to infer model on
        model: Model to use for inference. Can be a default model name from MODELS or path to custom model
        outdir: Path to output directory
        linear: Whether to infer linear model
        max_seqs: Maximum number of sequences to infer model on
        recompute_norm: Whether to recompute productive normalization
        n_gen_seqs: Number of sequences to generate
        epochs: Number of epochs to train model
        batch_size: Batch size for training model
        validation_split: Validation split for training model
        infile_gen: Path to input file for generated sequences
        junction_column: Column name for junction sequences
        v_gene_column: Column name for V gene sequences
        j_gene_column: Column name for J gene sequences
        paired: Whether to use paired chain model. If True, expects heavy/light chain columns
        no_header: Whether input file has no header
        delimiter: Delimiter for input file. Auto-detected from extension if "auto"
    """
    # Load input data
    typer.echo("Loading input data...")
    data_seqs = load_sequences(infile, delimiter, no_header, paired, junction_column, v_gene_column, j_gene_column)
    typer.echo(f'Successfully loaded {len(data_seqs)} sequences')

    if len(data_seqs) > max_seqs:
        typer.echo(f"WARNING: {len(data_seqs)} sequences found in input file, but max_seqs is set to {max_seqs}. Truncating to {max_seqs} sequences.")
        data_seqs = data_seqs[:max_seqs]
    # define number of gen_seqs:
    gen_seqs = []
    generate_sequences = False
    if infile_gen is None:
        generate_sequences = True
        if n_gen_seqs == 0:
            n_gen_seqs = np.min([max_gen_seqs, 3 * len(data_seqs)])
    else:
        gen_seqs = load_sequences(infile_gen, delimiter, no_header, paired, junction_column, v_gene_column, j_gene_column)

    typer.echo(f"Initializing model from {model}")
    if not linear:
        if paired:
            sonia_model = SoNNiaPaired(pgen_model=model,data_seqs=data_seqs,gen_seqs=gen_seqs)
        else:
            sonia_model = SoNNia(pgen_model=model,data_seqs=data_seqs,gen_seqs=gen_seqs)
    else:
        if paired:
            sonia_model = SoniaPaired(pgen_model=model,data_seqs=data_seqs,gen_seqs=gen_seqs)
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
def evaluate(
    infile: str = typer.Option(..., "--infile", "-i", help="Path to input file"),
    model: Optional[str] = typer.Option(..., "--model", help=f"Select model. If any from {MODELS.values()} then use default model. Otherwise, specify custom model folder."),
    outfile: str = typer.Option("evaluated_seqs.tsv", "--outfile", "-o", help="Path to output file"),
    max_seqs: int = typer.Option(int(1e8), "--max_seqs", "-m", help="Maximum number of sequences to evaluate"),
    junction_column: str = typer.Option("junction_aa", help="Column name for junction sequences. Only for single chain model."),
    v_gene_column: str = typer.Option("v_gene", "--v-gene-column", "-v", help="Column name for V gene sequences. Only for single chain model."),
    j_gene_column: str = typer.Option("j_gene", "--j-gene-column", "-j", help="Column name for J gene sequences. Only for single chain model."),
    paired: bool = typer.Option(False, "--paired", help="Use paired chain model. Assumes heavy and light chains are in separate columns named junction_aa_heavy, v_gene_heavy, j_gene_heavy, junction_aa_light, v_gene_light, j_gene_light."),
    no_header: bool = typer.Option(False, "--no-header", help="Input file does not have a header"),
    delimiter: str = typer.Option("auto", "--delimiter", "-d", help="Delimiter for input file. By default, it is inferred from the file extension."),
):
    """Evaluate sequences using a generative model.

    Args:
        infile: Path to input file containing sequences to evaluate
        model: Model to use for evaluation. Can be a default model name from MODELS or path to custom model
        outfile: Path to save evaluation results
        max_seqs: Maximum number of sequences to evaluate
        junction_column: Column name for CDR3 junction sequences (single chain only)
        v_gene_column: Column name for V gene sequences (single chain only) 
        j_gene_column: Column name for J gene sequences (single chain only)
        paired: Whether to use paired chain model. If True, expects heavy/light chain columns
        no_header: Whether input file has no header row
        delimiter: Delimiter for input file. Auto-detected from extension if "auto"
    """

    # Initialize model
    typer.echo(f"Initializing model from {model}")
    if paired:
        try:
            sonia_model = SoNNiaPaired(ppost_model=model)
        except:
            sonia_model = SoniaPaired(ppost_model=model)
    else:
        try:
            sonia_model = SoNNia(ppost_model=model)
        except:
            sonia_model = Sonia(ppost_model=model)
    
    # Load input data
    typer.echo("Loading input data...")
    data_seqs = load_sequences(infile, delimiter, no_header, paired, junction_column, v_gene_column, j_gene_column)
    typer.echo(f'Successfully loaded {len(data_seqs)} sequences')
    if pd.isna(data_seqs[:,1]).all() and pd.isna(data_seqs[:,2]).all():
        typer.echo("No V or J genes found in the data.")
        include_genes=False
    else:
        include_genes=True

    # Evaluate sequences
    typer.echo("Evaluating sequences...")
    sonia_model.update_model(add_data_seqs=data_seqs[:max_seqs])
    Q, pgen, ppost = sonia_model.evaluate_seqs(sonia_model.data_seqs.values, include_genes=include_genes)
    if paired:
        df_out = pd.DataFrame(sonia_model.data_seqs, columns=['junction_aa_heavy', 'v_gene_heavy', 'j_gene_heavy', 'junction_aa_light', 'v_gene_light', 'j_gene_light','sequence_id'])
    else:
        df_out = pd.DataFrame(sonia_model.data_seqs, columns=['junction_aa', 'v_gene', 'j_gene','sequence_id'])
    df_out['Q'] = Q
    df_out['Pgen'] = pgen
    df_out['Ppost'] = ppost
    
    # Save results
    typer.echo(f"Saving output to {outfile}")
    df_out.to_csv(outfile, sep="\t" if ".tsv" in outfile else "," if ".csv" in outfile else ";", index=False)

@app.command()
def generate(
    model: str = typer.Option(..., help=f"Select model. If any from {MODELS.values()} then use default model. Otherwise, specify custom model folder."),
    number_of_seqs: int = typer.Option(..., "--number_of_seqs", "-n", help="Number of sequences to generate"),
    outfile: str = typer.Option(None, "--outfile", "-o", help="Path to output file"),
    pgen: bool = typer.Option(False, "--pre", help="Generate sequences using pre-selection model"),
    ppost: bool = typer.Option(False,"--post", help="Generate sequences using post-selection model"),
    rejection_bound: int = typer.Option(10, help="Rejection bound for post-selection model"),
    chunk_size: int = typer.Option(1000, help="Chunk size for generating sequences"),
    junction_column: str = typer.Option("junction_aa", help="Column name for junction sequences"),
    v_gene_column: str = typer.Option("v_gene", help="Column name for V gene sequences"),
    j_gene_column: str = typer.Option("j_gene", help="Column name for J gene sequences"),
    paired: bool = typer.Option(False, help="Use paired chain model."),
    no_header: bool = typer.Option(False, help="Input file does not have a header"),
    delimiter: str = typer.Option("auto", help="Delimiter for input file. By default, it is inferred from the file extension."),
):
    """Generate sequences using the model.
    
    Args:
        model: Model to use for generation. Can be a default model name from MODELS or path to custom model
        number_of_seqs: Number of sequences to generate
        outfile_name: Path to output file
        pgen: Whether to generate sequences using pre-selection model
        ppost: Whether to generate sequences using post-selection model
        rejection_bound: Rejection bound for post-selection model
        chunk_size: Chunk size for generating sequences
        junction_column: Column name for junction sequences
        v_gene_column: Column name for V gene sequences
        j_gene_column: Column name for J gene sequences
        paired: Whether to use paired chain model. If True, expects heavy/light chain columns
        no_header: Whether input file has no header
        delimiter: Delimiter for input file. Auto-detected from extension if "auto"
    """

    # Initialize model
    typer.echo(f"Initializing model from {model}")
    if paired:
        try:
            sonia_model = SoNNiaPaired(ppost_model=model)
        except:
            sonia_model = SoniaPaired(ppost_model=model)
    else:
        try:
            sonia_model = SoNNia(ppost_model=model)
        except:
            sonia_model = Sonia(ppost_model=model)
    
    out_df = []
    to_generate = chunks(number_of_seqs, chunk_size)
    for t in to_generate:
        if pgen:
            seqs = sonia_model.generate_sequences_pre(num_seqs=t, nucleotide=True)
        elif ppost:
            seqs = sonia_model.generate_sequences_post(
                num_seqs=t, nucleotide=True, upper_bound=rejection_bound
            )
        else:
            raise RuntimeError("ERROR: give option between --pre or --post")
        
        if not outfile is  None:  # OUTFILE SPECIFIED
            if paired:
                out_df.append(pd.DataFrame(seqs,columns=['junction_aa_heavy', 'v_gene_heavy', 'j_gene_heavy', 'junction_aa_light', 'v_gene_light', 'j_gene_light','junction_heavy','junction_light']))
            else:
                out_df.append(pd.DataFrame(seqs,columns=[junction_column,v_gene_column,j_gene_column,'junction']))
        else:  # print to stdout
            if paired:
                print('junction_aa_heavy', 'v_gene_heavy', 'j_gene_heavy', 'junction_aa_light', 'v_gene_light', 'j_gene_light','junction_heavy','junction_light')
            else:
                print(junction_column,v_gene_column,j_gene_column,'junction')
            for seq in seqs:
                print(*seq)

    if outfile is not None: 
        pd.concat(out_df).to_csv(outfile,
            sep="\t" if ".tsv" in outfile else "," if ".csv" in outfile else ";",
            index=False
        )

if __name__ == "__main__":
    app()
